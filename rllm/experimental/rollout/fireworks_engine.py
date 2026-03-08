"""RolloutEngine backed by Fireworks ``DeploymentSampler``.

Inherits from ``TinkerEngine`` — the only differences are:

1. ``__init__``: creates a ``DeploymentSampler`` instead of requiring a
   ``tinker.ServiceClient``.  The sampler is stored as ``self.sampling_client``
   so that the inherited ``set_sampling_client`` / ``generate_episodes`` flow
   works unchanged.
2. ``get_token_output_from_token_input``: calls ``DeploymentSampler.completions``
   (token-in / token-out) and wraps the response in a ``SampledSequence``-compatible
   adapter so that the inherited ``assemble_model_output`` works unchanged.

Everything else — ``get_model_response``, ``assemble_model_output``,
``set_sampling_client``, ``_prepare_max_tokens``, chat-template rendering —
is inherited from ``TinkerEngine``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from fireworks.training.sdk import DeploymentSampler

from typing_extensions import override

from rllm.experimental.rollout.tinker_engine import (
    TinkerEngine,
    _flat_token_input_length,
)
from rllm.experimental.rollout.types import (
    TinkerTokenInput,
    TinkerTokenOutput,
    Tokenizer,
)
from rllm.workflows import TerminationEvent, TerminationReason

logger = logging.getLogger(__name__)

_MAX_SAMPLE_ATTEMPTS = 5
_TRANSIENT_ERROR_CODES = ("502", "503", "425", "Connection")


class _SampledSequenceAdapter:
    """Lightweight adapter so that a ``DeploymentSampler.completions`` response
    exposes the same ``.tokens``, ``.logprobs``, ``.stop_reason`` interface
    that ``tinker.SampledSequence`` (``TinkerTokenOutput``) provides."""

    __slots__ = ("tokens", "logprobs", "stop_reason")

    def __init__(
        self,
        tokens: list[int],
        logprobs: list[float] | None,
        stop_reason: str | None,
    ):
        self.tokens = tokens
        self.logprobs = logprobs
        self.stop_reason = stop_reason


class FireworksEngine(TinkerEngine):
    """``TinkerEngine`` subclass that uses a Fireworks ``DeploymentSampler``
    for inference instead of a Tinker ``SamplingClient``.

    ``DeploymentSampler`` supports token-in / token-out via the
    ``/inference/v1/completions`` endpoint, so ``TinkerTokenInput`` and
    ``TinkerTokenOutput`` are fully supported.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        sampler: DeploymentSampler,
        max_prompt_length: int = 4096,
        max_response_length: int = 4096,
        max_model_length: int = 32768,
        sampling_params: dict | None = None,
        disable_thinking: bool = False,
        accumulate_reasoning: bool = False,
        reasoning_effort: str = "medium",
        processor=None,
        **kwargs,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer for chat-template rendering.
            sampler: Pre-built ``DeploymentSampler``.  When provided,
                ``inference_url`` / ``model`` / ``api_key`` are ignored.
            inference_url: Fireworks inference base URL (used only when
                *sampler* is ``None``).
            model: Fully-qualified model / deployment name (used only when
                *sampler* is ``None``).
            api_key: Fireworks API key (used only when *sampler* is ``None``).
            max_prompt_length: Hard cap on prompt token length.
            max_response_length: Default max completion tokens.
            max_model_length: Total context window.
            sampling_params: Dict with optional ``"train"`` / ``"val"``
                sub-dicts for default sampling kwargs.
            disable_thinking: Suppress thinking tokens in the prompt.
            accumulate_reasoning: Accumulate reasoning across turns.
            reasoning_effort: Reasoning effort hint for the parser.
            processor: Optional ``ProcessorMixin`` for multimodal models.
        """
        from rllm.experimental.rollout.rollout_engine import RolloutEngine
        from rllm.parser import ChatTemplateParser

        # Skip TinkerEngine.__init__ (it requires tinker.ServiceClient);
        # set up the same attributes directly.
        RolloutEngine.__init__(self)

        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.max_model_length = (
            max_model_length - 1
            if max_model_length is not None
            else max_prompt_length + max_response_length - 1
        )
        self.bypass_render_with_parser = True
        self.accumulate_reasoning = accumulate_reasoning
        self.reasoning_effort = reasoning_effort

        self.train_sampling_params = dict((sampling_params or {}).get("train", {}))
        self.val_sampling_params = dict((sampling_params or {}).get("val", {}))

        # Not used by Fireworks, but kept so inherited helpers don't blow up
        self.service_client = None
        self.renderer = None

        # Chat template parser (same setup as TinkerEngine bypass mode)
        self.chat_parser = ChatTemplateParser.get_parser(
            tokenizer, processor=processor, disable_thinking=disable_thinking,
        )
        if hasattr(self.chat_parser, "stop_sequences") and self.chat_parser.stop_sequences:
            self.stop_sequences = self.chat_parser.stop_sequences
        elif hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id:
            self.stop_sequences = [tokenizer.eos_token_id]
        else:
            raise ValueError("No stop sequences found for tokenizer or chat parser")

        # DeploymentSampler — accept a pre-built instance
        self.sampler = sampler
        # Store as sampling_client so inherited set_sampling_client / guards work
        self.sampling_client = self.sampler

    # ------------------------------------------------------------------
    # Token-in / token-out override
    # ------------------------------------------------------------------

    @property
    def supports_token_in_token_out(self) -> bool:
        return True

    @override
    async def get_token_output_from_token_input(
        self, token_input: TinkerTokenInput, **kwargs
    ) -> TinkerTokenOutput:
        """Sample from the Fireworks deployment using pre-tokenized IDs.

        Returns a ``SampledSequence``-compatible object so that the inherited
        ``assemble_model_output`` works unchanged.
        """
        if self.sampling_client is None:
            raise RuntimeError("Sampling client not set. Call set_sampling_client() first.")

        input_length = _flat_token_input_length(token_input)

        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)
        if enforce_max_prompt_length and input_length > min(self.max_prompt_length, self.max_model_length):
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        # Flatten TinkerTokenInput to plain list[int]
        prompt_ids: list[int] = []
        for elem in token_input:
            if isinstance(elem, int):
                prompt_ids.append(elem)
            else:
                # tinker.EncodedTextChunk
                prompt_ids.extend(elem.tokens)

        sampling_params = self.val_sampling_params.copy() if self.is_validation else self.train_sampling_params.copy()
        requested_max_tokens = kwargs.pop("max_tokens", kwargs.pop("max_new_tokens", self.max_response_length))
        requested_max_tokens = sampling_params.pop("max_tokens", requested_max_tokens)
        max_tokens = self._prepare_max_tokens(requested_max_tokens, input_length)

        for key in ("temperature", "top_p", "top_k"):
            if key in kwargs:
                sampling_params[key] = kwargs[key]

        raw = await asyncio.to_thread(
            self._completions_with_retry,
            prompt_ids,
            max_tokens,
            sampling_params,
        )

        choice = raw["choices"][0]
        completion_ids: list[int] = list((choice.get("raw_output") or {}).get("completion_token_ids") or [])

        logprobs: list[float] | None = None
        lp_data = choice.get("logprobs")
        if lp_data and isinstance(lp_data, dict):
            content = lp_data.get("content")
            if isinstance(content, list) and content:
                logprobs = [tok.get("logprob", 0.0) for tok in content]

        finish_reason = choice.get("finish_reason", "stop")

        return _SampledSequenceAdapter(  # type: ignore[return-value]
            tokens=completion_ids,
            logprobs=logprobs,
            stop_reason=finish_reason,
        )

    # ------------------------------------------------------------------
    # Internal retry helper
    # ------------------------------------------------------------------

    def _completions_with_retry(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        sampling_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Call ``DeploymentSampler.completions`` with transient-error retries."""
        for attempt in range(_MAX_SAMPLE_ATTEMPTS):
            try:
                return self.sampler.completions(
                    prompt=prompt_ids,
                    n=1,
                    max_tokens=max_tokens,
                    raw_output=True,
                    logprobs=True,
                    top_logprobs=1,
                    **sampling_kwargs,
                )
            except Exception as exc:
                err = str(exc)
                transient = any(code in err for code in _TRANSIENT_ERROR_CODES)
                if transient and attempt < _MAX_SAMPLE_ATTEMPTS - 1:
                    wait = 10 * (attempt + 1)
                    logger.warning(
                        "Attempt %d/%d failed (%s), retrying in %ds…",
                        attempt + 1,
                        _MAX_SAMPLE_ATTEMPTS,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                logger.error(
                    "Sampling failed permanently after %d attempts: %s",
                    attempt + 1,
                    exc,
                )
                raise
        raise RuntimeError("unreachable")
                logger.error(
                    "Sampling failed permanently after %d attempts: %s",
                    attempt + 1,
                    exc,
                )
                raise
        raise RuntimeError("unreachable")
