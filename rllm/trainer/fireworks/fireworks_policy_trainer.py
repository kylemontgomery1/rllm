"""Policy training module for Firetitan-based RL.

Uses ``FiretitanTrainingClient`` / ``ReconnectableClient`` from the
Fireworks training SDK instead of Tinker's ``ServiceClient``.

This module handles gradient updates, model checkpointing, and data processing.
It does NOT contain any environment or agent logic.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

import tinker
from training.utils.client import ReconnectableClient
from fireworks.training.sdk import WeightSyncer
from tinker.types import AdamParams

from rllm.agents.agent import TrajectoryGroup
from rllm.experimental.common import (
    AlgorithmConfig,
    CompactFilteringConfig,
    TransformConfig,
)
from rllm.trainer.tinker.tinker_policy_trainer import (
    compute_schedule_lr_multiplier,
    require_training_client,
)
from rllm.trainer.tinker.transform import transform_trajectory_groups_to_datums

logger = logging.getLogger(__name__)


class FireworksPolicyTrainer:
    """Handles policy updates via gradient descent using Fireworks Firetitan.

    This class handles:
    - Training client management (``ReconnectableClient`` with auto-reconnect)
    - Data processing (filtering, advantages, datum conversion)
    - Forward-backward passes
    - Optimizer steps
    - Checkpoint saving / loading
    - Weight syncing to an inference deployment (``WeightSyncer``)

    It does NOT handle:
    - Environment or agent interactions
    - Trajectory collection
    - Sampling
    """

    _METRIC_SKIP_KEYS = {"step_id", "step"}

    def __init__(
        self,
        config,
        training_client: ReconnectableClient,
        reference_client: ReconnectableClient | None = None,
        weight_syncer: WeightSyncer | None = None,
        cf_config: CompactFilteringConfig | None = None,
        transform_config: TransformConfig | None = None,
        algorithm_config: AlgorithmConfig | None = None,
    ):
        """
        Args:
            config: Training configuration (OmegaConf).
            training_client: ``ReconnectableClient`` wrapping the policy
                ``FiretitanTrainingClient``.
            reference_client: Optional ``ReconnectableClient`` for the
                reference model (KL penalty, etc.).
            weight_syncer: ``WeightSyncer`` for pushing checkpoints to
                the inference deployment.
            cf_config: Compact filtering configuration.
            transform_config: Transform configuration.
            algorithm_config: Algorithm configuration.
        """
        self.config = config
        self.training_client = training_client
        self.reference_client = reference_client
        self.weight_syncer = weight_syncer

        self.cf_config = cf_config or CompactFilteringConfig.from_config(self.config.rllm.compact_filtering)
        self.transform_config = transform_config or TransformConfig()
        self.algorithm_config = algorithm_config or AlgorithmConfig.from_config(self.config)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def initialize_async(
        self,
        resume_from_checkpoint: bool = True,
        hot_load_before_training: bool = False,
    ) -> int:
        """Initialize or resume training.

        Handles checkpoint resume via ``FiretitanTrainingClient.list_checkpoints``
        and ``load_state_with_optimizer``.

        Args:
            resume_from_checkpoint: If True, attempt to resume from the
                last DCP checkpoint.
            hot_load_before_training: If True, push initial weights to the
                inference deployment before the first training step.

        Returns:
            The starting global step (0 when training from scratch).
        """
        start_step = 0

        if resume_from_checkpoint:
            start_step = await self._try_resume()

        if start_step == 0:
            logger.info("Starting training from scratch with model: %s", self.config.model.name)
            if hot_load_before_training:
                await self._initial_weight_sync()

        return start_step

    async def _try_resume(self) -> int:
        """Attempt to resume from a DCP checkpoint.

        Returns:
            The step to resume from, or 0 if no checkpoint was found.
        """
        inner = self.training_client.inner
        checkpoints = inner.list_checkpoints()
        if not checkpoints:
            logger.info("No existing checkpoints found.")
            return 0

        latest_name = checkpoints[-1]
        logger.info("Resuming from checkpoint: %s", latest_name)

        checkpoint_ref = inner.resolve_checkpoint_path(latest_name)
        await asyncio.to_thread(self.training_client.load_state_with_optimizer, checkpoint_ref)

        try:
            step = int(latest_name.split("-")[-1])
        except (ValueError, IndexError):
            step = 0

        await self._sync_weights(f"resume-{step}")
        return step

    async def _initial_weight_sync(self) -> None:
        """Push initial base weights to the inference deployment."""
        await self._sync_weights("step-0-base", checkpoint_type="base")

    async def _sync_weights(self, name: str, checkpoint_type: str | None = None) -> None:
        """Save sampler weights and hot-load them into the deployment."""
        if self.weight_syncer is None:
            return
        await asyncio.to_thread(
            self.weight_syncer.save_and_hotload,
            name,
            checkpoint_type=checkpoint_type,
        )
        logger.info("Weights synced to deployment: %s", name)

    # ------------------------------------------------------------------
    # Forward-backward
    # ------------------------------------------------------------------

    def _remove_mask(self, datum: tinker.Datum) -> tinker.Datum:
        """Remove mask from datum (not needed by forward_backward)."""
        return tinker.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
        )

    @staticmethod
    def _strip_routing_matrices(datum: tinker.Datum) -> tinker.Datum:
        """Strip routing matrices from a datum (reference model uses its own routing)."""
        mi = datum.model_input
        if getattr(mi, 'routing_matrices', None) is not None:
            mi = mi.model_copy(update={"routing_matrices": None})
            return tinker.Datum(model_input=mi, loss_fn_inputs=datum.loss_fn_inputs)
        return datum

    # ------------------------------------------------------------------
    # Custom loss helpers (cookbook callable path)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_inf_logprobs_and_prompt_lens(
        datums: list[tinker.Datum],
    ) -> tuple[list[list[float]], list[int]]:
        """Extract inference logprobs and prompt lengths from datums.

        Inference logprobs are stored in ``datum.loss_fn_inputs["logprobs"]``.
        Prompt length is derived from the mask (index of first nonzero element).
        """
        inf_logprobs: list[list[float]] = []
        prompt_lens: list[int] = []
        for datum in datums:
            lp = datum.loss_fn_inputs["logprobs"].data
            inf_logprobs.append(list(lp))

            mask = datum.loss_fn_inputs["mask"].data
            prompt_len = len(mask) + 1
            for i, m in enumerate(mask):
                if m != 0:
                    prompt_len = i + 1
                    break
            prompt_lens.append(prompt_len)
        return inf_logprobs, prompt_lens

    @staticmethod
    def _extract_scalar_advantages(datums: list[tinker.Datum]) -> list[float]:
        """Extract one scalar advantage per datum.

        rllm stores per-token advantages (broadcast for GRPO). The cookbook
        expects one scalar per datum — take the first response-token value.
        """
        advantages: list[float] = []
        for datum in datums:
            adv_data = datum.loss_fn_inputs["advantages"].data
            mask_data = datum.loss_fn_inputs["mask"].data
            # Find first response token (first nonzero mask position)
            scalar = 0.0
            for i, m in enumerate(mask_data):
                if m != 0:
                    scalar = float(adv_data[i])
                    break
            advantages.append(scalar)
        return advantages

    @staticmethod
    def _prepare_datum_for_custom_loss(datum: tinker.Datum) -> tinker.Datum:
        """Prepare a datum for the cookbook callable loss path.

        Renames ``mask`` → ``loss_mask`` (cookbook's ``_get_loss_mask`` reads
        ``loss_fn_inputs["loss_mask"]``) and removes fields the callable
        computes internally (``advantages``, ``logprobs``).
        """
        new_inputs = {}
        for k, v in datum.loss_fn_inputs.items():
            if k == "mask":
                new_inputs["loss_mask"] = v
            elif k in ("advantages", "logprobs"):
                continue  # handled by the callable
            else:
                new_inputs[k] = v
        return tinker.Datum(model_input=datum.model_input, loss_fn_inputs=new_inputs)

    async def _compute_proximal_logprobs(
        self,
        datums: list[tinker.Datum],
    ) -> list[list[float]]:
        """Compute proximal (π_old) logprobs via policy.forward().

        Only called when ``bypass_mode=False`` (3-policy / decoupled PPO).
        """
        stripped = [self._remove_mask(d) for d in datums]
        prox_fwd = await asyncio.to_thread(
            self.training_client.forward, stripped, "cross_entropy",
        )
        return [out["logprobs"].data for out in prox_fwd.loss_fn_outputs]

    def _build_custom_loss(
        self,
        algorithm_config: AlgorithmConfig,
        advantages: list[float],
        ref_logprobs: list[list[float]],
        prompt_lens: list[int],
        inf_logprobs: list[list[float]],
        prox_logprobs: list[list[float]],
    ):
        """Build a cookbook callable loss function.

        Returns a callable ``(data, logprobs_list) -> (loss, metrics)``
        suitable for ``forward_backward_custom``.
        """
        from training.utils.rl.losses import build_loss_fn
        from training.utils.rl.importance_sampling import ISConfig
        from training.utils.rl.dapo import DAPOConfig
        from training.utils.rl.gspo import GSPOConfig
        from training.utils.rl.cispo import CISPOConfig

        rc = algorithm_config.rollout_correction
        eps = algorithm_config.eps_clip
        eps_high = algorithm_config.eps_clip_high
        loss_fn_name = algorithm_config.loss_fn or "grpo"

        is_config = ISConfig(
            eps_clip=eps,
            eps_clip_high=eps_high,
            tis_cap=rc.tis_cap,
            tis_level=rc.tis_mode or "token",
        )
        dapo_config = DAPOConfig(
            eps_clip=eps,
            eps_clip_high=eps_high if eps_high is not None else 0.28,
        )
        gspo_config = GSPOConfig(
            clip_ratio=eps,
            clip_ratio_high=eps_high,
            kl_beta=algorithm_config.kl_beta,
        )
        cispo_config = CISPOConfig(
            eps_low=eps,
            eps_high=eps_high if eps_high is not None else 0.28,
        )

        builder = build_loss_fn(
            policy_loss=loss_fn_name,
            kl_beta=algorithm_config.kl_beta,
            dapo_config=dapo_config,
            gspo_config=gspo_config,
            cispo_config=cispo_config,
            is_config=is_config,
        )
        return builder(advantages, ref_logprobs, prompt_lens, inf_logprobs, prox_logprobs)

    # ------------------------------------------------------------------
    # Forward-backward
    # ------------------------------------------------------------------

    async def _build_loss_for_datums(
        self,
        datums: list[tinker.Datum],
        algorithm_config: AlgorithmConfig,
    ):
        """Compute proximal/ref logprobs and build a Fireworks cookbook callable loss.

        Fireworks always uses the cookbook callable path (``forward_backward_custom``
        only accepts callables, not string loss names).

        When ``bypass_mode=True`` (default), proximal logprobs are set to
        inference logprobs (no extra forward pass, TIS weight = 1.0).
        When ``bypass_mode=False``, a proximal forward pass is run for
        3-policy / decoupled PPO with active TIS correction.
        """
        rc = algorithm_config.rollout_correction
        inf_logprobs, prompt_lens = self._extract_inf_logprobs_and_prompt_lens(datums)
        advantages = self._extract_scalar_advantages(datums)

        # Proximal logprobs
        if rc.bypass_mode:
            prox_logprobs = inf_logprobs
        else:
            prox_logprobs = await self._compute_proximal_logprobs(datums)

        # Reference logprobs
        if algorithm_config.kl_beta > 0 and self.reference_client is not None:
            stripped = [self._remove_mask(d) for d in datums]
            if algorithm_config.router_replay:
                stripped = [self._strip_routing_matrices(d) for d in stripped]
            ref_logprobs = await self.compute_reference_logprobs(stripped)
        else:
            ref_logprobs = [[] for _ in datums]

        return self._build_custom_loss(
            algorithm_config, advantages, ref_logprobs, prompt_lens, inf_logprobs, prox_logprobs,
        )

    @require_training_client
    async def forward_backward_from_trajectory_groups(
        self,
        trajectory_groups: list[TrajectoryGroup],
        algorithm_config: AlgorithmConfig | None = None,
    ) -> tuple[list[tinker.Datum] | dict[str, list[tinker.Datum]], list[torch.Tensor], dict]:
        """Run forward-backward pass from trajectory groups.

        Always uses the Fireworks cookbook callable loss path. Optionally runs
        proximal and/or reference forward passes based on ``algorithm_config``.

        Args:
            trajectory_groups: List of TrajectoryGroup objects (already filtered/transformed).
            algorithm_config: Algorithm config for advantage computation
                (uses ``self.algorithm_config`` if None).

        Returns:
            ``(training_datums, training_logprobs, adv_metrics)``
        """
        if algorithm_config is None:
            algorithm_config = self.algorithm_config

        training_datums, adv_metrics = transform_trajectory_groups_to_datums(
            trajectory_groups,
            algorithm_config=algorithm_config,
        )

        loss_fn = await self._build_loss_for_datums(training_datums, algorithm_config)
        prepared = [self._prepare_datum_for_custom_loss(d) for d in training_datums]

        fwd_bwd_result = await asyncio.to_thread(
            self.training_client.forward_backward_custom, prepared, loss_fn,
        )

        training_logprobs = []
        for output in fwd_bwd_result.loss_fn_outputs:
            logprobs = output["logprobs"].to_torch()
            training_logprobs.append(logprobs)

        # Merge remote fwd/bwd metrics (e.g. loss) into adv_metrics
        if hasattr(fwd_bwd_result, "metrics") and fwd_bwd_result.metrics:
            for k, v in fwd_bwd_result.metrics.items():
                if k not in self._METRIC_SKIP_KEYS:
                    adv_metrics[f"train/{k}"] = v

        return training_datums, training_logprobs, adv_metrics

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    @require_training_client
    async def optim_step(
        self,
        step: int,
        total_steps: int,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        grad_clip_norm: float = 1.0,
    ) -> tuple[float, dict]:
        """Run optimizer step. Returns (scheduled_lr, metrics)."""
        scheduled_lr = learning_rate * compute_schedule_lr_multiplier(
            lr_schedule=self.algorithm_config.lr_schedule,
            warmup_steps_ratio=self.algorithm_config.warmup_steps_ratio,
            step=step,
            total_steps=total_steps,
        )

        adam_params = AdamParams(
            learning_rate=scheduled_lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
        )
        optim_result = await asyncio.to_thread(self.training_client.optim_step, adam_params)

        metrics = {}
        if hasattr(optim_result, "metrics") and optim_result.metrics:
            for k, v in optim_result.metrics.items():
                if k not in self._METRIC_SKIP_KEYS:
                    metrics[f"train/{k}"] = v

        return scheduled_lr, metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    @require_training_client
    async def sync_weights(self, step: int) -> None:
        """Hot-load current weights into the inference deployment."""
        await self._sync_weights(f"step-{step}")

    @require_training_client
    async def save_dcp_checkpoint(self, step: int) -> None:
        """Save a DCP checkpoint via WeightSyncer (includes dcp_timeout)."""
        name = f"step-{step}"
        try:
            await asyncio.to_thread(self.weight_syncer.save_dcp, name)
            logger.info("DCP checkpoint saved: %s", name)
        except Exception:
            logger.exception("Failed to save DCP checkpoint %s", name)
            raise

    # ------------------------------------------------------------------
    # Reference log-probs
    # ------------------------------------------------------------------

    @require_training_client
    async def compute_reference_logprobs(
        self,
        datums: list[tinker.Datum],
    ) -> list[list[float]]:
        """Compute reference log-probs for a batch of datums.

        Requires ``self.reference_client`` to be set.

        Returns:
            Per-datum list of per-token log-probs.
        """
        if self.reference_client is None:
            raise RuntimeError("reference_client not set")

        ref_fwd = await asyncio.to_thread(self.reference_client.forward, datums, "cross_entropy")
        return [out["logprobs"].data for out in ref_fwd.loss_fn_outputs]
