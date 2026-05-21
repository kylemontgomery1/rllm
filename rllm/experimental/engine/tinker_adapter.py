"""Create local rollout-engine handlers for the model gateway.

The handler is a plain ``async (dict) -> dict`` callable that translates
OpenAI-format request dicts into ``RolloutEngine.get_model_response()`` calls
and returns responses with embedded token IDs and logprobs in the format
expected by the gateway's ``data_process.py`` extractors.

This is used by local token-in/token-out engines such as Tinker and Fireworks,
eliminating an extra HTTP backend hop while preserving gateway trace capture.
"""

import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from rllm.experimental.rollout.rollout_engine import RolloutEngine

logger = logging.getLogger(__name__)


def _content_to_text(content: Any) -> str:
    """Normalize OpenAI content blocks for rollout engines that expect text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [_content_to_text(part) for part in content]
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        for key in ("text", "content"):
            value = content.get(key)
            if value is not None:
                return _content_to_text(value)
        return json.dumps(content)
    return str(content)


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        item = dict(message)
        if "content" in item:
            item["content"] = _content_to_text(item["content"])
        normalized.append(item)
    return normalized


def _to_openai_tool_calls(tool_calls: list) -> list[dict[str, Any]]:
    """Convert rLLM ToolCall objects to OpenAI-format tool_calls."""
    result = []
    for i, tc in enumerate(tool_calls):
        name = tc.name if hasattr(tc, "name") else tc.get("name", "")
        args = tc.arguments if hasattr(tc, "arguments") else tc.get("arguments", {})
        if isinstance(args, dict):
            args_str = json.dumps(args)
        else:
            args_str = str(args)
        result.append(
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            }
        )
    return result


def create_rollout_handler(engine: RolloutEngine) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Return an async handler that calls a rollout engine in-process.

    The returned callable accepts an OpenAI chat completion request dict and
    returns an OpenAI chat completion response dict with token extensions
    (``prompt_token_ids``, ``token_ids``, ``logprobs``) consistent with vLLM.
    """

    async def handler(request_body: dict[str, Any]) -> dict[str, Any]:
        messages = _normalize_messages(request_body.get("messages", []))
        tools = request_body.get("tools", [])

        kwargs: dict[str, Any] = {}
        if request_body.get("user") is not None:
            kwargs["application_id"] = request_body["user"]
        if tools:
            kwargs["tools"] = tools
        if request_body.get("temperature") is not None:
            kwargs["temperature"] = request_body["temperature"]
        if request_body.get("top_p") is not None:
            kwargs["top_p"] = request_body["top_p"]
        if request_body.get("top_k") is not None:
            kwargs["top_k"] = request_body["top_k"]
        if request_body.get("max_tokens") is not None:
            kwargs["max_tokens"] = request_body["max_tokens"]
        elif request_body.get("max_completion_tokens") is not None:
            kwargs["max_tokens"] = request_body["max_completion_tokens"]

        model_output = await engine.get_model_response(messages, **kwargs)

        response_text = model_output.content or ""
        prompt_ids = list(model_output.prompt_ids) if model_output.prompt_ids else []
        completion_ids = list(model_output.completion_ids) if model_output.completion_ids else []
        logprobs = model_output.logprobs or []
        routing_matrices = getattr(model_output, "routing_matrices", None) or []
        finish_reason = model_output.finish_reason or "stop"

        if routing_matrices and len(routing_matrices) != len(completion_ids):
            raise RuntimeError(
                f"routing_matrices length mismatch: {len(routing_matrices)} matrices vs "
                f"{len(completion_ids)} completion tokens"
            )

        logprob_content = [{"logprob": lp} for lp in logprobs]
        if routing_matrices:
            for entry, routing_matrix in zip(logprob_content, routing_matrices, strict=False):
                entry["routing_matrix"] = routing_matrix

        response_message: dict[str, Any] = {"role": "assistant", "content": response_text}
        if model_output.reasoning:
            response_message["reasoning"] = model_output.reasoning
        if model_output.tool_calls:
            response_message["tool_calls"] = _to_openai_tool_calls(model_output.tool_calls)
            if finish_reason == "stop":
                finish_reason = "tool_calls"

        prompt_len = model_output.prompt_length or len(prompt_ids)
        completion_len = model_output.completion_length or len(completion_ids)

        choice: dict[str, Any] = {
            "index": 0,
            "message": response_message,
            "finish_reason": finish_reason,
            "token_ids": completion_ids,
            "logprobs": {
                "content": logprob_content,
            },
        }
        if routing_matrices:
            choice["routing_matrices"] = list(routing_matrices)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_body.get("model", getattr(engine, "model_name", "default")),
            "choices": [choice],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": completion_len,
                "total_tokens": prompt_len + completion_len,
            },
            "prompt_token_ids": prompt_ids,
            "weight_version": getattr(model_output, "weight_version", None),
        }

    return handler


def create_tinker_handler(engine: RolloutEngine) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Backward-compatible alias for Tinker gateway setup."""
    return create_rollout_handler(engine)
