"""Compact rollout logger for inspection/debugging.

This intentionally omits token ids, logprobs, routing matrices, and model_output
payloads while preserving the chat transcript and episode-level metadata needed
to inspect rollouts.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from rllm.tools.tool_base import ToolCall, ToolOutput
from rllm.types import Episode


SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
TASK_METADATA_KEYS = (
    "id",
    "question",
    "ground_truth",
    "data_source",
    "parametric_acc",
    "requires_web_search",
    "requires_code_interpreter",
    "requires_multimodal",
    "requires_local_files",
    "is_verifiable",
    "difficulty",
    "is_english",
    "question_quality",
    "answer_quality",
    "question_type",
    "time_sensitive",
    "answer_correct",
    "answer_unique",
    "complexity",
    "source_obscurity",
)


def _safe_filename(value: Any, fallback: str = "unknown", max_len: int = 160) -> str:
    text = str(value) if value is not None else fallback
    text = SAFE_FILENAME_RE.sub("_", text).strip("._")
    if not text:
        text = fallback
    return text[:max_len]


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, ToolCall | ToolOutput):
        return value.to_dict()
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, list):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    return value


def _compact_error(info: dict[str, Any]) -> dict[str, Any] | None:
    error = info.get("error") if isinstance(info, dict) else None
    if not isinstance(error, dict):
        return None
    compact = {
        "error_type": error.get("error_type"),
        "error_message": error.get("error_message"),
    }
    traceback = error.get("traceback")
    if traceback:
        compact["traceback"] = str(traceback)[-4000:]
    return compact


def _split_episode_id(episode_id: str | None) -> tuple[str | None, int | None]:
    if not episode_id or ":" not in episode_id:
        return episode_id, None
    task_id, rollout_idx = episode_id.rsplit(":", 1)
    try:
        return task_id, int(rollout_idx)
    except ValueError:
        return task_id, None


def _last_messages(trajectory) -> list[dict[str, Any]]:
    if not trajectory.steps:
        return []
    return _as_jsonable(trajectory.steps[-1].chat_completions or [])


def _assistant_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [message for message in messages if message.get("role") == "assistant"]


def _final_answer(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return message.get("content")
    return None


def _tool_call_counts(messages: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for message in _assistant_messages(messages):
        for tool_call in message.get("tool_calls") or []:
            if isinstance(tool_call, dict):
                name = tool_call.get("name")
            else:
                name = getattr(tool_call, "name", None)
            if name:
                counts[str(name)] += 1
    return dict(sorted(counts.items()))


def compact_episode_to_dict(
    episode: Episode,
    step: int,
    mode: str = "train",
    epoch: int = 0,
) -> dict[str, Any]:
    episode_id = str(episode.id) if episode.id is not None else None
    engine_task_id, rollout_idx = _split_episode_id(episode_id)
    task = episode.task if isinstance(episode.task, dict) else {}
    dataset_task_id = task.get("id") or engine_task_id
    info = _as_jsonable(episode.info or {})

    task_data = {key: task.get(key) for key in TASK_METADATA_KEYS if key in task}
    for key, value in task.items():
        if key not in task_data and key not in {"image", "images"}:
            if key not in {"question", "ground_truth"}:
                task_data[key] = value

    trajectories = []
    rewards = []
    for trajectory in episode.trajectories:
        messages = _last_messages(trajectory)
        reward = float(trajectory.reward) if trajectory.reward is not None else None
        if reward is not None:
            rewards.append(reward)
        trajectories.append(
            {
                "name": trajectory.name,
                "uid": trajectory.uid,
                "reward": reward,
                "num_steps": len(trajectory.steps),
                "num_messages": len(messages),
                "num_assistant_turns": len(_assistant_messages(messages)),
                "tool_call_counts": _tool_call_counts(messages),
                "final_answer": _final_answer(messages),
                "messages": messages,
                "info": _as_jsonable(trajectory.info or {}),
            }
        )

    data = {
        "created_at_unix": time.time(),
        "pid": os.getpid(),
        "mode": mode,
        "epoch": epoch,
        "training_step": step,
        "episode_id": episode_id,
        "engine_task_id": engine_task_id,
        "dataset_task_id": dataset_task_id,
        "rollout_idx": rollout_idx,
        "task": task_data,
        "termination_reason": episode.termination_reason.value if episode.termination_reason else None,
        "is_correct": bool(episode.is_correct),
        "reward": sum(rewards) if rewards else None,
        "metrics": _as_jsonable(episode.metrics or {}),
        "judge": info.get("judge"),
        "error": _compact_error(info),
        "info": {k: v for k, v in info.items() if k not in {"judge", "error"}},
        "trajectories": trajectories,
    }
    return data


class CompactRolloutLogger:
    """Write one compact JSON file per rollout."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_step_dir(self, step: int, mode: str = "train", epoch: int = 0) -> Path:
        step_dir = self.base_dir / f"{mode}_epoch_{epoch}" / f"step_{step:08d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir

    def get_episode_path(self, episode: Episode, step: int, mode: str = "train", epoch: int = 0) -> Path:
        task = episode.task if isinstance(episode.task, dict) else {}
        dataset_task_id = task.get("id")
        episode_id = str(episode.id) if episode.id is not None else None
        engine_task_id, rollout_idx = _split_episode_id(episode_id)
        task_part = _safe_filename(dataset_task_id or engine_task_id, "task")
        episode_part = _safe_filename(episode_id, "episode")
        rollout_part = "unknown" if rollout_idx is None else str(rollout_idx)
        filename = f"task_{task_part}__rollout_{rollout_part}__episode_{episode_part}.json"
        return self.get_step_dir(step, mode, epoch) / filename

    def log_episode(self, episode: Episode, step: int, mode: str = "train", epoch: int = 0) -> Path:
        path = self.get_episode_path(episode, step, mode, epoch)
        if path.exists():
            stem = path.stem
            suffix = path.suffix
            counter = 1
            while path.exists():
                path = path.with_name(f"{stem}__dup_{counter}{suffix}")
                counter += 1

        data = compact_episode_to_dict(episode, step=step, mode=mode, epoch=epoch)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
            f.write("\n")
        tmp_path.replace(path)
        return path
