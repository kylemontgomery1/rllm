"""Reward shaping helpers for trajectory-level RL training."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

from rllm.experimental.common.config import PostAdvantageFilteringConfig, RewardShapingConfig
from rllm.types import Trajectory, TrajectoryGroup
from rllm.workflows.workflow import TerminationReason


def _token_input_length(token_input) -> int:
    """Count flattened prompt tokens without importing backend-specific types."""
    total = 0
    for item in token_input or []:
        tokens = getattr(item, "tokens", None)
        if tokens is not None:
            total += len(tokens)
        else:
            total += 1
    return total


def total_response_tokens(trajectory: Trajectory) -> int:
    return sum(len(step.response_ids) for step in trajectory.steps)


def post_initial_context_tokens(trajectory: Trajectory) -> int:
    if not trajectory.steps:
        return 0
    initial_prompt_len = _token_input_length(trajectory.steps[0].prompt_ids)
    max_context_len = max(
        _token_input_length(step.prompt_ids) + len(step.response_ids)
        for step in trajectory.steps
    )
    return max(0, max_context_len - initial_prompt_len)


def length_tokens_for_trajectory(trajectory: Trajectory, include_intermediary_obs_tokens: bool) -> int:
    if not include_intermediary_obs_tokens:
        return total_response_tokens(trajectory)
    measured = post_initial_context_tokens(trajectory)
    return measured if measured > 0 else total_response_tokens(trajectory)


def apply_reward_shaping(groups: list[TrajectoryGroup], config: RewardShapingConfig) -> dict[str, float]:
    """Apply configured reward shaping in-place and return summary metrics."""
    if not config.enable or config.length_penalty_coef == 0.0:
        return {}
    if config.length_penalty_cap_tokens <= 0:
        raise ValueError("reward_shaping.length_penalty_cap_tokens must be > 0")
    if config.length_penalty_free_tokens < 0:
        raise ValueError("reward_shaping.length_penalty_free_tokens must be >= 0")
    if config.length_penalty_free_tokens >= config.length_penalty_cap_tokens:
        raise ValueError("reward_shaping.length_penalty_free_tokens must be < length_penalty_cap_tokens")

    base_rewards: list[float] = []
    shaped_rewards: list[float] = []
    length_tokens: list[float] = []
    penalties: list[float] = []

    for group in groups:
        for trajectory in group.trajectories:
            if trajectory.reward is None:
                continue
            base_reward = float(trajectory.info.get("base_reward", trajectory.reward))
            tokens = length_tokens_for_trajectory(trajectory, config.include_intermediary_obs_tokens)
            capped_tokens = min(tokens, config.length_penalty_cap_tokens)
            penalized_tokens = max(0, capped_tokens - config.length_penalty_free_tokens)
            penalty_window = config.length_penalty_cap_tokens - config.length_penalty_free_tokens
            penalty = config.length_penalty_coef * (penalized_tokens / penalty_window)
            shaped_reward = base_reward - penalty

            trajectory.info["base_reward"] = base_reward
            trajectory.info["length_penalty_tokens"] = float(tokens)
            trajectory.info["length_penalty"] = float(penalty)
            trajectory.reward = shaped_reward

            base_rewards.append(base_reward)
            shaped_rewards.append(shaped_reward)
            length_tokens.append(float(tokens))
            penalties.append(float(penalty))

    if not penalties:
        return {}

    return {
        "reward_shaping/base_reward/mean": mean(base_rewards),
        "reward_shaping/shaped_reward/mean": mean(shaped_rewards),
        "reward_shaping/length_tokens/mean": mean(length_tokens),
        "reward_shaping/length_tokens/max": max(length_tokens),
        "reward_shaping/penalty/mean": mean(penalties),
        "reward_shaping/penalty/max": max(penalties),
    }


def filter_uniform_base_reward_groups(groups: list[TrajectoryGroup], eps: float = 1e-8) -> tuple[list[TrajectoryGroup], int]:
    """Drop groups whose verifier/base rewards have no variance."""
    kept: list[TrajectoryGroup] = []
    dropped = 0
    for group in groups:
        rewards = [
            float(traj.info.get("base_reward", traj.reward))
            for traj in group.trajectories
            if traj.reward is not None
        ]
        if not rewards or max(rewards) - min(rewards) <= eps:
            dropped += 1
            continue
        kept.append(group)
    return kept, dropped


def post_advantage_filter_groups(
    groups: list[TrajectoryGroup],
    config: PostAdvantageFilteringConfig,
) -> tuple[list[TrajectoryGroup], dict[str, int]]:
    """Drop configured trajectories after advantages are computed."""
    if not config.enable:
        return groups, {}

    kept_groups: list[TrajectoryGroup] = []
    dropped_by_reason: dict[str, int] = defaultdict(int)
    dropped_groups = 0
    dropped_trajs = 0

    for group in groups:
        kept_trajectories = []
        kept_metadata = []
        for idx, trajectory in enumerate(group.trajectories):
            metadata = group.metadata[idx] if idx < len(group.metadata) else {}
            raw_reason = metadata.get("termination_reason") or TerminationReason.UNKNOWN
            try:
                reason = raw_reason if isinstance(raw_reason, TerminationReason) else TerminationReason(raw_reason)
            except ValueError:
                reason = TerminationReason.UNKNOWN
            if config.should_mask(reason):
                reason_value = reason.value if hasattr(reason, "value") else str(reason)
                dropped_by_reason[reason_value] += 1
                dropped_trajs += 1
                continue
            kept_trajectories.append(trajectory)
            kept_metadata.append(metadata)

        if kept_trajectories:
            group.trajectories = kept_trajectories
            group.metadata = kept_metadata
            kept_groups.append(group)
        else:
            dropped_groups += 1

    metrics = {
        "post_advantage_filter/dropped_trajs": dropped_trajs,
        "post_advantage_filter/dropped_groups": dropped_groups,
    }
    for reason, count in dropped_by_reason.items():
        metrics[f"post_advantage_filter/dropped_trajs/{reason}"] = count
    return kept_groups, metrics
