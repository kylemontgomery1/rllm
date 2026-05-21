"""
Common metric utilities for rLLM. Work with TrajectoryGroups and Episodes.
For backend-dependent metrics, please implement them in the backend-specific modules.
TODO(listar2000): think hard about what are the actually important metrics in agentic RL settings.
"""

from collections import defaultdict
from collections.abc import Callable

import numpy as np

from rllm.types import Episode, TrajectoryGroup


def reduce_metrics_by_trajectory_name(trajectory_groups: list[TrajectoryGroup], prefix: str = "reward", include_fraction_zero: bool = False) -> dict:
    """
    Reduce reward metrics by trajectory name.

    Args:
        trajectory_groups: List of TrajectoryGroup objects
        prefix: Prefix for the metric keys
        include_fraction_zero: Whether to include the fraction of zero summary
    Returns:
        Dictionary of metrics by trajectory name
    """
    metrics, rewards_by_traj_name = {}, {}
    for group in trajectory_groups:
        for traj in group.trajectories:
            rewards_by_traj_name[traj.name] = traj.reward

    for traj_name, rewards in rewards_by_traj_name.items():
        metrics[f"{prefix}/{traj_name}/mean"] = np.mean(rewards)
        metrics[f"{prefix}/{traj_name}/max"] = np.max(rewards)
        metrics[f"{prefix}/{traj_name}/min"] = np.min(rewards)
        metrics[f"{prefix}/{traj_name}/std"] = np.std(rewards)
        if include_fraction_zero:
            metrics[f"{prefix}/{traj_name}/fraction_zero"] = np.sum(np.abs(rewards) < 1e-8) / len(rewards)

    return metrics


def reduce_episode_solve_status_metrics(episodes: list[Episode], prefix: str = "episode") -> dict[str, float]:
    """Classify prompt-level rollout groups by episode correctness.

    A prompt group is fully solved if every rollout episode is correct,
    partially solved if at least one but not all are correct, and none solved
    if no rollout episode is correct.
    """
    episodes_by_task: dict[str, list[Episode]] = defaultdict(list)
    for episode in episodes:
        episodes_by_task[episode.task_id].append(episode)

    total = len(episodes_by_task)
    if total == 0:
        return {}

    fully_solved = 0
    partially_solved = 0
    none_solved = 0
    for task_episodes in episodes_by_task.values():
        correct_count = sum(1 for episode in task_episodes if episode.is_correct)
        if correct_count == len(task_episodes):
            fully_solved += 1
        elif correct_count > 0:
            partially_solved += 1
        else:
            none_solved += 1

    return {
        f"{prefix}/solve_all": fully_solved / total,
        f"{prefix}/solve_partial": partially_solved / total,
        f"{prefix}/solve_none": none_solved / total,
    }


def reduce_metrics_lists(metrics: dict, reduce_fn: Callable = np.mean):
    """A simple helper function that average any metric value that is a list into a scalar.

    Args:
        metrics: Dictionary of metrics
    """
    for key, value in metrics.items():
        if isinstance(value, list):
            metrics[key] = reduce_fn(value)
