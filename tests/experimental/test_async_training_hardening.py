import asyncio

import pytest
from omegaconf import OmegaConf

from rllm.experimental.buffer import TrajectoryGroupBuffer
from rllm.experimental.common.config import (
    AlgorithmConfig,
    CompactFilteringConfig,
    RejectionSamplingConfig,
    TransformConfig,
)
from rllm.experimental.common.transform import transform_episodes_to_trajectory_groups
from rllm.experimental.metrics import MetricsAggregator
from rllm.experimental.sync_coordinator import SyncCoordinator, SyncCoordinatorConfig
from rllm.trainer.fireworks.fireworks_backend import FireworksBackend
from rllm.types import Episode
from rllm.workflows.workflow import TerminationReason


def _fireworks_config(loss_agg_mode):
    return OmegaConf.create(
        {
            "fuse_forward_backward_and_optim_step": False,
            "sampling": {
                "train": {"temperature": 1.0, "top_p": 1.0},
                "val": {"temperature": 1.0, "top_p": 1.0},
            },
            "rllm": {
                "algorithm": {
                    "loss_fn": "dapo",
                    "loss_agg_mode": loss_agg_mode,
                    "eps_clip_high": 0.28,
                    "kl_beta": 0.0,
                    "rollout_correction": {
                        "tis_mode": None,
                        "icepop_mode": None,
                        "icepop_beta": 2.0,
                        "bypass_mode": True,
                    },
                },
                "trainer": {"save_freq": -1},
                "async_training": {"enable": True, "trigger_parameter_sync_step": 1},
            },
        }
    )


def test_fireworks_accepts_dashed_loss_agg_modes():
    backend = FireworksBackend.__new__(FireworksBackend)

    for mode in (None, "token-mean", "seq-mean-token-sum", "seq-mean-token-mean"):
        backend.full_config = _fireworks_config(mode)
        backend.validate_config()


def test_fireworks_rejects_underscore_loss_agg_mode():
    backend = FireworksBackend.__new__(FireworksBackend)
    backend.full_config = _fireworks_config("seq_mean_token_sum")

    with pytest.raises(ValueError, match="loss_agg_mode"):
        backend.validate_config()


def test_transform_metrics_handle_all_filtered_groups():
    episodes = [
        Episode(id=f"task:{i}", trajectories=[], termination_reason=TerminationReason.TIMEOUT)
        for i in range(2)
    ]
    cf_config = CompactFilteringConfig(enable=True, mask_timeout=True)

    groups, metrics = transform_episodes_to_trajectory_groups(
        episodes,
        TransformConfig(),
        cf_config,
    )

    assert groups == []
    assert metrics["groups/num_groups"] == 0
    assert metrics["groups/num_trajs_after_filter"] == 0
    assert metrics["groups/avg_group_size"] == 0.0
    assert metrics["groups/max_group_size"] == 0
    assert metrics["groups/min_group_size"] == 0


def test_metrics_aggregator_accumulates_chunk_metrics_without_last_value_loss():
    aggregator = MetricsAggregator()
    aggregator.record("train/loss", 2.0, weight=10)
    aggregator.record("train/loss", 4.0, weight=30)
    aggregator.record("train/active_tokens", 10)
    aggregator.record("train/active_tokens", 30)
    aggregator.record("time/forward_backward", 3.0)
    aggregator.record("time/forward_backward", 5.0)
    aggregator.record("progress/batch", 7)
    aggregator.record("progress/batch", 8)

    metrics = aggregator.flush()

    assert metrics["train/loss"] == pytest.approx(3.5)
    assert metrics["train/active_tokens"] == 40.0
    assert metrics["time/forward_backward"] == 8.0
    assert metrics["progress/batch"] == 8.0


def test_metrics_aggregator_pools_distribution_statistics():
    aggregator = MetricsAggregator()
    aggregator.record_distribution("reward/search", [0.0, 1.0])
    aggregator.record_distribution("reward/search", [1.0, 1.0, 1.0])

    metrics = aggregator.flush()

    assert metrics["reward/search/mean"] == pytest.approx(0.8)
    assert metrics["reward/search/std"] == pytest.approx(0.4)
    assert metrics["reward/search/min"] == 0.0
    assert metrics["reward/search/max"] == 1.0


def test_buffer_all_filtered_group_decrements_in_flight_without_queueing_batch():
    async def run_test():
        coordinator = SyncCoordinator(
            SyncCoordinatorConfig(
                mini_batch_size=1,
                group_size=2,
                staleness_threshold=0.0,
                trigger_parameter_sync_step=1,
            )
        )
        aggregator = MetricsAggregator()
        buffer = TrajectoryGroupBuffer(
            group_size=2,
            coordinator=coordinator,
            aggregator=aggregator,
            algorithm_config=AlgorithmConfig(),
            transform_config=TransformConfig(),
            cf_config=CompactFilteringConfig(enable=True, mask_timeout=True),
            rs_config=RejectionSamplingConfig(),
        )

        coordinator.on_group_dispatched()
        assert await buffer.add_episode(
            "task",
            Episode(id="task:0", trajectories=[], termination_reason=TerminationReason.TIMEOUT),
        ) is False
        assert await buffer.add_episode(
            "task",
            Episode(id="task:1", trajectories=[], termination_reason=TerminationReason.TIMEOUT),
        ) is True

        assert coordinator.stats()["async/in_flight_groups"] == 0
        metrics = aggregator.flush()
        assert metrics["groups/num_groups"] == 0
        assert metrics["groups/dropped_min_trajs"] == 0

        buffer.mark_generation_complete()
        assert await buffer.get() is None

    asyncio.run(run_test())


def test_sync_coordinator_surfaces_background_task_errors():
    async def run_test():
        coordinator = SyncCoordinator(
            SyncCoordinatorConfig(
                mini_batch_size=1,
                group_size=1,
                staleness_threshold=0.0,
                trigger_parameter_sync_step=1,
            )
        )

        async def fail():
            raise ValueError("boom")

        task = asyncio.create_task(fail())
        coordinator.track_task(task)
        await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="Async rollout task failed") as exc_info:
            await coordinator.wait_for_drain()

        assert isinstance(exc_info.value.__cause__, ValueError)

    asyncio.run(run_test())
