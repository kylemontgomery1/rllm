import asyncio
import logging
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from rllm.agents.agent import Episode, TrajectoryGroup
from rllm.experimental.common.rejection_sampling import RejectionSamplingMetrics
from rllm.experimental.common.transform import transform_episodes_to_trajectory_groups
from rllm.experimental.tinker.tinker_backend import TinkerBackend, _build_interleave_batch
from rllm.experimental.tinker.transform import transform_trajectory_groups_to_datums
from rllm.experimental.unified_trainer import TrainerState, UnifiedTrainer

logger = logging.getLogger(__name__)

# Sentinel object to signal end of queue
_QUEUE_DONE = object()


class StreamingUnifiedTrainer(UnifiedTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self.backend, TinkerBackend):
            raise ValueError("StreamingUnifiedTrainer only supports TinkerBackend")

        if self.config.get("fuse_forward_backward_and_optim_step", False):
            raise ValueError(
                "fuse_forward_backward_and_optim_step is incompatible with streaming "
                "training."
            )

        if self.rs_config.mode == "episode":
            raise ValueError(
                "Rejection sampling is incompatible with streaming "
                "training. Use mode='none' with streaming."
            )

    async def _generate_episodes_streaming(
        self,
        batch: list[dict],
        queue: asyncio.Queue,
        is_validation: bool = False,
    ) -> None:
        try:
            backend: TinkerBackend = self.backend  # type: ignore[assignment]
            assert backend.rollout_engine is not None
            assert backend.sampling_client is not None

            backend.rollout_engine.set_sampling_client(backend.sampling_client)

            group_size = (
                self.config.rllm.rollout.n_val if is_validation
                else self.config.rllm.rollout.n
            )
            interleaved_batch = _build_interleave_batch(batch, group_size)
            task_ids = [item["uid"] for item in interleaved_batch]

            episodes_by_task: dict[str, list[Episode]] = defaultdict(list)
            expected_per_task: dict[str, int] = Counter(task_ids)

            # Spawn all rollout coroutines
            task_id_counter: defaultdict[str, int] = defaultdict(int)
            futures = []
            for idx, (task, task_id) in enumerate(zip(interleaved_batch, task_ids, strict=True)):
                rollout_idx = task_id_counter[task_id]
                futures.append(
                    self.agent_workflow_engine.process_task_with_retry(
                        task, task_id, rollout_idx, idx
                    )
                )
                task_id_counter[task_id] += 1

            # Collect results incrementally as they complete
            for future in asyncio.as_completed(futures):
                task_id, _, _, episode = await future
                episodes_by_task[task_id].append(episode)

                # When all rollouts for a task_id are done, push episodes to queue
                if len(episodes_by_task[task_id]) == expected_per_task[task_id]:
                    await queue.put(episodes_by_task[task_id])
        finally:
            await queue.put(_QUEUE_DONE)

    async def _train_batch_async(self, batch: Any, trainer_state: TrainerState) -> None:
        self.agent_workflow_engine.set_training_step(
            trainer_state.global_step, mode="train", epoch=trainer_state.epoch
        )

        backend: TinkerBackend = self.backend  # type: ignore[assignment]
        policy_trainer = backend.policy_trainer
        assert policy_trainer is not None

        queue: asyncio.Queue = asyncio.Queue()
        producer_task = asyncio.create_task(
            self._generate_episodes_streaming(batch, queue, is_validation=False)
        )

        all_episodes: list[Episode] = []
        all_trajectory_groups: list[TrajectoryGroup] = []
        all_training_datums: list = []
        fwd_bwd_futures: list = []
        rs_metrics = RejectionSamplingMetrics()

        # Consume episode lists from queue, transform and submit forward_backward
        tasks_received = 0
        tasks_skipped = 0
        fwd_bwd_submitted = 0
        total_tasks = len(batch)

        while True:
            item = await queue.get()
            if item is _QUEUE_DONE:
                break

            task_episodes: list[Episode] = item
            all_episodes.extend(task_episodes)
            tasks_received += 1

            # Track solve metrics and skip if no reward variance
            correct_mask = [ep.is_correct for ep in task_episodes]
            if all(correct_mask):
                rs_metrics.solve_all += 1
                tasks_skipped += 1
                logger.info(f"[streaming] task {tasks_received}/{total_tasks}: all correct, skipping")
                continue
            elif not any(correct_mask):
                rs_metrics.solve_none += 1
                tasks_skipped += 1
                logger.info(f"[streaming] task {tasks_received}/{total_tasks}: all incorrect, skipping")
                continue
            else:
                rs_metrics.solve_partial += 1

            # Transform episodes to trajectory groups
            task_groups, _ = transform_episodes_to_trajectory_groups(
                task_episodes,
                self.transform_config,
                self.cf_config,
                traj_grouping_hook=self.traj_grouping_hook,
            )
            if not task_groups:
                continue

            all_trajectory_groups.extend(task_groups)

            # Transform all task groups to datums and submit forward_backward
            training_datums, _ = transform_trajectory_groups_to_datums(
                task_groups,
                algorithm_config=self.algorithm_config,
            )

            mb_futures = await policy_trainer._get_forward_backward_futures(
                training_datums=training_datums,
                estimator_map=self.algorithm_config.estimator_map,
                algorithm_config=self.algorithm_config,
            )
            fwd_bwd_futures.extend(mb_futures)
            fwd_bwd_submitted += 1

            logger.info(
                f"[streaming] task {tasks_received}/{total_tasks}: "
                f"submitted forward_backward ({fwd_bwd_submitted} total, {tasks_skipped} skipped)"
            )

            if isinstance(training_datums, dict):
                for datums in training_datums.values():
                    all_training_datums.extend(datums)
            else:
                all_training_datums.extend(training_datums)

        logger.info(
            f"[streaming] all tasks done: {fwd_bwd_submitted} forward_backward submitted, "
            f"{tasks_skipped} skipped, awaiting results..."
        )

        # Re-raise any producer exception
        await producer_task

        # If no trajectory groups survived filtering, skip training
        if not all_trajectory_groups:
            trainer_state.episodes = all_episodes
            return

        # Await all forward_backward results
        all_training_logprobs = []
        for fwd_bwd_future in fwd_bwd_futures:
            fwd_bwd_result = await fwd_bwd_future.result_async()
            for output in fwd_bwd_result.loss_fn_outputs:
                all_training_logprobs.append(output["logprobs"].to_torch())

        # Submit and await optim_step after all forward_backwards are done
        optim_step_future, scheduled_learning_rate = await policy_trainer.optim_step_future(
            step=trainer_state.global_step,
            total_steps=trainer_state.total_steps,
            learning_rate=backend.learning_rate,
            beta1=backend.beta1,
            beta2=backend.beta2,
            eps=backend.eps,
        )
        await optim_step_future.result_async()

        # Populate trainer_state
        trainer_state.episodes = all_episodes
        trainer_state.trajectory_groups = all_trajectory_groups
        trainer_state.backend_batch = all_training_datums
        trainer_state.extra_info["training_logprobs"] = all_training_logprobs
        trainer_state.extra_info["scheduled_learning_rate"] = scheduled_learning_rate

        # Reward/advantage metrics from trajectory groups
        rewards = [traj.reward for g in all_trajectory_groups for traj in g.trajectories if traj.reward is not None]
        advantages = [step.advantage for g in all_trajectory_groups for traj in g.trajectories for step in traj.steps if step.advantage is not None]
        if rewards:
            trainer_state.metrics["reward/mean"] = np.mean(rewards)
            trainer_state.metrics["reward/std"] = np.std(rewards)
            trainer_state.metrics["reward/min"] = np.min(rewards)
            trainer_state.metrics["reward/max"] = np.max(rewards)
        if advantages:
            trainer_state.metrics["advantage/mean"] = np.mean(advantages)
            trainer_state.metrics["advantage/std"] = np.std(advantages)
            trainer_state.metrics["advantage/min"] = np.min(advantages)
            trainer_state.metrics["advantage/max"] = np.max(advantages)

        # Episode-level solve metrics (solve_none/solve_all/solve_partial)
        trainer_state.metrics.update(rs_metrics.to_dict())

        # Workflow metrics
        workflow_metrics, termination_counts = self._collect_workflow_metrics_from_episodes(all_episodes)
        for key, value in workflow_metrics.items():
            trainer_state.metrics[f"batch/{key}"] = value

        from rllm.workflows.workflow import TerminationReason

        total_counts = max(sum(termination_counts.values()), 1)
        for r in TerminationReason:
            trainer_state.metrics[f"batch/termination_reason/{r.value}"] = termination_counts[r.value] / total_counts

        if self.tokenizer is not None:
            from rllm.experimental.common.visualization import visualize_trajectory_last_steps

            visualize_trajectory_last_steps(
                trainer_state.trajectory_groups,
                tokenizer=self.tokenizer,
                max_steps_to_visualize=2,
                show_workflow_metadata=True,
            )
