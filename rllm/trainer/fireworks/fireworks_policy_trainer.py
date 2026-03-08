"""Policy training module for Firetitan-based RL.

Uses ``FiretitanTrainingClient`` / ``ReconnectableClient`` from the
Fireworks training SDK instead of Tinker's ``ServiceClient``.

This module handles gradient updates, model checkpointing, and data processing.
It does NOT contain any environment or agent logic.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

import tinker
from fireworks.training.cookbook.utils import ReconnectableClient
from fireworks.training.sdk import WeightSyncer
from tinker.types import AdamParams

from rllm.agents.agent import TrajectoryGroup
from rllm.experimental.common import (
    AlgorithmConfig,
    CompactFilteringConfig,
    TransformConfig,
    rLLMAdvantageEstimator,
)
from rllm.trainer.tinker.tinker_policy_trainer import (
    compute_schedule_lr_multiplier,
    require_training_client,
)
from rllm.trainer.tinker.transform import transform_trajectory_groups_to_datums

logger = logging.getLogger(__name__)


ADV_TO_LOSS_FN_AUTO_MAP = {
    rLLMAdvantageEstimator.REINFORCE: "importance_sampling",
    rLLMAdvantageEstimator.GRPO: "ppo",
    rLLMAdvantageEstimator.OTHER: "importance_sampling",
}


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

    async def initialize_async(self, resume_from_checkpoint: bool = True) -> int:
        """Initialize or resume training.

        Handles checkpoint resume via ``FiretitanTrainingClient.list_checkpoints``
        and ``load_state_with_optimizer``.

        Args:
            resume_from_checkpoint: If True, attempt to resume from the
                last DCP checkpoint.

        Returns:
            The starting global step (0 when training from scratch).
        """
        start_step = 0

        if resume_from_checkpoint:
            start_step = await self._try_resume()

        if start_step == 0:
            logger.info("Starting training from scratch with model: %s", self.config.model.name)
            await self._initial_weight_sync()

        return start_step

    async def _try_resume(self) -> int:
        """Attempt to resume from a DCP checkpoint.

        Returns:
            The step to resume from, or 0 if no checkpoint was found.
        """
        inner = self.training_client.inner
        checkpoints, _ = inner.list_checkpoints()
        if not checkpoints:
            logger.info("No existing checkpoints found.")
            return 0

        latest_name = checkpoints[-1]
        logger.info("Resuming from checkpoint: %s", latest_name)

        checkpoint_ref = inner.resolve_checkpoint_path(latest_name)
        await asyncio.to_thread(lambda: self.training_client.load_state_with_optimizer(checkpoint_ref).result())

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

    @require_training_client
    async def _get_forward_backward_futures(
        self,
        training_datums: list[tinker.Datum] | dict[str, list[tinker.Datum]],
        estimator_map: dict[str, rLLMAdvantageEstimator],
        algorithm_config: AlgorithmConfig,
    ) -> list[Any]:
        fwd_bwd_futures = []
        if isinstance(training_datums, dict):
            for group_role, datums in training_datums.items():
                estimator = estimator_map.get(group_role, self.algorithm_config.estimator)
                loss_fn = algorithm_config.loss_fn or ADV_TO_LOSS_FN_AUTO_MAP[estimator]
                fwd_bwd_future = self.training_client.forward_backward_custom(
                    [self._remove_mask(datum) for datum in datums],
                    loss_fn,
                )
                fwd_bwd_futures.append(fwd_bwd_future)
        else:
            loss_fn = algorithm_config.loss_fn or ADV_TO_LOSS_FN_AUTO_MAP[algorithm_config.estimator]
            fwd_bwd_future = self.training_client.forward_backward_custom(
                [self._remove_mask(datum) for datum in training_datums],
                loss_fn,
            )
            fwd_bwd_futures.append(fwd_bwd_future)

        return fwd_bwd_futures

    @require_training_client
    async def forward_backward_from_trajectory_groups(
        self,
        trajectory_groups: list[TrajectoryGroup],
        algorithm_config: AlgorithmConfig | None = None,
    ) -> tuple[list[tinker.Datum] | dict[str, list[tinker.Datum]], list[torch.Tensor], dict]:
        """Run forward-backward pass from trajectory groups.

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

        fwd_bwd_futures = await self._get_forward_backward_futures(
            training_datums=training_datums,
            estimator_map=algorithm_config.estimator_map,
            algorithm_config=algorithm_config,
        )

        fwd_bwd_results = await asyncio.gather(*[asyncio.to_thread(lambda f=fut: f.result()) for fut in fwd_bwd_futures])

        training_logprobs = []
        for fwd_bwd_result in fwd_bwd_results:
            for output in fwd_bwd_result.loss_fn_outputs:
                logprobs = output["logprobs"].to_torch()
                training_logprobs.append(logprobs)

        return training_datums, training_logprobs, adv_metrics

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    @require_training_client
    async def optim_step_future(
        self,
        step: int,
        total_steps: int,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
    ) -> tuple[Any, float]:
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
        )
        future = self.training_client.optim_step(adam_params)
        return future, scheduled_lr

    @require_training_client
    async def fused_forward_backward_and_optim_step(
        self,
        step: int,
        total_steps: int,
        trajectory_groups: list[TrajectoryGroup],
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
    ) -> tuple[list[tinker.Datum] | dict[str, list[tinker.Datum]], list[torch.Tensor], dict, float]:
        """Run forward-backward and optimizer step overlapped."""
        training_datums, adv_metrics = transform_trajectory_groups_to_datums(
            trajectory_groups,
            algorithm_config=self.algorithm_config,
        )

        fwd_bwd_futures = await self._get_forward_backward_futures(
            training_datums=training_datums,
            estimator_map=self.algorithm_config.estimator_map,
            algorithm_config=self.algorithm_config,
        )

        optim_future, scheduled_lr = await self.optim_step_future(
            step=step,
            total_steps=total_steps,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
        )

        # Wait for all futures together
        fwd_bwd_results = await asyncio.gather(*[asyncio.to_thread(lambda f=fut: f.result()) for fut in fwd_bwd_futures])
        await asyncio.to_thread(lambda: optim_future.result())

        training_logprobs = []
        for fwd_bwd_result in fwd_bwd_results:
            for output in fwd_bwd_result.loss_fn_outputs:
                logprobs = output["logprobs"].to_torch()
                training_logprobs.append(logprobs)

        return training_datums, training_logprobs, adv_metrics, scheduled_lr

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    @require_training_client
    async def save_checkpoint_and_sync_weights(
        self,
        step: int,
        save_dcp: bool = False,
    ) -> None:
        """Save sampler weights, hot-load into deployment, and optionally save a DCP checkpoint.

        After hot-load completes the existing ``DeploymentSampler`` will
        automatically serve the updated weights on its next request — no
        new sampler object is needed.

        Args:
            step: Current global step.
            save_dcp: Whether to also save a persistent DCP checkpoint.
        """
        name = f"step-{step}"

        if save_dcp:
            await asyncio.to_thread(lambda: self.training_client.inner.save_state(name).result(timeout=1800))
            logger.info("DCP checkpoint saved: %s", name)

        await self._sync_weights(name)

    @require_training_client
    async def save_dcp_checkpoint(self, step: int) -> None:
        """Save a DCP (distributed checkpoint) only, without hot-loading."""
        name = f"step-{step}"
        await asyncio.to_thread(lambda: self.training_client.inner.save_state(name).result(timeout=1800))
        logger.info("DCP checkpoint saved: %s", name)

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

        ref_fwd = await asyncio.to_thread(lambda: self.reference_client.forward(datums, "cross_entropy").result())
        return [out["logprobs"].data for out in ref_fwd.loss_fn_outputs]
