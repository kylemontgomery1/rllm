"""
Fireworks backend implementation for the UnifiedTrainer.

Inherits from ``TinkerBackend`` and overrides only what differs:
infrastructure setup (Fireworks DeploymentManager / TrainerJobManager),
rollout engine (FireworksEngine), and checkpoint lifecycle hooks
(weight syncing via ``WeightSyncer`` instead of Tinker sampler paths).
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from fireworks.training.cookbook.utils import (
    ReconnectableClient,
    create_trainer_job,
    setup_deployment,
)
from fireworks.training.sdk import (
    DeploymentManager,
    DeploymentSampler,
    TrainerJobManager,
    WeightSyncer,
)
from fireworks.training.sdk.client import FiretitanServiceClient
from omegaconf import DictConfig
from transformers import AutoTokenizer

from rllm.experimental.common import simple_timer
from rllm.experimental.rollout import FireworksEngine, RolloutEngine
from rllm.trainer.fireworks.fireworks_policy_trainer import FireworksPolicyTrainer
from rllm.trainer.tinker.tinker_backend import TinkerBackend
from rllm.trainer.tinker.tinker_metrics_utils import (
    print_metrics_table,
    update_training_metrics,
)

if TYPE_CHECKING:
    from rllm.experimental.unified_trainer import TrainerState

logger = logging.getLogger(__name__)


class FireworksBackend(TinkerBackend):
    """Fireworks backend for the unified trainer.

    Extends ``TinkerBackend`` with Fireworks-specific infrastructure:
        - ``FireworksEngine`` for rollout (via ``DeploymentSampler``)
        - ``FireworksPolicyTrainer`` for gradient updates (via ``ReconnectableClient``)
        - ``WeightSyncer`` for hot-loading checkpoints into an inference deployment

    Inherited unchanged from ``TinkerBackend``:
        - ``get_dataloader``, ``shutdown``
        - ``generate_episodes``, ``transform_to_backend_batch``
        - ``process_backend_batch``, ``compute_advantages``, ``update_policy``
        - ``on_epoch_start/end``, ``on_validation_start/end``
    """

    name: str = "fireworks"

    def __init__(self, config: DictConfig, **kwargs):
        # Intentionally skip TinkerBackend.__init__ to avoid creating a
        # tinker.ServiceClient — we set up Fireworks-specific clients instead.
        from rllm.experimental.protocol import BackendProtocol

        BackendProtocol.__init__(self, config, **kwargs)

        self.full_config = config

        self.policy_trainer: FireworksPolicyTrainer | None = None
        self.tokenizer = None
        self.rollout_engine: FireworksEngine | None = None

        # In TinkerBackend this is a tinker.SamplingClient; here it's a
        # DeploymentSampler — but both get passed to set_sampling_client().
        self.sampling_client: DeploymentSampler | None = None
        self._algorithm_config = None

        self.learning_rate = config.training.get("learning_rate", 1e-6)
        self.beta1 = config.training.get("beta1", 0.9)
        self.beta2 = config.training.get("beta2", 0.95)
        self.eps = config.training.get("eps", 1e-8)

        # Fireworks-specific handles (populated in _init_fireworks_infra)
        self.weight_syncer: WeightSyncer | None = None
        self._policy_rc: ReconnectableClient | None = None
        self._reference_rc: ReconnectableClient | None = None

    # ------------------------------------------------------------------
    # Fireworks infrastructure setup
    # ------------------------------------------------------------------

    def _init_fireworks_infra(self, **kwargs) -> None:
        """Create Fireworks TrainerJobManager, DeploymentManager,
        ReconnectableClient, WeightSyncer, and DeploymentSampler."""
        cfg = self.full_config
        api_key = os.environ["FIREWORKS_API_KEY"]
        account = cfg.get("account", "rllm-project")
        base_url = cfg.get("fireworks_base_url", "https://api.fireworks.ai")

        rlor_mgr = TrainerJobManager(api_key=api_key, account_id=account, base_url=base_url)
        deploy_mgr = DeploymentManager(api_key=api_key, account_id=account, base_url=base_url)

        deployment_id = cfg.deployment.deployment_id
        dep_info = setup_deployment(deploy_mgr, cfg.deployment, cfg.model.name, cfg.training_infra)

        ref_extra = list(cfg.training_infra.get("extra_args", []) or [])
        if "--forward-only" not in ref_extra:
            ref_extra.append("--forward-only")
        if "--no-compile" not in ref_extra:
            ref_extra.append("--no-compile")

        with ThreadPoolExecutor(max_workers=2) as pool:
            pol_fut = pool.submit(
                create_trainer_job,
                rlor_mgr,
                base_model=cfg.model.name,
                infra=cfg.training_infra,
                lora_rank=cfg.model.get("lora_rank", 0),
                max_seq_len=cfg.training.max_length,
                learning_rate=cfg.training.learning_rate,
                display_name=cfg.get("display_name", "rllm-policy"),
                hot_load_deployment_id=deployment_id,
            )
            ref_fut = pool.submit(
                create_trainer_job,
                rlor_mgr,
                base_model=cfg.model.name,
                infra=cfg.training_infra,
                lora_rank=cfg.model.get("lora_rank", 0),
                max_seq_len=cfg.training.max_length,
                learning_rate=cfg.training.learning_rate,
                display_name=cfg.get("display_name", "rllm-reference"),
                extra_args=ref_extra,
            )
            policy_ep = pol_fut.result()
            reference_ep = ref_fut.result()

        self._policy_rc = ReconnectableClient(
            rlor_mgr, policy_ep.job_id, cfg.model.name,
            lora_rank=cfg.model.get("lora_rank", 0),
        )
        self._reference_rc = ReconnectableClient(
            rlor_mgr, reference_ep.job_id, cfg.model.name,
            lora_rank=cfg.model.get("lora_rank", 0),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.deployment.get("tokenizer_model", cfg.model.name),
            trust_remote_code=True,
        )
        inference_model = dep_info.inference_model if dep_info else cfg.model.name
        self.sampling_client = DeploymentSampler(
            inference_url=deploy_mgr.inference_url,
            model=inference_model,
            api_key=api_key,
            tokenizer=self.tokenizer,
        )
        self.weight_syncer = WeightSyncer(
            policy_client=self._policy_rc.inner,
            deploy_mgr=deploy_mgr,
            deployment_id=deployment_id,
            base_model=cfg.model.name,
            hotload_timeout=cfg.get("hotload", {}).get("hot_load_timeout", 300),
            first_checkpoint_type=cfg.get("hotload", {}).get("first_checkpoint_type", "base"),
        )

    # ------------------------------------------------------------------
    # BackendProtocol overrides
    # ------------------------------------------------------------------

    def init_rollout_engine(self, **kwargs) -> RolloutEngine:
        self._init_fireworks_infra(**kwargs)

        self.policy_trainer = FireworksPolicyTrainer(
            config=self.full_config,
            training_client=self._policy_rc,
            reference_client=self._reference_rc,
            weight_syncer=self.weight_syncer,
            cf_config=kwargs.get("cf_config"),
            transform_config=kwargs.get("transform_config"),
            algorithm_config=kwargs.get("algorithm_config"),
        )

        self.rollout_engine = FireworksEngine(
            tokenizer=self.tokenizer,
            sampler=self.sampling_client,
            max_prompt_length=self.full_config.data.max_prompt_length,
            max_response_length=self.full_config.data.max_response_length,
            max_model_length=self.full_config.training.max_length,
            sampling_params=self.full_config.sampling,
            **self.full_config.get("rollout_engine", {}),
        )
        return self.rollout_engine

    def validate_config(self) -> None:
        sampling_params = self.full_config.sampling
        if sampling_params.get("temperature", 1.0) != 1.0 or sampling_params.get("top_p", 1.0) != 1.0:
            logger.warning(
                "Temperature and top_p are set away from 1.0, this can cause "
                "issues with logprobs accuracy."
            )

    # ------------------------------------------------------------------
    # Train lifecycle hooks (overrides)
    # ------------------------------------------------------------------

    async def on_train_start(self, trainer_state: TrainerState) -> None:
        assert self.policy_trainer is not None, "policy_trainer is not initialized"
        os.makedirs(self.full_config.training.default_local_dir, exist_ok=True)

        start_step = await self.policy_trainer.initialize_async(resume_from_checkpoint=True)
        trainer_state.global_step = start_step

    async def on_train_end(self, trainer_state: TrainerState) -> None:
        assert self.policy_trainer is not None, "policy_trainer is not initialized"

        if trainer_state.global_step % self.full_config.rllm.trainer.save_freq != 0:
            logger.info("Saving final checkpoint at step %d", trainer_state.global_step)
            await self.policy_trainer.save_checkpoint_and_sync_weights(
                trainer_state.global_step, save_dcp=True,
            )

    async def on_batch_end(self, trainer_state: TrainerState) -> None:
        assert self.policy_trainer is not None, "policy_trainer is not initialized"

        global_step = trainer_state.global_step
        with simple_timer("save_checkpoint", trainer_state.timing_dict):
            save_freq = self.full_config.rllm.trainer.save_freq
            save_dcp = save_freq > 0 and global_step % save_freq == 0
            logger.info("Syncing weights at step %d (save_dcp=%s)", global_step, save_dcp)
            await self.policy_trainer.save_checkpoint_and_sync_weights(
                global_step, save_dcp=save_dcp,
            )

        learning_rate = trainer_state.extra_info.get("scheduled_learning_rate", self.learning_rate)
        update_training_metrics(trainer_state, learning_rate, trainer_state.total_steps)

        if trainer_state.metrics:
            print_metrics_table(trainer_state.metrics, global_step)
        update_training_metrics(trainer_state, learning_rate, trainer_state.total_steps)

        if trainer_state.metrics:
            print_metrics_table(trainer_state.metrics, global_step)
