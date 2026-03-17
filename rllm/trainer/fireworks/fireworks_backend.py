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

from training.utils import (
    ReconnectableClient,
    create_trainer_job,
    setup_deployment,
)
from training.utils.config import DeployConfig, InfraConfig
from fireworks.training.sdk import (
    DeploymentManager,
    DeploymentSampler,
    TrainerJobManager,
    WeightSyncer,
)
from omegaconf import DictConfig
from transformers import AutoTokenizer

# fix:fireworks - tinker 0.15.0 sends project_id=None, server rejects it
from tinker.types import CreateSessionRequest
_orig_model_dump = CreateSessionRequest.model_dump
def _patched_model_dump(self, **kwargs):
    result = _orig_model_dump(self, **kwargs)
    if result.get("project_id") is None:
        result.pop("project_id", None)
    return result
CreateSessionRequest.model_dump = _patched_model_dump

# fix:fireworks - optim_step sends grad_accumulation_normalization via extra_body, server rejects it
from fireworks.training.sdk.client import FiretitanTrainingClient
from tinker.lib.public_interfaces.training_client import TrainingClient
FiretitanTrainingClient.optim_step = TrainingClient.optim_step
from training.utils.client import ReconnectableClient as _RC
_orig_rc_optim_step = _RC.optim_step
def _patched_rc_optim_step(self, params, grad_accumulation_normalization=None):
    return self._client.optim_step(params).result(timeout=self._default_timeout)
_RC.optim_step = _patched_rc_optim_step

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

        self._policy_updated_this_step: bool = False

        self.learning_rate = config.training.get("learning_rate", 1e-6)
        self.beta1 = config.training.get("beta1", 0.9)
        self.beta2 = config.training.get("beta2", 0.95)
        self.eps = config.training.get("eps", 1e-8)
        self.weight_decay = config.training.get("weight_decay", 0.01)
        self.grad_clip_norm = config.training.get("grad_clip_norm", 1.0)

        # Fireworks-specific handles (populated in _init_fireworks_infra)
        self.weight_syncer: WeightSyncer | None = None
        self._policy_rc: ReconnectableClient | None = None
        self._reference_rc: ReconnectableClient | None = None
        self._rlor_mgr: TrainerJobManager | None = None
        self._deploy_mgr: DeploymentManager | None = None
        self._policy_job_id: str | None = None
        self._reference_job_id: str | None = None
        self._deployment_id: str | None = None

    # ------------------------------------------------------------------
    # Fireworks infrastructure setup
    # ------------------------------------------------------------------

    @staticmethod
    def _to_infra_config(cfg_section: DictConfig) -> InfraConfig:
        """Convert an OmegaConf ``training_infra`` section to an ``InfraConfig`` dataclass."""
        return InfraConfig(
            training_shape_id=cfg_section.get("training_shape_id"),
            ref_training_shape_id=cfg_section.get("ref_training_shape_id"),
            region=cfg_section.get("region"),
            custom_image_tag=cfg_section.get("custom_image_tag"),
            accelerator_type=cfg_section.get("accelerator_type"),
            accelerator_count=cfg_section.get("accelerator_count"),
            node_count=cfg_section.get("node_count", 1),
            extra_args=list(cfg_section.get("extra_args") or []),
        )

    @staticmethod
    def _to_deploy_config(cfg_section: DictConfig) -> DeployConfig:
        """Convert an OmegaConf ``deployment`` section to a ``DeployConfig`` dataclass."""
        return DeployConfig(
            deployment_id=cfg_section.get("deployment_id"),
            deployment_shape=cfg_section.get("deployment_shape"),
            deployment_region=cfg_section.get("deployment_region"),
            deployment_accelerator_type=cfg_section.get("deployment_accelerator_type"),
            hot_load_bucket_type=cfg_section.get("hot_load_bucket_type", "FW_HOSTED"),
            deployment_timeout_s=cfg_section.get("deployment_timeout_s", 5400),
            deployment_extra_args=list(cfg_section.get("deployment_extra_args") or []) or None,
            tokenizer_model=cfg_section.get("tokenizer_model"),
            sample_timeout=cfg_section.get("sample_timeout", 600),
            disable_speculative_decoding=cfg_section.get("disable_speculative_decoding", True),
            extra_values=dict(cfg_section.get("extra_values") or {}) or None,
        )

    def _init_fireworks_infra(self, **kwargs) -> None:
        """Create Fireworks TrainerJobManager, DeploymentManager,
        ReconnectableClient, WeightSyncer, and DeploymentSampler."""
        cfg = self.full_config
        api_key = os.environ["FIREWORKS_API_KEY"]
        account = cfg.get("account") or os.environ.get("FIREWORKS_ACCOUNT_ID", "")
        base_url = cfg.get("fireworks_base_url", "https://api.fireworks.ai")

        self._rlor_mgr = TrainerJobManager(api_key=api_key, account_id=account, base_url=base_url)
        self._deploy_mgr = DeploymentManager(api_key=api_key, account_id=account, base_url=base_url)
        rlor_mgr = self._rlor_mgr
        deploy_mgr = self._deploy_mgr

        infra = self._to_infra_config(cfg.training_infra)
        deploy = self._to_deploy_config(cfg.deployment)

        # Resolve training shape profile and auto-derive config values
        profile = None
        if infra.training_shape_id:
            profile = rlor_mgr.resolve_training_profile(infra.training_shape_id)
            dep_shape = getattr(profile, "deployment_shape", None) or getattr(profile, "deployment_shape_version", None)
            if dep_shape and not deploy.deployment_shape:
                deploy.deployment_shape = dep_shape
                logger.info("Auto-derived deployment_shape from training shape: %s", dep_shape)
            if profile.max_supported_context_length and not cfg.training.get("max_length"):
                cfg.training.max_length = profile.max_supported_context_length
                logger.info("Auto-derived max_length from training shape: %d", cfg.training.max_length)

        deployment_id = deploy.deployment_id
        dep_info = setup_deployment(deploy_mgr, deploy, cfg.model.name, infra)

        use_reference = cfg.rllm.algorithm.get("kl_beta", 0.0) > 0

        ref_profile = None
        if use_reference:
            if infra.ref_training_shape_id:
                ref_profile = rlor_mgr.resolve_training_profile(infra.ref_training_shape_id)
            elif profile is not None:
                ref_profile = profile

        with ThreadPoolExecutor(max_workers=2) as pool:
            pol_fut = pool.submit(
                create_trainer_job,
                rlor_mgr,
                base_model=cfg.model.name,
                infra=infra,
                profile=profile,
                lora_rank=cfg.model.get("lora_rank", 0),
                max_seq_len=cfg.training.max_length,
                learning_rate=cfg.training.learning_rate,
                display_name=cfg.get("display_name", "rllm-policy"),
                hot_load_deployment_id=deployment_id,
            )
            if use_reference:
                ref_fut = pool.submit(
                    create_trainer_job,
                    rlor_mgr,
                    base_model=cfg.model.name,
                    infra=infra,
                    profile=ref_profile,
                    lora_rank=cfg.model.get("lora_rank", 0),
                    max_seq_len=cfg.training.max_length,
                    learning_rate=cfg.training.learning_rate,
                    display_name=cfg.get("display_name", "rllm-ref"),
                    forward_only=True,
                )
            policy_ep = pol_fut.result()
            reference_ep = ref_fut.result() if use_reference else None

        self._policy_job_id = policy_ep.job_id
        self._reference_job_id = reference_ep.job_id if reference_ep else None
        self._deployment_id = deployment_id

        self._policy_rc = ReconnectableClient(
            rlor_mgr, policy_ep.job_id, cfg.model.name,
            lora_rank=cfg.model.get("lora_rank", 0),
        )
        self._reference_rc = (
            ReconnectableClient(
                rlor_mgr, reference_ep.job_id, cfg.model.name,
                lora_rank=cfg.model.get("lora_rank", 0),
            )
            if reference_ep else None
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            deploy.tokenizer_model or cfg.model.name,
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
            hotload_timeout=cfg.hotload.hot_load_timeout,
            dcp_timeout=cfg.hotload.get("dcp_timeout", 2700),
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
            sample_timeout=self.full_config.deployment.get("sample_timeout", 600),
            router_replay=self.full_config.rllm.algorithm.get("router_replay", False),
            **self.full_config.get("rollout_engine", {}),
        )
        return self.rollout_engine

    def validate_config(self) -> None:
        if self.full_config.get("fuse_forward_backward_and_optim_step", False):
            raise ValueError(
                "fuse_forward_backward_and_optim_step is not supported by the Fireworks backend. "
                "Set fuse_forward_backward_and_optim_step: false in your config."
            )

        sampling_params = self.full_config.sampling
        if sampling_params.get("temperature", 1.0) != 1.0 or sampling_params.get("top_p", 1.0) != 1.0:
            logger.warning(
                "Temperature and top_p are set away from 1.0, this can cause "
                "issues with logprobs accuracy."
            )

        # --- Algorithm / loss function validation ---
        alg = self.full_config.rllm.algorithm
        loss_fn = alg.get("loss_fn", None)
        eps_clip_high = alg.get("eps_clip_high", None)
        rc = alg.get("rollout_correction", {})
        tis_mode = rc.get("tis_mode", None)
        bypass_mode = rc.get("bypass_mode", True)

        _FIREWORKS_COOKBOOK_LOSS_FNS = {"grpo", "dapo", "gspo", "cispo"}
        if loss_fn is not None and loss_fn not in _FIREWORKS_COOKBOOK_LOSS_FNS:
            raise ValueError(
                f"loss_fn='{loss_fn}' is not a supported Fireworks cookbook loss function. "
                f"Supported: {sorted(_FIREWORKS_COOKBOOK_LOSS_FNS)}"
            )

        # eps_clip_high only meaningful for dapo/cispo (asymmetric clipping)
        if eps_clip_high is not None and loss_fn not in ("dapo", "cispo"):
            logger.warning(
                "eps_clip_high is set but loss_fn='%s' does not use asymmetric "
                "clipping. eps_clip_high is only used by 'dapo' and 'cispo'.",
                loss_fn,
            )

        # rollout_correction.tis_mode validation
        if tis_mode is not None and tis_mode not in ("token", "sequence"):
            raise ValueError(
                f"rollout_correction.tis_mode must be null, 'token', or 'sequence', got '{tis_mode}'"
            )

        # TIS with bypass is a no-op (prox = inf → weight = 1.0)
        if tis_mode is not None and bypass_mode:
            logger.warning(
                "rollout_correction.tis_mode='%s' with bypass_mode=true — TIS weight "
                "will be 1.0 (no correction). Set bypass_mode=false for active TIS.",
                tis_mode,
            )

    # ------------------------------------------------------------------
    # Policy update (override — no fused path, uses ReconnectableClient)
    # ------------------------------------------------------------------

    async def update_policy(self, trainer_state: TrainerState, **kwargs) -> None:
        assert self.policy_trainer is not None, "policy_trainer is not initialized"

        with simple_timer("optim_step", trainer_state.timing_dict):
            scheduled_lr, optim_metrics = await self.policy_trainer.optim_step(
                step=trainer_state.global_step,
                total_steps=trainer_state.total_steps,
                learning_rate=self.learning_rate,
                beta1=self.beta1,
                beta2=self.beta2,
                eps=self.eps,
                weight_decay=self.weight_decay,
                grad_clip_norm=self.grad_clip_norm,
            )
            trainer_state.extra_info["scheduled_learning_rate"] = scheduled_lr
            trainer_state.metrics.update(optim_metrics)

    # ------------------------------------------------------------------
    # Train lifecycle hooks (overrides)
    # ------------------------------------------------------------------

    async def on_train_start(self, trainer_state: TrainerState) -> None:
        assert self.policy_trainer is not None, "policy_trainer is not initialized"

        start_step = await self.policy_trainer.initialize_async(
            resume_from_checkpoint=True,
            hot_load_before_training=self.full_config.hotload.get("hot_load_before_training", False),
        )
        trainer_state.global_step = start_step

    async def on_train_end(self, trainer_state: TrainerState) -> None:
        assert self.policy_trainer is not None, "policy_trainer is not initialized"
        logger.info("Saving final DCP checkpoint at step %d", trainer_state.global_step)
        await self.policy_trainer.save_dcp_checkpoint(trainer_state.global_step)

    async def on_policy_updated(self, trainer_state: TrainerState) -> None:
        assert self.policy_trainer is not None
        self._policy_updated_this_step = True

        global_step = trainer_state.global_step
        save_freq = self.full_config.rllm.trainer.save_freq

        if save_freq > 0 and global_step % save_freq == 0:
            await self.policy_trainer.save_dcp_checkpoint(global_step)

        await self.policy_trainer.sync_weights(global_step)

    async def on_batch_end(self, trainer_state: TrainerState) -> None:
        assert self.policy_trainer is not None, "policy_trainer is not initialized"

        global_step = trainer_state.global_step

        # In async mode, on_policy_updated already handled checkpoint + hotload
        if not self._policy_updated_this_step:
            save_freq = self.full_config.rllm.trainer.save_freq
            if save_freq > 0 and global_step % save_freq == 0:
                with simple_timer("save_checkpoint", trainer_state.timing_dict):
                    await self.policy_trainer.save_dcp_checkpoint(global_step)

            hot_load_interval = self.full_config.hotload.get("hot_load_interval", 1)
            if hot_load_interval > 0 and global_step % hot_load_interval == 0:
                with simple_timer("sync_weights", trainer_state.timing_dict):
                    await self.policy_trainer.sync_weights(global_step)
        self._policy_updated_this_step = False

        learning_rate = trainer_state.extra_info.get("scheduled_learning_rate", self.learning_rate)
        update_training_metrics(trainer_state, learning_rate, trainer_state.total_steps)

        if trainer_state.metrics:
            print_metrics_table(trainer_state.metrics, global_step)

    def shutdown(self) -> None:
        """Cleanup Fireworks resources: delete trainer jobs and scale deployment to zero."""
        if self._rlor_mgr:
            if self._policy_job_id:
                try:
                    logger.info("Deleting policy trainer job %s", self._policy_job_id)
                    self._rlor_mgr.delete(self._policy_job_id)
                except Exception as e:
                    logger.warning("Failed to delete policy job: %s", e)
            if self._reference_job_id:
                try:
                    logger.info("Deleting reference trainer job %s", self._reference_job_id)
                    self._rlor_mgr.delete(self._reference_job_id)
                except Exception as e:
                    logger.warning("Failed to delete reference job: %s", e)
        if self._deploy_mgr and self._deployment_id:
            try:
                logger.info("Scaling deployment %s to zero", self._deployment_id)
                self._deploy_mgr.scale_to_zero(self._deployment_id)
            except Exception as e:
                logger.warning("Failed to scale deployment to zero: %s", e)
