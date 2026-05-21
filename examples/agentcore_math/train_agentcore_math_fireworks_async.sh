#!/usr/bin/env bash
set -ex

# Load environment variables (FIREWORKS_API_KEY, AGENTCORE_AGENT_ARN, AGENTCORE_S3_BUCKET)
set -a && source .env && set +a

export RLLM_GATEWAY_LOG_LEVEL=${RLLM_GATEWAY_LOG_LEVEL:-CRITICAL}

python -m examples.agentcore_math.train_agentcore_math_fireworks \
    rllm/backend=fireworks \
    model.name=accounts/fireworks/models/qwen3-4b-instruct-2507 \
    training_infra.training_shape_id=accounts/fireworks/trainingShapes/qwen3-4b-minimum/versions/ykacgllm \
    deployment.deployment_shape=accounts/fireworks/deploymentShapes/rft-qwen3-4b/versions/az2rbxop \
    deployment.replica_count=1 \
    deployment.tokenizer_model=Qwen/Qwen3-4B-Instruct-2507 \
    hotload.reset_prompt_cache=true \
    training.group_size=8 \
    validation.group_size=1 \
    training.learning_rate=5e-6 \
    training.max_length=16384 \
    sampling.train.temperature=1.0 \
    data.max_prompt_length=14336 \
    data.max_response_length=2048 \
    data.train_batch_size=1 \
    data.val_batch_size=-1 \
    rllm.workflow.n_parallel_tasks=64 \
    rllm.workflow.retry_limit=1 \
    rllm.remote_runtime.enabled=true \
    rllm.remote_runtime.backend=agentcore \
    rllm.remote_runtime.agentcore.agent_runtime_arn=$AGENTCORE_AGENT_ARN \
    rllm.remote_runtime.agentcore.s3_bucket=$AGENTCORE_S3_BUCKET \
    rllm.remote_runtime.agentcore.tps_limit=25 \
    rllm.remote_runtime.agentcore.max_pool_connections=64 \
    rllm.remote_runtime.session_timeout=300 \
    rllm.gateway.port=9092 \
    rllm.gateway.public_url=http://5.78.144.17:9091 \
    rllm.gateway.sampling_params_priority=session \
    rllm.async_training.enable=true \
    rllm.async_training.mini_batch_size=32 \
    rllm.async_training.fwd_bwd_group_size=32 \
    rllm.async_training.staleness_threshold=2.0 \
    rllm.async_training.trigger_parameter_sync_step=1 \
    rllm.async_training.partial_rollout=true \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.norm_adv_by_std_in_grpo=true \
    rllm.algorithm.loss_fn=dapo \
    rllm.algorithm.loss_agg_mode=token-mean \
    rllm.algorithm.kl_beta=0.0 \
    rllm.algorithm.eps_clip=0.2 \
    rllm.algorithm.eps_clip_high=0.28 \
    rllm.algorithm.rollout_correction.bypass_mode=false \
    rllm.algorithm.rollout_correction.icepop_mode=token \
    rllm.algorithm.rollout_correction.icepop_beta=2.0 \
    rllm.rejection_sample.filter_uniform_groups=true \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.logger=['console','wandb'] \
    rllm.trainer.project_name=agentcore-math \
    rllm.trainer.experiment_name=gsm8k-agentcore-fireworks-async-4b-stale-bsz-32-filter \
    rllm.trainer.val_before_train=false \
    rllm.trainer.test_freq=-1 \
    rllm.trainer.save_freq=-1
