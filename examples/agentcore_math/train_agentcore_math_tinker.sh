#!/usr/bin/env bash
set -x

# Load environment variables (TINKER_API_KEY, AGENTCORE_AGENT_ARN, AGENTCORE_S3_BUCKET)
set -a && source .env && set +a

export RLLM_GATEWAY_LOG_LEVEL=${RLLM_GATEWAY_LOG_LEVEL:-CRITICAL}

python -m examples.agentcore_math.train_agentcore_math_tinker \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-30B-A3B-Instruct-2507 \
    model.lora_rank=16 \
    training.group_size=4 \
    training.learning_rate=2e-5 \
    training.max_length=32768 \
    rllm.rollout.train.temperature=1.0 \
    data.max_prompt_length=30720 \
    data.max_response_length=2048 \
    data.train_batch_size=64 \
    rllm.workflow.n_parallel_tasks=128 \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.norm_adv_by_std_in_grpo=true \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.logger="['console', 'wandb']" \
    rllm.trainer.project_name=agentcore-math \
    rllm.trainer.experiment_name=gsm8k-agentcore-tinker-sync-30b-moe \
    rllm.trainer.val_before_train=false \
    rllm.trainer.test_freq=-1\
    rllm.trainer.save_freq=-1 \
    rllm.remote_runtime.enabled=true \
    rllm.remote_runtime.backend=agentcore \
    rllm.remote_runtime.agentcore.agent_runtime_arn=$AGENTCORE_AGENT_ARN \
    rllm.remote_runtime.agentcore.s3_bucket=$AGENTCORE_S3_BUCKET \
    rllm.remote_runtime.agentcore.tps_limit=25 \
    rllm.remote_runtime.session_timeout=300 \
    rllm.remote_runtime.agentcore.max_pool_connections=128 \
    rllm.gateway.port=9092 \
    rllm.gateway.public_url=http://5.78.144.17:9091 \
    rllm.gateway.sampling_params_priority=session
