set -x

python -m examples.fireworks_countdown.train \
    rllm/backend=fireworks \
    account=rllm-project \
    model.name=accounts/fireworks/models/qwen3-4b \
    deployment.tokenizer_model=Qwen/Qwen3-4B \
    model.lora_rank=0 \
    training.group_size=8 \
    training.learning_rate=1e-5 \
    training.max_length=2048 \
    training_infra.training_shape_id=qwen3-4b-b200 \
    deployment.deployment_shape=accounts/fireworks/deploymentShapes/rft-qwen3-4b-b200-v2 \
    validation.group_size=1 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.train_batch_size=32 \
    rllm.workflow.n_parallel_tasks=256 \
    rllm.workflow.retry_limit=1 \
    rllm.workflow.raise_on_error=true \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.logger=['console','wandb'] \
    rllm.trainer.project_name='countdown' \
    rllm.trainer.experiment_name='fireworks-sync' \
    rllm.trainer.val_before_train=true \
    rllm.trainer.test_freq=10 \
    rllm.trainer.save_freq=-1 \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.norm_adv_by_std_in_grpo=true \
    rllm.algorithm.loss_fn=dapo \
    rllm.algorithm.eps_clip=0.2 \
    rllm.algorithm.eps_clip_high=0.28 \
    rllm.algorithm.rollout_correction.bypass_mode=true \
    rllm.algorithm.rollout_correction.tis_mode=null
