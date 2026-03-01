set -x

python -m examples.countdown.streaming.train \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-8B \
    model.lora_rank=32 \
    training.group_size=8 \
    training.learning_rate=2e-5 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.train_batch_size=32 \
    sampling.train.temperature=1.0 \
    sampling.train.top_p=1.0 \
    rllm.trainer.total_epochs=100 \
    rllm.trainer.logger=['console','wandb'] \
    rllm.trainer.project_name='rllm-agent' \
    rllm.trainer.experiment_name='countdown-streaming' \
    rllm.trainer.val_before_train=false \
    rllm.trainer.test_freq=-1 \
    rollout_engine.bypass_render_with_parser=false \
    rllm.streaming_minibatch.enable=true
