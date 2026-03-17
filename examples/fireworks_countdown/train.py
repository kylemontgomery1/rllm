import hydra
from omegaconf import DictConfig

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer
from rllm.rewards.countdown_reward import countdown_reward_fn
from rllm.workflows.simple_workflow import SimpleWorkflow


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    trainer = AgentTrainer(
        workflow_class=SimpleWorkflow,
        workflow_args={
            "reward_function": countdown_reward_fn,
        },
        config=config,
        train_dataset=DatasetRegistry.load_dataset("countdown", "train"),
        val_dataset=DatasetRegistry.load_dataset("countdown", "test"),
        backend="fireworks",
    )
    trainer.train()


if __name__ == "__main__":
    main()
