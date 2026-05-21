"""Train a math agent on GSM8K using Fireworks backend + AgentCore runtime.

The agent runs inside an AgentCore container (strands_math_agent) and calls back
to the rllm-model-gateway for model inference. The gateway captures all traces
(token IDs, logprobs) while the agent returns only the reward.

Usage:
    # First prepare data:
    python -m examples.agentcore_math.prepare_gsm8k_data

    # Then train:
    bash examples/agentcore_math/train_agentcore_math_fireworks_async.sh
"""

import logging
import os

import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer


def _configure_external_loggers() -> None:
    agentcore_level = os.getenv("AGENTCORE_LOG_LEVEL", "WARNING").upper()
    urllib3_level = os.getenv("URLLIB3_LOG_LEVEL", "ERROR").upper()

    logging.getLogger("agentcore_rl_toolkit.client").setLevel(agentcore_level)
    logging.getLogger("urllib3.connectionpool").setLevel(urllib3_level)


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config):
    _configure_external_loggers()

    train_dataset = DatasetRegistry.load_dataset("gsm8k_agentcore", "train")
    test_dataset = DatasetRegistry.load_dataset("gsm8k_agentcore", "test")

    trainer = AgentTrainer(
        backend="fireworks",
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
