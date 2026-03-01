import logging

from omegaconf import DictConfig, OmegaConf

from rllm.data import Dataset
from rllm.experimental.tinker.tinker_backend import TinkerBackend
from rllm.experimental.unified_trainer import TrainerLauncher, UnifiedTrainer
from rllm.workflows.workflow import Workflow

logger = logging.getLogger(__name__)


class TinkerTrainerLauncher(TrainerLauncher):
    """
    Tinker trainer launcher that scaffolds the tinker backend.
    """

    def __init__(
        self,
        config: DictConfig,
        workflow_class: type[Workflow],
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        workflow_args: dict | None = None,
        **kwargs,
    ):
        """Initialize the TinkerTrainerLauncher. Nothing special here, just use the parent class's init."""
        super().__init__(config, workflow_class, train_dataset, val_dataset, workflow_args, **kwargs)

    def _build_trainer(self) -> UnifiedTrainer:
        """Build the appropriate trainer based on config."""
        streaming_cfg = OmegaConf.select(self.config, "rllm.streaming_minibatch", default=None)
        use_streaming = streaming_cfg is not None and streaming_cfg.get("enable", False)

        trainer_kwargs = dict(
            backend_cls=TinkerBackend,
            config=self.config,
            workflow_class=self.workflow_class,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            workflow_args=self.workflow_args,
            **self.kwargs,
        )

        if use_streaming:
            from rllm.experimental.streaming_trainer import StreamingUnifiedTrainer

            logger.info("Using StreamingUnifiedTrainer")
            return StreamingUnifiedTrainer(**trainer_kwargs)
        else:
            return UnifiedTrainer(**trainer_kwargs)

    def train(self):
        trainer = None
        try:
            trainer = self._build_trainer()
            trainer.fit()
        except Exception as e:
            print(f"Error training Tinker: {e}")
            raise e
        finally:
            if trainer is not None:
                trainer.shutdown()
