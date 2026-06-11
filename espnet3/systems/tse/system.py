"""TSE system implementation and training helpers."""

import logging

from espnet3.systems.base.system import BaseSystem


logger = logging.getLogger(__name__)


class TSESystem(BaseSystem):
    """TSE-specific system.

    Inherits ``create_dataset`` from :class:`~espnet3.systems.base.system.BaseSystem`,
    which uses the recipe-local ``dataset/builder.py`` DatasetBuilder lifecycle.
    """

    def train(self, *args, **kwargs):
        """Train the TSE model.

        Raises:
            RuntimeError: If ``train_config.dataset_dir`` is not set.
        """
        self._reject_stage_args("train", args, kwargs)
        logger.info("TSESystem.train(): starting training process")

        dataset_dir = getattr(self.training_config, "dataset_dir", None)
        if dataset_dir is None:
            raise RuntimeError("training_config.dataset_dir must be set for training.")

        return super().train()
