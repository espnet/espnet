"""Trainer class for the espnet3 package."""

import warnings
from argparse import Namespace
from typing import Any, Dict, Union

import lightning
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from typeguard import typechecked

from espnet2.torch_utils.initialize import initialize
from espnet3.trainer.callbacks import get_default_callbacks
from espnet3.trainer.model import LitESPnetModel


def _get_or_initialize(config, item_name: str = None, default=None) -> Any:
    if item_name is not None:
        item = getattr(config, item_name, default)
    else:
        item = config

    if isinstance(item, DictConfig):
        return instantiate(item)
    elif isinstance(item, ListConfig):
        return [_get_or_initialize(c) for c in item]
    else:
        return item


class ESPnet3LightningTrainer:
    """A wrapper around Lightning's Trainer to provide ESPnet3-specific integration.

    This trainer ensures compatibility with ESPnet's dataloader, callbacks,
    and configuration system. It initializes the model, handles weight
    initialization, sets up the training strategy, logger, plugins, and
    integrates with ESPnet-specific callbacks and samplers.

    Attributes:
        config (Union[DictConfig, Namespace, Dict[str, Any]]): Training configuration.
        model (LitESPnetModel): ESPnet3 Lightning model instance.
        trainer (lightning.Trainer): Underlying PyTorch Lightning trainer object.
    """

    @typechecked
    def __init__(
        self,
        model: LitESPnetModel = None,
        expdir: str = None,
        config: Union[DictConfig, Namespace, Dict[str, Any]] = None,
        best_model_criterion=None,
    ):
        """Initialize the trainer with model, configuration, and training setup.

        Sets up weight initialization, accelerator/strategy/logger/profiler/plugins,
        applies ESPnet-specific dataloader constraints, prepares callbacks, and finally
        constructs the underlying Lightning Trainer.

        Args:
            model (LitESPnetModel, optional): Lightning model to train.
            expdir (str, optional): Experiment directory for logs and checkpoints.
            config (DictConfig | Namespace | Dict[str, Any], optional): Training config.
            best_model_criterion (ListConfig, optional): Criteria for selecting ckpt.
        """
        if best_model_criterion is None:
            best_model_criterion = ListConfig([("valid/loss", 3, "min")])

        # HP and configs
        self.config = config

        # Instantiate the Lightning Model
        self.model = model
        if getattr(self.config, "init", None) is not None:
            initialize(self.model, self.config.init)

        # Accelerator
        accelerator = _get_or_initialize(self.config, "accelerator", "auto")
        if hasattr(self.config, "accelerator"):
            self._del_config_key("accelerator")

        # strategy
        strategy = _get_or_initialize(self.config, "strategy", "auto")
        if hasattr(self.config, "strategy"):
            self._del_config_key("strategy")

        # logger
        logger = _get_or_initialize(self.config, "logger")
        if logger is not None:
            self._del_config_key("logger")

        # profiler
        profiler = _get_or_initialize(self.config, "profiler")
        if profiler is not None:
            self._del_config_key("profiler")

        # plugins
        plugins = _get_or_initialize(self.config, "plugins")
        if plugins is not None:
            self._del_config_key("plugins")

        # Callbacks
        callbacks = get_default_callbacks(
            expdir,
            self.config.log_every_n_steps,
            OmegaConf.to_container(best_model_criterion),
        )
        if getattr(self.config, "callbacks", None):
            assert isinstance(
                self.config.callbacks, ListConfig
            ), "callbacks should be a list"
            for callback in self.config.callbacks:
                callbacks.append(instantiate(callback))
            self._del_config_key("callbacks")

        # Since espnet's sampler requires to set the following configs:
        # Reload dataloaders every epoch to reuse ESPnet's dataloader
        # reload_dataloaders_every_n_epochs=1
        # ESPnet's dataloader already shards the dataset based on distributed setups
        # use_distributed_sampler=False
        if hasattr(self.model.config.dataloader.train, "iter_factory"):
            if (
                hasattr(self.config, "reload_dataloaders_every_n_epochs")
                and self.config.reload_dataloaders_every_n_epochs != 1
            ):
                warnings.warn(
                    "ESPnet's dataloader requires to set"
                    "reload_dataloaders_every_n_epochs = 1. "
                    "Override the config to 1."
                )
            if (
                hasattr(self.model.config, "use_distributed_sampler")
                and not self.config.use_distributed_sampler
            ):
                warnings.warn(
                    "ESPnet's dataloader requires to set"
                    "use_distributed_sampler to False. "
                    "Override the config to False."
                )

            self.config.reload_dataloaders_every_n_epochs = 1

        # Check training dataloader and if it is using the espnet's sampler
        # then we had to set the distributed_sampler to False.
        if self.model.is_espnet_sampler:
            self.config.use_distributed_sampler = False

        # Set up the trainer
        self.trainer = lightning.Trainer(
            accelerator=accelerator,
            # Temporarily disabled for the code review.
            callbacks=callbacks,
            strategy=strategy,
            logger=logger,
            profiler=profiler,
            plugins=plugins,
            **self.config,
        )

    def _del_config_key(self, key):
        if isinstance(self.config, DictConfig) or isinstance(self.config, Namespace):
            delattr(self.config, key)
        elif isinstance(self.config, dict):
            self.config.pop(key)

    def fit(self, *args, **kwargs):
        """Start the training loop using Lightning's fit method.

        Args:
            *args: Positional arguments passed to `trainer.fit()`.
            **kwargs: Keyword arguments passed to `trainer.fit()`.

        Note:
            Always uses the internally stored model (`self.model`) when calling `fit`.
        """
        self.trainer.fit(
            *args,
            model=self.model,
            **kwargs,
        )

    def validate(self, *args, **kwargs):
        """Run validation using Lightning's validate method.

        Args:
            *args: Positional arguments passed to `trainer.validate()`.
            **kwargs: Keyword arguments passed to `trainer.validate()`.

        Returns:
            List[Dict[str, Any]]: Validation results.
        """
        return self.trainer.validate(
            *args,
            model=self.model,
            **kwargs,
        )

    def collect_stats(self, *args, **kwargs):
        """Collect dataset statistics with the espnet-3's parallel package.

        Args:
            *args: Positional arguments passed to `model.collect_stats()`.
            **kwargs: Keyword arguments passed to `model.collect_stats()`.

        """
        return self.model.collect_stats(*args, **kwargs)
