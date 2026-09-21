"""Trainer class for the espnet3 package."""

import copy
import warnings
from argparse import Namespace
from typing import Any, Dict, Union

import lightning
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from typeguard import typechecked

from espnet2.torch_utils.initialize import initialize
from espnet3.components.callbacks.default_callbacks import get_default_callbacks
from espnet3.components.modeling.lightning_module import ESPnetLightningModule


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
        model (ESPnetLightningModule): ESPnet3 LightningModule instance.
        trainer (lightning.Trainer): Underlying PyTorch Lightning trainer object.
    """

    @typechecked
    def __init__(
        self,
        model: ESPnetLightningModule = None,
        exp_dir: str = None,
        config: Union[DictConfig, Namespace, Dict[str, Any]] = None,
        best_model_criterion=None,
    ):
        """Initialize the trainer with model, configuration, and training setup.

        Sets up weight initialization, accelerator/strategy/logger/profiler/plugins,
        applies ESPnet-specific dataloader constraints, prepares callbacks, and finally
        constructs the underlying Lightning Trainer.

        Args:
            model (ESPnetLightningModule, optional): LightningModule to train.
            exp_dir (str, optional): Experiment directory for logs and checkpoints.
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

        # strategy
        self._validate_multi_optimizer_trainer_config()
        self._validate_strategy_config_compatibility()
        strategy = _get_or_initialize(self.config, "strategy", "auto")
        self._validate_strategy_compatibility(strategy)

        # logger
        logger = _get_or_initialize(self.config, "logger")

        # profiler
        profiler = _get_or_initialize(self.config, "profiler")

        # plugins
        plugins = _get_or_initialize(self.config, "plugins")

        # Callbacks
        callbacks = get_default_callbacks(
            exp_dir,
            self.config.log_every_n_steps,
            OmegaConf.to_container(best_model_criterion),
        )
        if getattr(self.config, "callbacks", None):
            assert isinstance(
                self.config.callbacks, ListConfig
            ), "callbacks should be a list"
            for callback in self.config.callbacks:
                callbacks.append(instantiate(callback))

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

        trainer_config = copy.deepcopy(self.config)
        for key in (
            "accelerator",
            "strategy",
            "logger",
            "profiler",
            "plugins",
            "callbacks",
        ):
            self._del_config_key_on(trainer_config, key)

        # Set up the trainer
        self.trainer = lightning.Trainer(
            accelerator=accelerator,
            # Temporarily disabled for the code review.
            callbacks=callbacks,
            strategy=strategy,
            logger=logger,
            profiler=profiler,
            plugins=plugins,
            **trainer_config,
        )

    def _validate_strategy_compatibility(self, strategy) -> None:
        """Reject unsupported strategies for the multiple-optimizer path only."""
        # Named `optimizers` enables the manual multi-optimizer training path.
        if getattr(self.model.config, "optimizers", None) is None:
            return

        strategy_name = type(strategy).__name__.lower()
        strategy_repr = str(strategy).lower()
        if "deepspeed" in strategy_name or "deepspeed" in strategy_repr:
            raise RuntimeError(
                "ESPnet3 does not support DeepSpeed with multiple optimizers. "
                "Use a single optimizer or switch to a supported strategy such as "
                "DDP/FSDP."
            )

    def _validate_multi_optimizer_trainer_config(self) -> None:
        """Reject trainer options that conflict with manual multi-optimizer logic."""
        if getattr(self.model.config, "optimizers", None) is None:
            return

        clip_val = getattr(self.config, "gradient_clip_val", None)
        if clip_val not in (None, 0, 0.0):
            raise AssertionError(
                "Trainer-level `gradient_clip_val` is not supported when multiple "
                "optimizers are configured. Set per-optimizer `gradient_clip_val` in "
                "`config.optimizers.<name>` instead."
            )

        clip_algorithm = getattr(self.config, "gradient_clip_algorithm", None)
        if clip_algorithm not in (None, "norm"):
            raise AssertionError(
                "Trainer-level `gradient_clip_algorithm` is not supported when "
                "multiple optimizers are configured. Set per-optimizer "
                "`gradient_clip_algorithm` in `config.optimizers.<name>` instead."
            )

    def _validate_strategy_config_compatibility(self) -> None:
        """Reject unsupported strategy configs for the multiple-optimizer path only.

        This runs before Hydra instantiates the strategy so ESPnet3 can fail fast
        on unsupported multiple-optimizer combinations such as DeepSpeed.
        """
        # Named `optimizers` enables the manual multi-optimizer training path.
        if getattr(self.model.config, "optimizers", None) is None:
            return

        strategy_cfg = getattr(self.config, "strategy", None)
        if isinstance(strategy_cfg, DictConfig):
            target = str(getattr(strategy_cfg, "_target_", "")).lower()
            if "deepspeed" in target:
                raise RuntimeError(
                    "ESPnet3 does not support DeepSpeed with multiple optimizers. "
                    "Use a single optimizer or switch to a supported "
                    "strategy such as DDP/FSDP."
                )
        elif isinstance(strategy_cfg, str) and "deepspeed" in strategy_cfg.lower():
            raise RuntimeError(
                "ESPnet3 does not support DeepSpeed with multiple optimizers. "
                "Use a single optimizer or switch to a supported strategy such as "
                "DDP/FSDP."
            )

    def _del_config_key(self, key):
        if isinstance(self.config, DictConfig) or isinstance(self.config, Namespace):
            delattr(self.config, key)
        elif isinstance(self.config, dict):
            self.config.pop(key)

    @staticmethod
    def _del_config_key_on(config, key):
        if isinstance(config, DictConfig) or isinstance(config, Namespace):
            if hasattr(config, key):
                delattr(config, key)
        elif isinstance(config, dict):
            config.pop(key, None)

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
