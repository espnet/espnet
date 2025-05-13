from argparse import Namespace
from typing import Any, Dict, List, Tuple, Union

import lightning as L
import torch
import torch.nn as nn
from hydra.utils import instantiate
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, ListConfig, OmegaConf
from typeguard import typechecked

from espnet3.trainer.callbacks import get_default_callbacks
from espnet3.trainer.model import LitESPnetModel


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def get_or_initialize(config, item_name: str = None, default=None) -> Any:
    if item_name is not None:
        item = getattr(config, item_name, default)
    else:
        item = config

    if type(item) == DictConfig:
        return instantiate(item)
    elif type(item) == ListConfig:
        return [get_or_initialize(c) for c in item]
    else:
        return item


class ESPnetEZLightningTrainer:
    @typechecked
    def __init__(
        self,
        model: LitESPnetModel = None,
        expdir: str = None,
        config: Union[DictConfig, Namespace, Dict[str, Any]] = None,
        best_model_criterion=None,
    ):
        assert model is not None, "model must be provided."
        assert expdir is not None, "expdir must be provided."
        assert config is not None, "config must be provided."
        if best_model_criterion is None:
            best_model_criterion = ListConfig([("valid/loss", 3, "min")])

        # HP and configs
        self.config = config

        # Instantiate the Lightning Model
        self.model = model
        init_weights(self.model)

        # Accelerator
        accelerator = get_or_initialize(self.config, "accelerator", "auto")
        if accelerator != "auto":
            self.config.pop("accelerator")

        # strategy
        strategy = get_or_initialize(self.config, "strategy", "auto")
        if strategy != "auto":
            self.config.pop("strategy")

        # logger
        logger = get_or_initialize(self.config, "logger")
        if logger is not None:
            self.config.pop("logger")

        # profiler
        profiler = get_or_initialize(self.config, "profiler")
        if profiler is not None:
            self.config.pop("profiler")

        # plugins
        plugins = get_or_initialize(self.config, "plugins")
        if plugins is not None:
            self.config.pop("plugins")

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
            self.config.pop("callbacks")

        # Set up the trainer
        self.trainer = L.Trainer(
            accelerator=accelerator,
            callbacks=callbacks,
            strategy=strategy,
            logger=logger,
            profiler=profiler,
            plugins=plugins,
            **self.config,
        )

    def fit(self, *args, **kwargs):
        self.trainer.fit(
            *args,
            model=self.model,
            **kwargs,
        )

    def validate(self, *args, **kwargs):
        return self.trainer.validate(
            *args,
            model=self.model,
            **kwargs,
        )

    def collect_stats(self, *args, **kwargs):
        return self.model.collect_stats(*args, **kwargs)
