from argparse import Namespace
from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import Any, Dict, List, Union

import lightning as L
import torch
from lightning.pytorch.loggers import CSVLogger
from espnetez.trainer.callbacks import get_default_callbacks
from espnetez.trainer.model import LitESPnetModel
from typeguard import typechecked


class ESPnetEZLightningTrainer:
    @typechecked
    def __init__(
        self,
        config: Union[DictConfig, Namespace, Dict[str, Any]],
        model: LitESPnetModel,
        **lightning_kwargs,
    ):
        # HP and configs
        self.config = config
        if type(self.config) is dict:
            config.update(lightning_kwargs)
            self.config = Namespace(**self.config)
        elif type(self.config) is Namespace or isinstance(self.config, DictConfig):
            for key, value in lightning_kwargs.items():
                setattr(self.config, key, value)
        else:
            raise ValueError(
                "config should be dict or Namespace, but got {}.".format(
                    type(self.config)
                )
            )

        # Check if necessary configurations are present
        assert hasattr(
            self.config, "expdir"
        ), "expdir is required to save logs and checkpoints."

        # Set random seed if required
        if getattr(self.config, "seed", None) is not None:
            assert isinstance(self.config.seed, int), "seed should be an integer"
            L.seed_everything(self.config.seed)

        # Set additional configurations that might be helpful
        torch.set_float32_matmul_precision("high")

        # Instantiate the Lightning Model
        self.model = model

        # Set strategy. Default is DDP
        if getattr(self.config, "strategy") and type(self.config.strategy) == str:
            strategy = self.config.strategy
        elif getattr(self.config, "strategy"):
            strategy = instantiate(self.config.strategy)
        else:
            logging.warning("Using default DDP strategy")
            strategy = "ddp"

        # Callbacks
        callbacks = get_default_callbacks(self.config)
        if getattr(self.config, "callbacks", None):
            assert isinstance(self.config.callbacks, list), "callbacks should be a list"
            for callback in self.config.callbacks:
                callbacks.append(instantiate(callback))

        # Set up the loggers
        loggers = []
        if getattr(self.config, "loggers", None) is not None:
            try:
                list(self.config.loggers)
            except:
                raise ValueError("loggers should be a list")
            
            for logger in self.config.loggers:
                loggers.append(instantiate(logger))
        else:
            # Use csv logger for default logging.
            loggers.append(CSVLogger(self.config.expdir, name="default_logger"))

        # Set up the trainer
        self.trainer = L.Trainer(
            callbacks=callbacks,
            strategy=strategy,
            logger=loggers,
            **lightning_kwargs,
        )

    def train(self):
        self.trainer.fit(model=self.model, ckpt_path="last")
