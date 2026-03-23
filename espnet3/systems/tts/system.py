"""TTS system implementation with local GAN-TTS trainer selection."""

from __future__ import annotations

import logging
import time
from importlib import import_module
from pathlib import Path
from typing import Any, Dict

import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet3.components.modeling.lightning_module import ESPnetLightningModule
from espnet3.components.trainers.trainer import ESPnet3LightningTrainer
from espnet3.parallel.parallel import set_parallel
from espnet3.systems.base.system import BaseSystem
from espnet3.systems.tts.models.gan_model import GANTTSLightningModule
from espnet3.utils.task_utils import get_espnet_model, save_espnet_config

logger = logging.getLogger(__name__)


def load_function(path: str):
    """Load a callable from a dotted module path."""
    module_path, func_name = path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, func_name)


def _instantiate_model(config: DictConfig) -> Any:
    task = config.get("task")
    if task:
        model_config = OmegaConf.to_container(config.model, resolve=True)
        return get_espnet_model(task, model_config)
    return instantiate(config.model)


class TTSSystem(BaseSystem):
    """TTS-specific system with local trainer customization."""

    def create_dataset(self, *args, **kwargs):
        self._reject_stage_args("create_dataset", args, kwargs)
        start = time.perf_counter()
        config = getattr(self.training_config, "create_dataset", None)
        if config is None or not getattr(config, "func", None):
            raise RuntimeError(
                "training_config.create_dataset.func must be set to run create_dataset"
            )
        fn = load_function(config.func)
        extra = {k: v for k, v in config.items() if k != "func"}
        result = fn(**extra)
        logger.info(
            "Dataset creation completed in %.2fs using %s",
            time.perf_counter() - start,
            config.func,
        )
        return result

    def _ensure_directories(self) -> None:
        config = self.training_config
        Path(config.exp_dir).mkdir(parents=True, exist_ok=True)
        if hasattr(config, "stats_dir"):
            Path(config.stats_dir).mkdir(parents=True, exist_ok=True)

    def _build_trainer(self) -> ESPnet3LightningTrainer:
        config = self.training_config
        model = _instantiate_model(config)
        if isinstance(model, AbsGANESPnetModel):
            lit_model = GANTTSLightningModule(model, config)
        else:
            lit_model = ESPnetLightningModule(model, config)
        return ESPnet3LightningTrainer(
            model=lit_model,
            exp_dir=config.exp_dir,
            config=config.trainer,
            best_model_criterion=config.best_model_criterion,
        )

    def _prepare_training_runtime(self) -> None:
        config = self.training_config
        self._ensure_directories()

        if config.get("parallel"):
            set_parallel(config.parallel)

        if config.get("seed") is not None:
            L.seed_everything(int(config.seed), workers=True)

        torch.set_float32_matmul_precision("high")

    def collect_stats(self, *args, **kwargs):
        self._reject_stage_args("collect_stats", args, kwargs)
        start = time.perf_counter()
        self._prepare_training_runtime()

        # Preserve `normalize: null` from recipe configs.
        trainer = self._build_trainer()
        trainer.collect_stats()
        logger.info(
            "Collect stats finished in %.2fs | exp_dir=%s stats_dir=%s",
            time.perf_counter() - start,
            self.training_config.exp_dir,
            getattr(self.training_config, "stats_dir", None),
        )

    def train(self, *args, **kwargs):
        self._reject_stage_args("train", args, kwargs)
        start = time.perf_counter()
        self._prepare_training_runtime()

        task = self.training_config.get("task")
        if task:
            save_espnet_config(task, self.training_config, self.training_config.exp_dir)

        trainer = self._build_trainer()

        fit_kwargs: Dict[str, Any] = {}
        if hasattr(self.training_config, "fit") and self.training_config.fit:
            fit_kwargs = OmegaConf.to_container(self.training_config.fit, resolve=True)

        trainer.fit(**fit_kwargs)
        logger.info(
            "Training finished in %.2fs | exp_dir=%s model=%s",
            time.perf_counter() - start,
            self.training_config.exp_dir,
            (
                self.training_config.model.get("_target_", None)
                if isinstance(self.training_config.model, DictConfig)
                else None
            ),
        )
