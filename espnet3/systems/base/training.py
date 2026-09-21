"""Training entrypoint for ESPnet3 systems."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict

import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet3.components.modeling.lightning_module import ESPnetLightningModule
from espnet3.components.trainers.trainer import ESPnet3LightningTrainer
from espnet3.parallel.parallel import set_parallel
from espnet3.utils.task_utils import get_espnet_model, save_espnet_config

logger = logging.getLogger(__name__)


def _instantiate_model(config: DictConfig) -> Any:
    task = config.get("task")
    if task:
        model_config = OmegaConf.to_container(config.model, resolve=True)
        return get_espnet_model(task, model_config)
    return instantiate(config.model)


def _build_trainer(config: DictConfig) -> ESPnet3LightningTrainer:
    model = _instantiate_model(config)
    lit_model = ESPnetLightningModule(model, config)
    trainer = ESPnet3LightningTrainer(
        model=lit_model,
        exp_dir=config.exp_dir,
        config=config.trainer,
        best_model_criterion=config.best_model_criterion,
    )
    return trainer


def _ensure_directories(config: DictConfig) -> None:
    Path(config.exp_dir).mkdir(parents=True, exist_ok=True)
    if hasattr(config, "stats_dir"):
        Path(config.stats_dir).mkdir(parents=True, exist_ok=True)


def collect_stats(config: DictConfig) -> None:
    """Collect statistics required by the training pipeline."""
    _ensure_directories(config)
    start = time.perf_counter()

    if config.get("parallel"):
        set_parallel(config.parallel)

    if config.get("seed") is not None:
        L.seed_everything(int(config.seed), workers=True)

    torch.set_float32_matmul_precision("high")

    if "normalize" in config.model:
        config.model.pop("normalize")
    if "normalize_conf" in config.model:
        config.model.pop("normalize_conf")

    trainer = _build_trainer(config)
    trainer.collect_stats()
    logger.info(
        "Collect stats finished in %.2fs | exp_dir=%s stats_dir=%s",
        time.perf_counter() - start,
        config.exp_dir,
        getattr(config, "stats_dir", None),
    )


def train(config: DictConfig) -> None:
    """Run the training loop."""
    _ensure_directories(config)
    start = time.perf_counter()

    if config.get("parallel"):
        set_parallel(config.parallel)

    if config.get("seed") is not None:
        L.seed_everything(int(config.seed), workers=True)

    torch.set_float32_matmul_precision("high")

    task = config.get("task")
    if task:
        save_espnet_config(task, config, config.exp_dir)

    trainer = _build_trainer(config)

    fit_kwargs: Dict[str, Any] = {}
    if hasattr(config, "fit") and config.fit:
        fit_kwargs = OmegaConf.to_container(config.fit, resolve=True)

    trainer.fit(**fit_kwargs)
    logger.info(
        "Training finished in %.2fs | exp_dir=%s model=%s",
        time.perf_counter() - start,
        config.exp_dir,
        (
            config.model.get("_target_", None)
            if isinstance(config.model, DictConfig)
            else None
        ),
    )
