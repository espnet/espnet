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
from espnet3.components.training.trainer import ESPnet3LightningTrainer
from espnet3.parallel.parallel import set_parallel
from espnet3.utils.task_utils import get_espnet_model, save_espnet_config

logger = logging.getLogger(__name__)


def _instantiate_model(cfg: DictConfig) -> Any:
    task = cfg.get("task")
    if task:
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        return get_espnet_model(task, model_cfg)
    return instantiate(cfg.model)


def _build_trainer(cfg: DictConfig) -> ESPnet3LightningTrainer:
    model = _instantiate_model(cfg)
    logger.info("Model:\n%s", model)
    lit_model = ESPnetLightningModule(model, cfg)
    trainer = ESPnet3LightningTrainer(
        model=lit_model,
        expdir=cfg.exp_dir,
        config=cfg.trainer,
        best_model_criterion=cfg.best_model_criterion,
    )
    return trainer


def _ensure_directories(cfg: DictConfig) -> None:
    Path(cfg.exp_dir).mkdir(parents=True, exist_ok=True)
    if hasattr(cfg, "stats_dir"):
        Path(cfg.stats_dir).mkdir(parents=True, exist_ok=True)


def collect_stats(cfg: DictConfig) -> None:
    """Collect statistics required by the training pipeline."""
    _ensure_directories(cfg)
    start = time.perf_counter()

    if cfg.get("parallel"):
        set_parallel(cfg.parallel)

    if cfg.get("seed") is not None:
        L.seed_everything(int(cfg.seed), workers=True)

    torch.set_float32_matmul_precision("high")

    if "normalize" in cfg.model:
        cfg.model.pop("normalize")
    if "normalize_conf" in cfg.model:
        cfg.model.pop("normalize_conf")

    trainer = _build_trainer(cfg)
    trainer.collect_stats()
    logger.info(
        "Collect stats finished in %.2fs | exp_dir=%s stats_dir=%s",
        time.perf_counter() - start,
        cfg.exp_dir,
        getattr(cfg, "stats_dir", None),
    )


def train(cfg: DictConfig) -> None:
    """Run the training loop."""
    _ensure_directories(cfg)
    start = time.perf_counter()

    if cfg.get("parallel"):
        set_parallel(cfg.parallel)

    if cfg.get("seed") is not None:
        L.seed_everything(int(cfg.seed), workers=True)

    torch.set_float32_matmul_precision("high")

    task = cfg.get("task")
    if task:
        save_espnet_config(task, cfg, cfg.exp_dir)

    trainer = _build_trainer(cfg)

    fit_kwargs: Dict[str, Any] = {}
    if hasattr(cfg, "fit") and cfg.fit:
        fit_kwargs = OmegaConf.to_container(cfg.fit, resolve=True)

    trainer.fit(**fit_kwargs)
    logger.info(
        "Training finished in %.2fs | exp_dir=%s model=%s",
        time.perf_counter() - start,
        cfg.exp_dir,
        cfg.model.get("_target_", None) if isinstance(cfg.model, DictConfig) else None,
    )
