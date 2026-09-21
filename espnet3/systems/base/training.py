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
    model_config = OmegaConf.create(config.model)
    # ``freeze_param`` belongs to ESPnet3's Lightning wrapper.  It is applied
    # after model construction and must not be forwarded to a model constructor
    # or an ESPnet2 task's build_model arguments.
    model_config.pop("freeze_param", None)
    task = config.get("task")
    if task:
        return get_espnet_model(
            task, OmegaConf.to_container(model_config, resolve=True)
        )
    return instantiate(model_config)


def _freeze_parameters(model: torch.nn.Module, prefixes) -> None:
    """Disable gradients for parameters matching ``freeze_param`` prefixes.

    Mirrors ESPnet2's ``--freeze_param``: a prefix ``teacher`` freezes
    ``teacher`` itself and every parameter under ``teacher.``. Frozen
    parameters are skipped when the optimizer is built.
    """
    for prefix in prefixes or []:
        matched = False
        for name, param in model.named_parameters():
            if name == prefix or name.startswith(prefix + "."):
                param.requires_grad = False
                matched = True
        if not matched:
            raise ValueError(
                f"freeze_param entry {prefix!r} does not match any model parameter."
            )
        logger.info("Froze parameters under %r", prefix)


def _build_trainer(config: DictConfig) -> ESPnet3LightningTrainer:
    model = _instantiate_model(config)
    _freeze_parameters(model, config.get("freeze_param"))
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


def train(config: DictConfig) -> ESPnet3LightningTrainer:
    """Run the training loop and return the trainer that ran it.

    The returned trainer lets a caller inspect what this fit produced, for
    example the checkpoints its ``ModelCheckpoint`` callbacks kept.
    """
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
    return trainer
