"""Training entrypoint for the LibriSpeech 100h recipe."""

from __future__ import annotations

from distutils.util import strtobool
from pathlib import Path
from typing import Any, Dict

import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet3.base.model.model import LitESPnetModel
from espnet3.base.task import get_espnet_model, save_espnet_config
from espnet3.base.trainer.trainer import ESPnet3LightningTrainer
from espnet3.parallel.parallel import set_parallel
from espnet3.utils.config import load_config_with_defaults


def _instantiate_model(cfg: DictConfig) -> Any:
    task = cfg.get("task")
    if task:
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        return get_espnet_model(task, model_cfg)
    return instantiate(cfg.model)


def _build_trainer(cfg: DictConfig) -> ESPnet3LightningTrainer:
    model = _instantiate_model(cfg)
    lit_model = LitESPnetModel(model, cfg)
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


def train(cfg: DictConfig, collect_stats: bool) -> None:
    """Main training loop."""

    _ensure_directories(cfg)

    if cfg.get("parallel"):
        set_parallel(cfg.parallel)

    if cfg.get("seed") is not None:
        L.seed_everything(int(cfg.seed), workers=True)

    torch.set_float32_matmul_precision("high")

    normalize = None
    normalize_conf = None
    if collect_stats:
        if "normalize" in cfg.model:
            normalize = cfg.model.pop("normalize")
        if "normalize_conf" in cfg.model:
            normalize_conf = cfg.model.pop("normalize_conf")

    trainer = _build_trainer(cfg)

    if collect_stats:
        trainer.collect_stats()
        if normalize is not None:
            cfg.model["normalize"] = normalize
        if normalize_conf is not None:
            cfg.model["normalize_conf"] = normalize_conf
        trainer = _build_trainer(cfg)

    fit_kwargs: Dict[str, Any] = {}
    if hasattr(cfg, "fit") and cfg.fit:
        fit_kwargs = OmegaConf.to_container(cfg.fit, resolve=True)

    trainer.fit(**fit_kwargs)

    task = cfg.get("task")
    if task:
        save_espnet_config(task, cfg, cfg.exp_dir)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config", help="Name of the Hydra config to load"
    )
    parser.add_argument(
        "--collect_stats",
        type=strtobool,
        default=True,
        help="Flag to run collect-stats stage",
    )
    args = parser.parse_args()
    config = load_config_with_defaults(args.config)
    train(config, args.collect_stats)
