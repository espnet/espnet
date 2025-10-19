"""Training entrypoint for the LibriSpeech 100h recipe."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import lightning as L
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet3.parallel.parallel import set_parallel
from espnet3.task import get_espnet_model, save_espnet_config
from espnet3.trainer.model import LitESPnetModel
from espnet3.trainer.trainer import ESPnet3LightningTrainer

from .tokenizer import train_sentencepiece_from_config


def _instantiate_model(cfg: DictConfig, model_cfg: DictConfig) -> Any:
    task = cfg.get("task")
    if task:
        return get_espnet_model(task, model_cfg)
    return instantiate(model_cfg)


def _build_trainer(cfg: DictConfig, model_cfg: DictConfig) -> ESPnet3LightningTrainer:
    model = _instantiate_model(cfg, model_cfg)
    lit_model = LitESPnetModel(model, cfg)
    trainer = ESPnet3LightningTrainer(
        model=lit_model,
        expdir=cfg.expdir,
        config=cfg.trainer,
        best_model_criterion=cfg.best_model_criterion,
    )
    return trainer


def _prepare_model_config(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))


def _ensure_directories(cfg: DictConfig) -> None:
    Path(cfg.expdir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.tokenizer_dir).mkdir(parents=True, exist_ok=True)
    if hasattr(cfg, "statsdir"):
        Path(cfg.statsdir).mkdir(parents=True, exist_ok=True)


def train(cfg: DictConfig) -> None:
    """Main training loop."""

    _ensure_directories(cfg)

    should_train_tokenizer = cfg.runtime.get("train_tokenizer") or bool(OmegaConf.select(cfg, "tokenizer.train", default=False))
    if should_train_tokenizer:
        train_sentencepiece_from_config(cfg)

    if cfg.get("parallel"):
        set_parallel(cfg.parallel)

    if cfg.get("seed") is not None:
        L.seed_everything(int(cfg.seed), workers=True)

    torch.set_float32_matmul_precision("high")

    model_cfg = _prepare_model_config(cfg)

    normalize = model_cfg.pop("normalize", None) if cfg.runtime.get("collect_stats") else None
    normalize_conf = (
        model_cfg.pop("normalize_conf", None) if cfg.runtime.get("collect_stats") else None
    )

    trainer = _build_trainer(cfg, model_cfg)

    if cfg.runtime.get("collect_stats"):
        trainer.collect_stats()
        if normalize is not None:
            model_cfg["normalize"] = normalize
        if normalize_conf is not None:
            model_cfg["normalize_conf"] = normalize_conf
        trainer = _build_trainer(cfg, model_cfg)

    fit_kwargs: Dict[str, Any] = {}
    if hasattr(cfg, "fit") and cfg.fit:
        fit_kwargs = OmegaConf.to_container(cfg.fit, resolve=True)

    trainer.fit(**fit_kwargs)

    task = cfg.get("task")
    if task:
        save_espnet_config(task, cfg, cfg.expdir)


def main(config_name: str = "config", overrides: list[str] | None = None) -> None:
    """Hydra-style entrypoint that mirrors lightning-hydra-template usage."""

    overrides = overrides or []
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name=config_name, overrides=overrides)
    train(cfg)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config", help="Name of the Hydra config to load")
    parser.add_argument(
        "overrides",
        nargs="*",
        default=[],
        help="Optional Hydra-style overrides (e.g. data=debug runtime.train_tokenizer=true)",
    )
    args = parser.parse_args()
    main(config_name=args.config, overrides=args.overrides)
