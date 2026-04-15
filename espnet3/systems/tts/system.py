"""TTS system implementation with local GAN-TTS trainer selection."""

from __future__ import annotations

import logging
import time
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
from espnet3.systems.tts.gan_trainer import build_gan_trainer
from espnet3.utils.task_utils import get_espnet_model, save_espnet_config

logger = logging.getLogger(__name__)


def _instantiate_model(config: DictConfig) -> Any:
    task = config.get("task")
    if task:
        model_config = OmegaConf.to_container(config.model, resolve=True)
        return get_espnet_model(task, model_config)
    return instantiate(config.model)


class TTSSystem(BaseSystem):
    """TTS-specific system with local trainer customization."""

    def _ensure_directories(self) -> None:
        config = self.training_config
        Path(config.exp_dir).mkdir(parents=True, exist_ok=True)
        if hasattr(config, "stats_dir"):
            Path(config.stats_dir).mkdir(parents=True, exist_ok=True)

    def _build_trainer(self) -> ESPnet3LightningTrainer:
        config = self.training_config
        model = _instantiate_model(config)
        if isinstance(model, AbsGANESPnetModel):
            return build_gan_trainer(config, model)

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
        """Run the collect_stats stage using the configured trainer.

        Prepares the training runtime (directories, parallelism, seed), then
        delegates to the trainer's ``collect_stats`` method.  Positional and
        keyword stage arguments are rejected to avoid silent misconfiguration.

        Args:
            *args: Must be empty.  Passing any positional argument raises
                ``ValueError`` via ``_reject_stage_args``.
            **kwargs: Must be empty.  Passing any keyword argument raises
                ``ValueError`` via ``_reject_stage_args``.

        Returns:
            None

        Raises:
            ValueError: If any positional or keyword arguments are passed.

        Notes:
            The ``normalize: null`` pattern from recipe configs is intentionally
            preserved — no normalization is applied during stats collection.

        Examples:
            >>> from omegaconf import OmegaConf
            >>> cfg = OmegaConf.create({"exp_dir": "/tmp/exp"})
            >>> system = TTSSystem(training_config=cfg)
            >>> system.collect_stats()  # runs stats collection end-to-end
        """
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
        """Run the training stage using the configured trainer.

        Prepares the runtime, optionally saves the ESPnet config, then calls
        ``trainer.fit`` with any keyword arguments drawn from
        ``training_config.fit``.  For GAN models, a ``GANTTSLightningTrainer``
        is used automatically; for all other models, ``ESPnet3LightningTrainer``
        is used.

        Args:
            *args: Must be empty.  Passing any positional argument raises
                ``ValueError`` via ``_reject_stage_args``.
            **kwargs: Must be empty.  Passing any keyword argument raises
                ``ValueError`` via ``_reject_stage_args``.

        Returns:
            None

        Raises:
            ValueError: If any positional or keyword arguments are passed.

        Notes:
            ``training_config.fit`` is forwarded verbatim to ``trainer.fit``.
            Common keys include ``max_epochs``, ``ckpt_path``, etc.

        Examples:
            >>> from omegaconf import OmegaConf
            >>> cfg = OmegaConf.create({
            ...     "exp_dir": "/tmp/exp",
            ...     "task": "tts",
            ...     "model": {"_target_": "my.Model"},
            ...     "fit": {"max_epochs": 10},
            ... })
            >>> system = TTSSystem(training_config=cfg)
            >>> system.train()  # trains the model for 10 epochs
        """
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
