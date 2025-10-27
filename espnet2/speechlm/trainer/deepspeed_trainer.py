"""
DeepSpeed Trainer for SpeechLM with clean and simple setup.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
import deepspeed
import wandb

from espnet2.speechlm.utils.data import to_device


logger = logging.getLogger(__name__)


class DeepSpeedTrainer:
    """DeepSpeed trainer with simple setup for SpeechLM.

    IMPORTANT: wandb is MANDATORY and must be initialized before creating this trainer.
    The trainer will raise an error if wandb.run is None.
    Wandb should always be initialized in offline mode for local-only logging.
    All training metrics, losses, and stats are logged to local wandb files.
    """

    def __init__(
        self,
        train_data_factory,
        valid_data_factories: Dict,
        model: nn.Module,
        resume_path: Optional[Path],
        output_dir: Path,
        trainer_args: Dict[str, Any],
    ):
        """Initialize DeepSpeed trainer.

        Args:
            train_data_factory: Training data iterator factory
            valid_data_factories: Dictionary of validation data factories
            model: Model to train
            resume_path: Path to checkpoint for resuming training
            output_dir: Directory for saving outputs
            trainer_args: Training configuration dictionary containing:
                - max_steps: Maximum number of training steps
                - log_interval: Steps between logging
                - val_interval: Steps between validation
                - save_interval: Steps between checkpoints
                - deepspeed_config: Path to DeepSpeed JSON configuration file (required)
        """
        self.train_data_factory = train_data_factory
        self.valid_data_factories = valid_data_factories
        self.output_dir = Path(output_dir)
        self.trainer_args = trainer_args
        (self.output_dir / "checkpoints").mkdir(exist_ok=True, parents=True)

        self.global_step = 0
        self.max_step = trainer_args["max_step"]
        self.save_interval = trainer_args["save_interval"]
        self.log_interval = trainer_args["log_interval"]

        # freeze parameters
        for t in trainer_args.get("freeze_param", []):
            for k, p in model.named_parameters():
                if k.startswith(t + ".") or k == t:
                    logger.info(f"Setting {k}.requires_grad = False")
                    p.requires_grad = False

        # Initialization
        ds_config_path = self.trainer_args["deepspeed_config"]
        with open(ds_config_path, "r") as f:
            ds_config = json.load(f)
        self.model_engine, _, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config,
        )
        logger.info("Successfully initialize DeepSpeed with configuration")
        logger.info(json.dumps(ds_config, indent=2))
        wandb.config.update({"deepspeed_config": ds_config})

        # Load checkpoint
        self._load_checkpoint(resume_path)

        # train dtype
        self.dtype = self.train_dtype(ds_config)

    def _load_checkpoint(self, resume_path: Optional[Path]) -> None:
        """Load checkpoint for resuming training."""
        checkpoint_path = None

        # Step 1: Check resume_path
        if resume_path and resume_path.exists():
            checkpoint_path = resume_path

        # Step 2: Check latest checkpoint in output_dir
        elif (self.output_dir / "checkpoints").exists():
            ckpt_dir = self.output_dir / "checkpoints"
            checkpoints = [
                d for d in ckpt_dir.iterdir() if d.is_dir() and "step_" in d.name
            ]
            if checkpoints:
                # Sort by step number and get latest
                checkpoint_path = sorted(
                    checkpoints,
                    key=lambda x: int(x.name.split("step_")[-1]),
                    reverse=True,
                )[0]

        if checkpoint_path:
            _, client_state = self.model_engine.load_checkpoint(str(checkpoint_path))
            # Restore global_step from client_state if available
            if client_state and "global_step" in client_state:
                self.global_step = client_state["global_step"]
            logger.info(
                f"Loaded checkpoint: {checkpoint_path} | step={self.global_step}"
            )
        else:
            logger.info("No checkpoint found, starting from step 0")

    def run(self) -> None:
        """Main training loop."""
        while self.global_step < self.max_step:

            self.train()

            self.valid()

            # Save checkpoint with client_state containing global_step
            client_state = {"global_step": self.global_step}
            self.model_engine.save_checkpoint(
                self.output_dir / "checkpoints" / f"step_{self.global_step}",
                client_state=client_state,
            )

    def train(self) -> None:
        """Execute one training epoch."""
        self.model_engine.train()

        iterator = self.train_data_factory.get_iterator(
            global_step=self.global_step,
            length=self.save_interval,
        )
        for batch in iterator:
            batch = to_device(batch, "cuda", dtype=self.dtype)
            out = self.model_engine(**batch)

            self.model_engine.backward(out["loss"])
            self.model_engine.step()

            # TODO(deepspeed): sync the stats across GPUs before logging
            stats = {k: float(v) for k, v in out["stats"].items()}
            stats = {f"train/{key}": value for key, value in stats.items()}
            wandb.log(stats, step=self.global_step)

            self.global_step += 1

    def valid(self) -> None:
        """Run validation on all validation datasets."""
        self.model_engine.eval()

        for name, factory in self.valid_data_factories.items():
            iterator = factory.get_iterator()

            # Collect all batch metrics
            all_stats = {}

            with torch.no_grad():
                for batch in iterator:
                    batch = to_device(batch, "cuda", dtype=self.dtype)
                    out = self.model_engine(**batch)

                    stats = {k: float(v) for k, v in out["stats"].items()}
                    for key, value in stats.items():
                        if key not in all_stats:
                            all_stats[key] = []
                        all_stats[key].append(value)

            # Compute averages and log (should be outside the batch loop)
            all_stats = {
                f"val/{name}/{key}": sum(value) / len(value)
                for key, value in all_stats.items()
            }
            wandb.log(all_stats, step=self.global_step)

    def train_dtype(self, ds_config):
        # Check if bf16 is enabled
        if ds_config.get("bf16", {}).get("enabled", False):
            dtype = torch.bfloat16
        # Check if fp16 is enabled
        elif ds_config.get("fp16", {}).get("enabled", False):
            dtype = torch.float16
        # Check if amp (automatic mixed precision) is enabled
        elif ds_config.get("amp", {}).get("enabled", False):
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        else:
            dtype = torch.float

        logger.info(f"Convert all float input data to dtype={dtype}")
        return dtype
