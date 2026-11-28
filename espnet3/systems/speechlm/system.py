"""ESPnet3 stages for the native SpeechLM training engine."""

import os
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

from omegaconf import ListConfig

from espnet2.speechlm.dataloader.task_conf import parse_data_specifier
from espnet2.speechlm.utils.checkpoint import latest_checkpoint
from espnet3.contract.stages import StageSpec
from espnet3.systems.base.system import BaseSystem


class SpeechLMSystem(BaseSystem):
    """Train SpeechLM from prepared manifests and run ESPnet3 inference.

    The training config points to the engine's YAML with ``speechlm_config``.
    Both stages accept the engine's registered and unregistered data specifiers
    as lists, one per split and specifier type. The engine owns data sharding,
    model construction, training, and checkpoints. Fine-tuning recipes can
    set ``require_checkpoint`` to require initial weights on a fresh run and
    resume from the experiment's checkpoint on subsequent launches.
    """

    stages: ClassVar[tuple[StageSpec, ...]] = (
        StageSpec(name="collect_stats", config="training", log_dir="stats_dir"),
        StageSpec(name="train", config="training", log_dir="exp_dir"),
        StageSpec(name="infer", config="inference", log_dir="inference_dir"),
    )

    def _native_args(self, *, require_both_splits):
        config = self.training_config
        speechlm_config = self._get_required_config(
            config, "speechlm_config", "Set speechlm_config to the SpeechLM YAML path."
        )
        if not Path(speechlm_config).is_file():
            raise FileNotFoundError(
                f"SpeechLM config does not exist: {speechlm_config}"
            )
        data_args = []
        stats_sources = {}
        for split in ("train", "valid"):
            split_args = []
            for kind in ("registered", "unregistered"):
                key = f"{split}_{kind}_specifier"
                specifiers = config.get(key, [])
                if not isinstance(specifiers, (list, ListConfig)) or not all(
                    isinstance(spec, str) and spec.strip() for spec in specifiers
                ):
                    raise ValueError(f"{key} must be a list of nonempty specifiers")
                for spec in specifiers:
                    parsed = parse_data_specifier(spec, registered=kind == "registered")
                    task, name = parsed[:2]
                    source = None
                    if kind == "unregistered":
                        source = Path(parsed[2]).resolve()
                        if not source.is_file():
                            raise FileNotFoundError(
                                f"Missing SpeechLM manifest: {source}"
                            )
                    # The native sampler shares this filename across splits. Reusing
                    # one source with different sampling factors is supported.
                    path = Path(config.stats_dir) / f"stats_{task}_{name}.jsonl"
                    origin = (task, name, source)
                    if stats_sources.setdefault(path, origin) != origin:
                        raise ValueError(f"Conflicting SpeechLM datasets for {path}")
                    split_args.extend([f"--{split}-{kind}-specifier", spec])
            if require_both_splits and not split_args:
                raise ValueError(f"Provide at least one {split} data specifier")
            data_args.extend(split_args)
        if not data_args:
            raise ValueError(
                "Provide at least one data specifier to collect statistics"
            )
        if require_both_splits:
            for path in stats_sources:
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(f"Run collect_stats to produce {path}")
        return ["--train-config", str(speechlm_config), *data_args]

    def collect_stats(self, *args, **kwargs):
        """Write the per-example length JSONL files consumed by the native sampler."""
        self._reject_stage_args("collect_stats", args, kwargs)
        native_args = self._native_args(require_both_splits=False)
        config = self.training_config
        if type(config.stats_num_workers) is not int or config.stats_num_workers < 1:
            raise ValueError("stats_num_workers must be a positive integer")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "espnet2.speechlm.bin.prepare_length_stats",
                "--output-dir",
                str(config.stats_dir),
                "--num-workers",
                str(config.stats_num_workers),
                *native_args,
            ],
            check=True,
        )

    def train(self, *args, **kwargs):
        """Launch a single-node SpeechLM job with one worker per configured GPU."""
        self._reject_stage_args("train", args, kwargs)
        native_args = self._native_args(require_both_splits=True)
        config = self.training_config
        if type(config.num_device) is not int or config.num_device < 1:
            raise ValueError("num_device must be a positive integer")
        initializer = config.get("resume_path")
        require_checkpoint = config.get("require_checkpoint", False)
        if require_checkpoint:
            # A complete output checkpoint restores optimizer, scheduler and step.
            # Passing --resume-path would instead reset them, even on a retry.
            if latest_checkpoint(self.exp_dir) is not None:
                initializer = None
            elif not initializer:
                raise ValueError(
                    "Set resume_path to initial weights for a fresh fine-tuning run; "
                    f"no complete checkpoint exists under {self.exp_dir}/checkpoints"
                )
        if initializer:
            path = Path(initializer)
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint does not exist: {initializer}")
            if (
                require_checkpoint
                and path.is_dir()
                and not (path / ".metadata").is_file()
            ):
                raise ValueError(f"Incomplete distributed checkpoint: {path}")
            native_args.extend(["--resume-path", str(initializer)])
        if config.save_loader_state:
            native_args.append("--save-loader-state")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node",
                str(config.num_device),
                "--module",
                "espnet2.speechlm.bin.train",
                "--output-dir",
                str(config.exp_dir),
                "--stats-dir",
                str(config.stats_dir),
                "--wandb-mode",
                config.wandb_mode,
                *native_args,
            ],
            # Dataset resampling hashes (task, dataset) on each rank.
            env={**os.environ, "PYTHONHASHSEED": "0"},
            check=True,
        )
