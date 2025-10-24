"""Convenience script to orchestrate dataset creation, training, and evaluation."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import List
from distutils.util import strtobool

import os
import lightning as L
from espnet3.utils.config import load_config_with_defaults


def run_command(cmd: List[str], dry_run: bool = False, *, env=None) -> None:
    print(f">>> {shlex.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def build_eval_overrides(args: argparse.Namespace) -> List[str]:
    overrides: List[str] = []
    if args.debug_sample:
        overrides.append("runtime.debug_sample=true")
    if args.eval_overrides:
        overrides.extend(args.eval_overrides)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",
                        choices=["create_dataset", "train", "evaluate", "all"],
                        nargs="+",
                        default=["all"]
    )

    parser.add_argument("--train_config", default="config", help="Hydra config name for training")
    parser.add_argument("--eval_config", default="evaluate", help="Hydra config name for decoding")

    parser.add_argument("--launcher", default="python", help="Executable used to launch training")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--dataset_dir", type=str, help="LibriSpeech root used for dataset creation")

    parser.add_argument("--collect_stats", type=strtobool, default=True,
                        help="Running collect-stats stage.")

    parser.add_argument("--debug_sample", action="store_true")

    args = parser.parse_args()

    recipe_root = Path(__file__).resolve().parent
    repo_root = recipe_root.parents[2]
    base_env = os.environ.copy()
    existing_pythonpath = base_env.get("PYTHONPATH", "")
    pythonpath = str(repo_root) if not existing_pythonpath else os.pathsep.join([str(repo_root), existing_pythonpath])
    base_env["PYTHONPATH"] = pythonpath

    if "create_dataset" in args.stage or "all" in args.stage:
        if not args.dataset_dir:
            raise ValueError("--dataset_dir must be provided for dataset creation")
        # Load configs for handling params from config file
        config = load_config_with_defaults(args.train_config)
        create_cmd = [
            "python",
            "-m",
            "src.bin.create_dataset",
            "--dataset_dir",
            args.dataset_dir,
            "--output_dir",
            config.datadir,
        ]
        run_command(create_cmd, dry_run=args.dry_run, env=base_env)

        create_cmd = [
            "python",
            "-m",
            "src.bin.train_tokenizer",
            "--dataset_dir",
            args.dataset_dir,
            "--save_path",
            config.tokenizer.save_path,
            "--vocab_size",
            str(config.tokenizer.vocab_size),
            "--model_type",
            config.tokenizer.model_type,
        ]
        run_command(create_cmd, dry_run=args.dry_run, env=base_env)

    if "train" in args.stage or "all" in args.stage:
        train_cmd = [
            args.launcher,
            "-m",
            "template.train",
            "--config",
            args.train_config,
            "--collect_stats",
            str(args.collect_stats),
        ]
        run_command(train_cmd, dry_run=args.dry_run, env=base_env)

    if "evaluate" in args.stage or "all" in args.stage:
        eval_cmd = [
            args.launcher,
            "-m",
            "src.bin.decode",
            "--config",
            args.eval_config,
        ]
        run_command(eval_cmd, dry_run=args.dry_run, env=base_env)
        eval_cmd = [
            args.launcher,
            "-m",
            "template.score",
            "--config",
            args.eval_config,
        ]
        run_command(eval_cmd, dry_run=args.dry_run, env=base_env)


if __name__ == "__main__":  # pragma: no cover
    main()
