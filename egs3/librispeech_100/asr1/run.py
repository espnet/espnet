"""Convenience script to orchestrate dataset creation, training, and evaluation."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import List

import os


def run_command(cmd: List[str], dry_run: bool = False, *, env=None) -> None:
    print(f">>> {shlex.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def build_train_overrides(args: argparse.Namespace) -> List[str]:
    overrides: List[str] = []
    if args.train_tokenizer:
        overrides.append("runtime.train_tokenizer=true")
    if args.collect_stats:
        overrides.append("runtime.collect_stats=true")
    if args.train_overrides:
        overrides.extend(args.train_overrides)
    return overrides


def build_eval_overrides(args: argparse.Namespace) -> List[str]:
    overrides: List[str] = []
    if args.debug_sample:
        overrides.append("runtime.debug_sample=true")
    if args.eval_overrides:
        overrides.extend(args.eval_overrides)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["create_dataset", "train", "evaluate", "all"], default="all")

    parser.add_argument("--train_config", default="config", help="Hydra config name for training")
    parser.add_argument("--eval_config", default="evaluate", help="Hydra config name for decoding")

    parser.add_argument("--train_launcher", default="python", help="Executable used to launch training")
    parser.add_argument("--eval_launcher", default="python", help="Executable used to launch evaluation")

    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--input_dir", type=str, help="LibriSpeech root used for dataset creation")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for HF dataset")

    parser.add_argument("--train_tokenizer", action="store_true")
    parser.add_argument("--collect_stats", action="store_true")
    parser.add_argument("--train_overrides", nargs="*", help="Additional Hydra overrides for training")

    parser.add_argument("--debug_sample", action="store_true")
    parser.add_argument("--eval_overrides", nargs="*", help="Additional Hydra overrides for evaluation")

    args = parser.parse_args()

    recipe_root = Path(__file__).resolve().parent
    repo_root = recipe_root.parents[2]
    base_env = os.environ.copy()
    existing_pythonpath = base_env.get("PYTHONPATH", "")
    pythonpath = str(repo_root) if not existing_pythonpath else os.pathsep.join([str(repo_root), existing_pythonpath])
    base_env["PYTHONPATH"] = pythonpath


    if args.stage in ("create_dataset", "all"):
        if not args.input_dir:
            raise ValueError("--input_dir must be provided for dataset creation")
        create_cmd = [
            "python",
            "-m",
            "egs3.librispeech_100.asr1.src.data.create_dataset",
            "--input_dir",
            args.input_dir,
            "--output_dir",
            args.output_dir,
        ]
        run_command(create_cmd, dry_run=args.dry_run, env=base_env)

    if args.stage in ("train", "all"):
        train_cmd = [
            args.train_launcher,
            "-m",
            "egs3.librispeech_100.asr1.src.train",
            "--config",
            args.train_config,
        ]
        train_cmd.extend(build_train_overrides(args))
        run_command(train_cmd, dry_run=args.dry_run, env=base_env)

    if args.stage in ("evaluate", "all"):
        eval_cmd = [
            args.eval_launcher,
            "-m",
            "egs3.librispeech_100.asr1.src.evaluate",
            "--config",
            args.eval_config,
        ]
        eval_cmd.extend(build_eval_overrides(args))
        run_command(eval_cmd, dry_run=args.dry_run, env=base_env)


if __name__ == "__main__":  # pragma: no cover
    main()
