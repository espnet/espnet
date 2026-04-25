#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-processing inference script for SpeechLM with data sharding."""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torch.multiprocessing as mp
import yaml

from espnet2.speechlm.dataloader.iterator import DataIteratorFactory
from espnet2.speechlm.model import _all_job_types
from espnet2.speechlm.utils.data import to_device


def get_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="SpeechLM Multi-Processing Inference Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--train-config",
        type=Path,
        required=True,
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--inference-config",
        type=Path,
        required=True,
        help="Path to inference configuration file",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint to load",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/inference_mp"),
        help="Directory to save inference results",
    )
    parser.add_argument(
        "--test-unregistered-specifier",
        type=str,
        default=None,
        help="Unregistered test data specifier " "(e.g., 'asr:librispeech:test.json')",
    )
    parser.add_argument(
        "--test-registered-specifier",
        type=str,
        default=None,
        help="Registered test data specifier " "(e.g., 'asr:librispeech')",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for inference",
    )
    parser.add_argument(
        "--rank",
        type=int,
        help="GPU rank in the whole inference job",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        help="number of GPUs in the whole inference job",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible inference",
    )
    parser.add_argument(
        "--add-generation-prompt",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=None,
        help="Whether to add generation prompt",
    )

    return parser


def setup_worker_logger(rank: int) -> logging.Logger:
    """Set up logger for worker process.

    Args:
        rank: Worker rank/ID

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f"inference_worker_{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        f"[Worker-{rank}] [%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint.

    Args:
        model: The model instance to load weights into.
        checkpoint_path: Path to the checkpoint file containing model weights.

    Returns:
        The model instance with loaded weights.

    Raises:
        KeyError: If 'module' key is not found in checkpoint.
        RuntimeError: If checkpoint loading fails or state dict doesn't match.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["module"]
    model.load_state_dict(state_dict, strict=True)
    return model


def split_specifiers(specifier_text: str) -> list[str]:
    """Split space-separated dataset specifiers into a clean list."""
    if not specifier_text:
        return []
    return [item for item in specifier_text.split() if item]


@torch.no_grad()
def inference_worker(
    rank: int,
    num_worker: int,
    world_size: int,
    train_config_path: Path,
    inference_config_path: Path,
    model_checkpoint_path: Path,
    unregistered_specifier: str,
    registered_specifier: str,
    output_dir: Path,
    seed: int,
    add_generation_prompt: bool,
    continuation_prefix_ratio: Optional[float] = None,
):
    """Worker process for inference with data sharding."""
    # Set up logger for this worker
    logger = setup_worker_logger(rank)

    device_id = rank // num_worker
    torch.cuda.set_device(f"cuda:{device_id}")
    logger.info(
        f"Starting inference worker (rank {rank}/{world_size}) with cuda = cuda:{device_id}"
    )

    # Reset random seed for each sample for independent reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load configs in worker
    with open(train_config_path, "r") as f:
        train_config = yaml.safe_load(f)

    with open(inference_config_path, "r") as f:
        inference_config = yaml.safe_load(f)

    if add_generation_prompt is not None:
        effective_add_gen_prompt = add_generation_prompt
    elif "add_generation_prompt" in inference_config:
        effective_add_gen_prompt = inference_config["add_generation_prompt"]
    else:
        effective_add_gen_prompt = True
    if effective_add_gen_prompt:
        logger.warning(
            "We should try native inference.py with `add_generation_prompt=True`"
        )

    inference_config["add_generation_prompt"] = effective_add_gen_prompt
    train_config["add_generation_prompt"] = effective_add_gen_prompt

    # Propagate continuation_prefix_ratio from inference config to train config
    if continuation_prefix_ratio is not None:
        inference_config["continuation_prefix_ratio"] = continuation_prefix_ratio
    if "continuation_prefix_ratio" in inference_config:
        train_config["continuation_prefix_ratio"] = inference_config[
            "continuation_prefix_ratio"
        ]

    job_template_class = _all_job_types[train_config["job_type"] + "_with_prefix"]
    job_template = job_template_class(train_config, is_train=False)

    # Build model and preprocessor in worker
    model = job_template.build_model()
    model = load_checkpoint(model, model_checkpoint_path)
    model.prepare_inference()
    dtype = inference_config.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype)
    model = model.to(device="cuda", dtype=dtype).eval()
    preprocessor = job_template.build_preprocessor()

    use_registered_specifier = bool(registered_specifier.strip())
    specifiers = split_specifiers(
        registered_specifier if use_registered_specifier else unregistered_specifier
    )

    if not specifiers:
        logger.warning("No valid dataset specifiers found. Worker exits.")
        return

    for dataset_specifier in specifiers:
        logger.info(f"Starting inference on dataset: {dataset_specifier}")

        # Build data iterator with sharding
        iterator_factory = DataIteratorFactory(
            unregistered_specifier=(
                "" if use_registered_specifier else dataset_specifier
            ),
            registered_specifier=(
                dataset_specifier if use_registered_specifier else ""
            ),
            collate_fn=preprocessor.collate_fn,
            num_workers=0,
            rank=rank,
            world_size=world_size,
            sequential_load=True,
        )

        dataset_output_dir = output_dir / dataset_specifier.replace(":", "_")
        dataset_output_dir.mkdir(exist_ok=True, parents=True)
        output_file = dataset_output_dir / f"results_{rank}.jsonl"

        test_iterator = iterator_factory.build_iter()

        for sample_idx, sample in enumerate(test_iterator):
            task, data_name, example_id = sample.pop("keys")[0]
            logger.info(
                f"Processing sample {sample_idx}: {task}/{data_name}/{example_id}"
            )

            sample = to_device(sample, "cuda", dtype=dtype)
            result_entry = dict(example_id=example_id)

            try:
                messages, _ = model.inference(inference_config, **sample)
            except Exception as e:
                logger.exception(e)
                if isinstance(sample.get("seqs"), torch.Tensor):
                    logger.error(
                        f"Failed to process sample {task, data_name, example_id}, sample={sample['seqs'].shape}"
                    )
                else:
                    logger.error(
                        f"Failed to process sample {task, data_name, example_id}"
                    )
                del sample
                continue

            write_messages = []
            for seg_idx, (role, modality, content) in enumerate(messages):
                if modality == "audio":
                    audio, length, sample_rate = content
                    audio, length = audio[0], length[0]
                    audio = audio.cpu().float().numpy()

                    if role == "prefix":
                        continue
                        # skip prefix waveform in output, only use this for debug
                        content = dataset_output_dir / f"{example_id}_prefix.wav"
                    else:
                        content = (
                            dataset_output_dir / f"{example_id}_segment{seg_idx+1}.wav"
                        )

                    sf.write(content, audio.T, sample_rate)
                    write_messages.append((role, modality, str(content)))
                else:
                    write_messages.append((role, modality, content))

                logger.info(
                    f"Segment {seg_idx}, role={role}, modality={modality}, content= {content}"
                )
            result_entry["messages"] = write_messages

            with open(output_file, "a", encoding="utf-8") as writer:
                writer.write(json.dumps(result_entry, ensure_ascii=False) + "\n")


def main():
    parser = get_parser()
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires GPU.")
        sys.exit(1)

    if not args.test_registered_specifier and not args.test_unregistered_specifier:
        parser.error(
            "Provide either --test-registered-specifier or "
            "--test-unregistered-specifier"
        )
    if args.test_registered_specifier and args.test_unregistered_specifier:
        parser.error(
            "Provide only one of --test-registered-specifier or "
            "--test-unregistered-specifier"
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    mp.set_start_method("spawn", force=True)

    processes = []
    args.rank -= 1  # Rank provided from 1 rather than 0
    if args.num_workers == 1:
        # Single worker, run inference directly without multiprocessing
        inference_worker(
            rank=args.rank,
            num_worker=1,
            world_size=args.world_size,
            train_config_path=args.train_config,
            inference_config_path=args.inference_config,
            model_checkpoint_path=args.model_checkpoint,
            unregistered_specifier=args.test_unregistered_specifier or "",
            registered_specifier=args.test_registered_specifier or "",
            output_dir=output_dir,
            seed=args.seed,
            add_generation_prompt=args.add_generation_prompt,
        )
        print("Worker=1 Inference completed!")
        return

    start_rank = args.rank * args.num_workers
    end_rank = (args.rank + 1) * args.num_workers
    for rank in range(start_rank, end_rank):
        p = mp.Process(
            target=inference_worker,
            args=(
                rank,
                args.num_workers,
                args.world_size * args.num_workers,
                args.train_config,
                args.inference_config,
                args.model_checkpoint,
                args.test_unregistered_specifier or "",
                args.test_registered_specifier or "",
                output_dir,
                args.seed,
                args.add_generation_prompt,
            ),
        )
        p.start()
        processes.append(p)

        time.sleep(60)  # Stagger process startups

    # Wait for all workers
    for p in processes:
        p.join()

    print("All workers completed!")


if __name__ == "__main__":
    main()
