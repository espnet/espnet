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
        default=4,
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


@torch.no_grad()
def inference_worker(
    rank: int,
    world_size: int,
    train_config_path: Path,
    inference_config_path: Path,
    model_checkpoint_path: Path,
    unregistered_specifier: str,
    registered_specifier: str,
    output_dir: Path,
    seed: int,
):
    """Worker process for inference with data sharding."""
    # Set up logger for this worker
    logger = setup_worker_logger(rank)
    logger.info(f"Starting inference worker (rank {rank}/{world_size})")

    torch.cuda.set_device("cuda:0")

    # Load configs in worker
    with open(train_config_path, "r") as f:
        train_config = yaml.safe_load(f)

    with open(inference_config_path, "r") as f:
        inference_config = yaml.safe_load(f)

    job_template_class = _all_job_types[train_config["job_type"]]
    job_template = job_template_class(train_config, is_train=False)

    # Build model and preprocessor in worker
    model = job_template.build_model()
    model = load_checkpoint(model, model_checkpoint_path)
    model.prepare_inference()
    dtype = inference_config.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype)
    model = model.to(device="cuda", dtype=dtype).eval()
    preprocessor = job_template.build_preprocessor()

    # Build data iterator with sharding
    iterator_factory = DataIteratorFactory(
        unregistered_specifier=unregistered_specifier,
        registered_specifier=registered_specifier,
        collate_fn=preprocessor.collate_fn,
        num_workers=0,
        rank=rank,
        world_size=world_size,
        sequential_load=True,
    )

    output_dir = output_dir / f"inference_rank{rank}"
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / "results.json"

    test_iterator = iterator_factory.build_iter()
    results = dict()
    logger.info("Starting inference on data shard")

    for idx, sample in enumerate(test_iterator):

        sample = to_device(sample, "cuda", dtype=dtype)
        task, data_name, example_id = sample.pop("keys")[0]

        # Reset random seed for each sample for independent reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        logger.info(f"Processing sample {idx}: {task}/{data_name}/{example_id}")
        messages, _ = model.inference(inference_config, **sample)

        for idx, (role, modality, content) in enumerate(messages):
            if modality == "audio":
                audio, length, sample_rate = content
                audio, length = audio[0], length[0]
                audio = audio.cpu().float().numpy()

                content = output_dir / f"{example_id}_segment{idx + 1}.wav"
                sf.write(content, audio.T, sample_rate)

                messages[idx][2] = str(content)

            logger.info(
                f"Segment {idx}, role={role}, modality={modality}, content={content}"
            )

        results[example_id] = messages
        with open(output_file, "wb") as writer:
            writer.write(
                json.dumps(
                    results, indent=4, ensure_ascii=False, sort_keys=False
                ).encode("utf_8")
            )


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

    specifier = args.test_registered_specifier or args.test_unregistered_specifier
    output_dir = args.output_dir / specifier.replace(":", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    mp.set_start_method("spawn", force=True)

    processes = []
    args.rank -= 1  # Rank provided from 1 rather than 0
    start_rank = args.rank * args.num_workers
    end_rank = (args.rank + 1) * args.num_workers
    for rank in range(start_rank, end_rank):
        p = mp.Process(
            target=inference_worker,
            args=(
                rank,
                args.world_size * args.num_workers,
                args.train_config,
                args.inference_config,
                args.model_checkpoint,
                args.test_unregistered_specifier or "",
                args.test_registered_specifier or "",
                output_dir,
                args.seed,
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
