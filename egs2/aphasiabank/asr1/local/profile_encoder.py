#!/usr/bin/env python

# Copyright 2023 Yifan Peng (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Profile ASR encoder."""

import argparse
import torch
from typing import Optional
from pathlib import Path
from deepspeed.profiling.flops_profiler import FlopsProfiler
# deepspeed version >= 0.8.1

from espnet2.tasks.asr import ASRTask
from espnet2.torch_utils.device_funcs import to_device


def get_parser() -> argparse.Namespace:
    """Create an argument parser."""

    parser = argparse.ArgumentParser(description="Profile encoder.")
    parser.add_argument(
        "--second",
        type=int,
        default=10,
        help="Input speech length in second."
    )
    parser.add_argument(
        '--fs',
        type=int,
        default=16000,
        help="Sample rate of speech signal."
    )
    parser.add_argument(
        "--model_file",
        required=True,
        type=Path,
        help="Path to the model file."
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=Optional[Path],
        help="Path to the config file (optional)."
    )
    return parser


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for profiler.")

    model, _ = ASRTask.build_model_from_file(args.config_file, args.model_file, "cuda")
    model.eval()
    prof = FlopsProfiler(model)
    prof.start_profile()

    # Forward
    batch = {
        "speech": torch.rand(1, args.second * args.fs).float(),
        "speech_lengths": torch.tensor([args.second * args.fs], dtype=torch.long)
    }
    batch = to_device(batch, device="cuda")
    model.encode(**batch)
    prof.stop_profile()

    # Fetch stats
    out = args.model_file.parent / f"profile.{args.second}sec.log"
    prof.print_model_profile(top_modules=10, output_file=out)
    flops = prof.get_total_flops()
    macs = prof.get_total_macs()
    params = prof.get_total_params()

    with open(out, "a") as fp:
        fp.write("\n" * 2 + "-" * 30 + " Overall " + "-" * 30 + "\n")
        fp.write(f"model: {args.model_file.resolve()}\n")
        fp.write(f"second: {args.second}\n")
        fp.write(f"fs: {args.fs}\n")
        fp.write(f"flops: {flops}, macs: {macs}, params: {params}\n")

    prof.end_profile()


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
