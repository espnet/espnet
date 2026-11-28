#!/usr/bin/env python3
# Copyright 2026 Carnegie Mellon University
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""Export SpeechLM model weights from a Titan DCP directory for inference."""

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


def export_checkpoint(checkpoint_dir: Path, output: Path, dtype=None):
    """Gather only model tensors on CPU; optimizer state is not read."""
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    reader = dcp.FileSystemReader(checkpoint_dir)
    metadata = reader.read_metadata()
    state = {}
    for name, item in metadata.state_dict_metadata.items():
        if name.startswith("model.") and isinstance(item, TensorStorageMetadata):
            tensor_dtype = item.properties.dtype
            if dtype is not None and tensor_dtype.is_floating_point:
                tensor_dtype = dtype
            state[name[len("model.") :]] = torch.empty(
                item.size, dtype=tensor_dtype, device="cpu"
            )
    if not state:
        raise ValueError(f"No model tensors found in {checkpoint_dir}")
    dcp.load({"model": state}, storage_reader=reader, no_dist=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"module": state}, output)
    print(f"Exported {len(state)} model tensors to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"])
    args = parser.parse_args()
    export_checkpoint(
        args.checkpoint_dir,
        args.output,
        getattr(torch, args.dtype) if args.dtype else None,
    )


if __name__ == "__main__":
    main()
