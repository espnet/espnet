#!/usr/bin/env python3
# Copyright 2026 Carnegie Mellon University
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""Export SpeechLM model weights from a Titan DCP directory for inference."""

import argparse
import math
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from filelock import FileLock
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


def _merge_lora_weight(state, prefix, dtype):
    """Fold a saved linear adapter into its dense weight before output casting."""
    weight = state[prefix + "weight"]
    a = state.pop(prefix + "lora_A")
    b = state.pop(prefix + "lora_B")
    scale = state.pop(prefix + "lora_scale")
    if (
        weight.ndim != 2
        or a.ndim != 2
        or b.ndim != 2
        or not a.size(0)
        or b.shape != (weight.size(0), a.size(0))
        or a.size(1) != weight.size(1)
        or scale.ndim != 0
        or not all(t.is_floating_point() for t in (weight, a, b, scale))
        or not math.isfinite(scale.item())
    ):
        raise ValueError(f"Invalid LoRA tensors for {prefix}weight")
    compute_dtype = torch.promote_types(weight.dtype, torch.float32)
    merged = torch.addmm(
        weight.to(compute_dtype),
        b.to(compute_dtype),
        a.to(compute_dtype),
        alpha=scale.item(),
    )
    state[prefix + "weight"] = merged.to(dtype or weight.dtype)


def export_checkpoint(checkpoint_dir: Path, output: Path, dtype=None):
    """Gather model tensors on CPU and merge any trained SpeechLM LoRA weights.

    Args:
        checkpoint_dir: A complete native Titan DCP directory.
        output: Destination for the standalone native inference checkpoint.
        dtype: Optional output dtype for floating-point model tensors.
    """
    if os.path.lexists(output):
        raise FileExistsError(f"Output already exists: {output}")
    reader = dcp.FileSystemReader(checkpoint_dir)
    tensors = {
        name[len("model.") :]: item
        for name, item in reader.read_metadata().state_dict_metadata.items()
        if name.startswith("model.") and isinstance(item, TensorStorageMetadata)
    }
    if not tensors:
        raise ValueError(f"No model tensors found in {checkpoint_dir}")
    factors = ("lora_A", "lora_B", "lora_scale")
    layers = {
        name[: -len(field)]
        for name in tensors
        for field in factors
        if name == field or name.endswith("." + field)
    }
    operands = {prefix + field for prefix in layers for field in ("weight", *factors)}
    if missing := operands.difference(tensors):
        raise ValueError(f"Incomplete LoRA checkpoint, missing: {sorted(missing)}")
    state = {}
    for name, item in tensors.items():
        tensor_dtype = item.properties.dtype
        if (
            dtype is not None
            and tensor_dtype.is_floating_point
            and name not in operands
        ):
            tensor_dtype = dtype
        state[name] = torch.empty(item.size, dtype=tensor_dtype, device="cpu")
    dcp.load({"model": state}, storage_reader=reader, no_dist=True)
    for prefix in layers:
        _merge_lora_weight(state, prefix, dtype)

    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete file, preserving an existing export if two jobs race.
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        path = Path(staging) / output.name
        torch.save({"module": state}, path)
        with FileLock(output.with_name(f".{output.name}.lock")):
            if os.path.lexists(output):
                raise FileExistsError(f"Output already exists: {output}")
            path.replace(output)
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
