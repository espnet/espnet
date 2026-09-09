#!/usr/bin/env python3
# Copyright 2026 Carnegie Mellon University
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""Audit a Bagpiper smoke DCP against its released initialization checkpoint.

Checks sampled trainable/frozen tensors, real Adam moments and saved step
counters. Reads selected tensors on CPU, without constructing the full model.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument(
        "--expect-audio-input-update",
        action="store_true",
        help="Require the continuous-audio adaptor to change (audio-input training)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = torch.load(
        args.reference, map_location="cpu", weights_only=True, mmap=True
    )["module"]
    trainable = "model.layers.0.self_attn.q_proj.weight"
    frozen = "multimodal_io_dict.continuous_audio.model.audio_tower.proj1.weight"
    model_keys = (
        trainable,
        frozen,
        "adaptor.continuous_audio.weight",
        "stream_emb.weight",
    )
    expected_updates = {trainable, "stream_emb.weight"}
    if args.expect_audio_input_update:
        expected_updates.add("adaptor.continuous_audio.weight")
    reader = dcp.FileSystemReader(args.checkpoint)
    metadata = reader.read_metadata()

    def allocate(name):
        item = metadata.state_dict_metadata[name]
        return torch.empty(item.size, dtype=item.properties.dtype)

    state = {
        "global_step": 0,
        "lr_scheduler": {"last_epoch": 0, "_last_lr": [0.0, 0.0], "_step_count": 0},
        "model": {key: allocate(f"model.{key}") for key in model_keys},
        "optimizer": {
            "state": {
                trainable: {
                    key: allocate(f"optimizer.state.{trainable}.{key}")
                    for key in ("step", "exp_avg", "exp_avg_sq")
                }
            }
        },
    }
    dcp.load(state, storage_reader=reader, no_dist=True)
    optimizer = state["optimizer"]["state"][trainable]
    if not (
        state["global_step"]
        == state["lr_scheduler"]["last_epoch"]
        == optimizer["step"].item()
        == args.expected_step
    ):
        raise ValueError("Global, scheduler and optimizer steps do not match")
    for key in ("exp_avg", "exp_avg_sq"):
        value = optimizer[key]
        if (
            value.dtype != torch.float32
            or not torch.isfinite(value).all()
            or not value.count_nonzero()
        ):
            raise ValueError(f"Adam {key} must be finite, nonzero FP32")

    deltas = {}
    for key, value in state["model"].items():
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"Expected finite FP32 model storage: {key}")
        delta = (value - reference[key].float()).abs()
        changed = int(delta.count_nonzero())
        if (key == frozen and changed) or (key in expected_updates and not changed):
            raise ValueError(f"Unexpected update for {key}: {changed} changed elements")
        deltas[key] = {
            "max_abs_delta": delta.max().item(),
            "changed_elements": changed,
            "elements": value.numel(),
        }
    shards = len(list(args.checkpoint.glob("*.distcp")))
    if shards != 8:
        raise ValueError(f"Expected 8 FSDP smoke shards, found {shards}")
    result = {
        "checkpoint": str(args.checkpoint),
        "step": state["global_step"],
        "lr_scheduler": state["lr_scheduler"],
        "optimizer_step": optimizer["step"].item(),
        "optimizer_moments_finite_nonzero_fp32": True,
        "shards": shards,
        "sampled_parameter_deltas": deltas,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
