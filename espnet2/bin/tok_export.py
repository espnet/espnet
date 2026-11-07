#!/usr/bin/env python3
"""Export tokenizer parameters from a joint speech-tokenizer checkpoint."""

import argparse
from pathlib import Path

import torch


def export_tokenizer_checkpoint(model_file: str, output_file: str) -> None:
    """Save only tokenizer parameters using normalized ``tokenizer.*`` keys."""
    checkpoint = torch.load(model_file, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(checkpoint)}")

    tokenizer_state = {}
    for name, value in checkpoint.items():
        if name.startswith("module.tokenizer."):
            name = name[len("module.") :]
        if name.startswith("tokenizer."):
            tokenizer_state[name] = value
    if not tokenizer_state:
        raise RuntimeError(f"No tokenizer.* parameters found in {model_file}")

    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": tokenizer_state}, output)


def get_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--output_file", required=True)
    return parser


def main(cmd=None) -> None:
    """Export a tokenizer-only checkpoint."""
    args = get_parser().parse_args(cmd)
    export_tokenizer_checkpoint(args.model_file, args.output_file)


if __name__ == "__main__":
    main()
