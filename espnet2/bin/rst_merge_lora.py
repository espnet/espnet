#!/usr/bin/env python3
"""Merge the LoRA adapter of a restoration feature predictor into its weights.

Training keeps the adapter separate (a frozen base plus low-rank updates on the
student's adapted layers), which costs two extra matmuls per adapted layer at
every forward pass. After training the adapter never changes again, so this
folds it into the base weights once and writes a predictor that computes the
same features without it:

    <output_dir>/config.yaml   the training config with lora_rank: 0
    <output_dir>/model.pth     the merged weights

rst_inference (--train_config/--model_file) and vocoder finetuning
(--fp_model_path with --lora_rank 0) load the pair like any other predictor.
Before writing, the merged student is run on a random waveform and compared
with the adapted one; the merge is refused if they disagree.
"""

import argparse
import logging
from pathlib import Path

import torch
import yaml

from espnet2.bin.rst_inference import _load_feature_predictor
from espnet2.rst.rst_model import merge_lora_adapters

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Merge the LoRA adapter of a restoration feature predictor"
    )
    parser.add_argument(
        "--train_config", required=True, help="config.yaml of the predictor"
    )
    parser.add_argument(
        "--model_file", required=True, help="its checkpoint, e.g. valid.loss.best.pth"
    )
    parser.add_argument(
        "--output_dir", required=True, help="where config.yaml and model.pth go"
    )
    parser.add_argument(
        "--check_duration",
        type=float,
        default=2.0,
        help="seconds of random audio on which the merged predictor must "
        "reproduce the adapted one; 0 skips the check",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="largest absolute feature difference accepted by the check",
    )
    return parser


@torch.no_grad()
def _student_features(encoder, waveform):
    lengths = torch.tensor([waveform.size(1)])
    inputs = encoder._wav_to_ssl_inputs(waveform, lengths)
    return encoder.encode(inputs)[0]


def main(cmd=None):
    args = get_parser().parse_args(cmd)
    logging.basicConfig(level=logging.INFO)

    model = _load_feature_predictor(args.train_config, args.model_file, "cpu")
    encoder = model.ssl_encoder
    waveform = None
    if args.check_duration > 0:
        generator = torch.Generator().manual_seed(0)
        samples = int(args.check_duration * encoder.input_sr)
        waveform = 0.1 * torch.randn(1, samples, generator=generator)
        adapted = _student_features(encoder, waveform)

    merged = merge_lora_adapters(encoder.student)
    if merged == 0:
        raise RuntimeError(f"{args.model_file} has no LoRA adapter to merge")
    logger.info("merged the LoRA adapter of %d layers", merged)

    if waveform is not None:
        difference = (_student_features(encoder, waveform) - adapted).abs().max()
        if difference > args.atol:
            raise RuntimeError(
                f"the merged predictor differs from the adapted one by "
                f"{difference:.3g} > --atol {args.atol}"
            )
        logger.info("merged features match (max abs difference %.3g)", difference)

    with open(args.train_config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    config["lora_rank"] = 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pth")
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    logger.info("wrote %s", output_dir)


if __name__ == "__main__":
    main()
