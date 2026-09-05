#!/usr/bin/env python3
"""Extract discrete speech tokens without constructing downstream objectives."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import yaml

from espnet2.legacy.utils.cli_utils import get_commandline_args
from espnet2.tasks.gan_tok import (
    GANSpeechTokenizerTask,
    frontend_choices,
    quantizer_choices,
)
from espnet2.tok.tokenizer import SpeechTokenizer
from espnet2.torch_utils.device_funcs import to_device
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str


def build_tokenizer(
    train_config: str,
    model_file: str,
    device: str,
    dtype: str,
) -> SpeechTokenizer:
    """Build and restore only the shared tokenizer from a training checkpoint."""
    with Path(train_config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    frontend = frontend_choices.get_class(config["frontend"])(
        **config.get("frontend_conf", {})
    )
    quantizer = quantizer_choices.get_class(config["quantizer"])(
        **config.get("quantizer_conf", {})
    )
    tokenizer = SpeechTokenizer(
        frontend=frontend,
        quantizer=quantizer,
        **config.get("tokenizer_conf", {}),
    )

    checkpoint = torch.load(model_file, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(checkpoint)}")

    tokenizer_state = {}
    for name, value in checkpoint.items():
        if name.startswith("module.tokenizer."):
            tokenizer_state[name[len("module.tokenizer.") :]] = value
        elif name.startswith("tokenizer."):
            tokenizer_state[name[len("tokenizer.") :]] = value
    if not tokenizer_state:
        raise RuntimeError(f"No tokenizer.* parameters found in {model_file}")
    tokenizer.load_state_dict(tokenizer_state, strict=True)
    return tokenizer.to(device=device, dtype=getattr(torch, dtype)).eval()


@torch.inference_mode()
def inference(
    output_dir: str,
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    train_config: str,
    model_file: str,
    ngpu: int,
    dtype: str,
    batch_size: int,
    num_workers: int,
    allow_variable_data_keys: bool,
    log_level: str,
) -> None:
    """Write deterministic nearest-centroid token IDs for each utterance."""
    if ngpu > 1:
        raise NotImplementedError("Only single-GPU inference is supported")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    device = "cuda" if ngpu == 1 else "cpu"
    tokenizer = build_tokenizer(train_config, model_file, device, dtype)

    iterator = GANSpeechTokenizerTask.build_streaming_iterator(
        data_path_and_name_and_type=data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=None,
        collate_fn=GANSpeechTokenizerTask.build_collate_fn(
            argparse.Namespace(), train=False
        ),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (
        (output_path / "token").open("w", encoding="utf-8") as token_file,
        (output_path / "token_length").open("w", encoding="utf-8") as length_file,
    ):
        for keys, batch in iterator:
            batch = to_device(batch, device)
            output = tokenizer.encode(batch["speech"], batch["speech_lengths"])
            for index, key in enumerate(keys):
                length = int(output.lengths[index])
                ids = output.token_ids[index, :length].tolist()
                token_file.write(f"{key} {' '.join(map(str, ids))}\n")
                length_file.write(f"{key} {length}\n")


def get_parser() -> argparse.ArgumentParser:
    """Build the tokenizer-only inference parser."""
    parser = config_argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--ngpu", type=int, default=0)
    parser.add_argument(
        "--dtype", choices=("float16", "float32", "float64"), default="float32"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        action="append",
        required=True,
    )
    parser.add_argument("--key_file")
    parser.add_argument("--allow_variable_data_keys", type=str2bool, default=False)
    return parser


def main(cmd=None) -> None:
    """Run tokenizer-only inference."""
    print(get_commandline_args(), file=sys.stderr)
    args = get_parser().parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
