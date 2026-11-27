#!/usr/bin/env python3
"""Resynthesize waveforms from the hard token IDs written by tokenizer stage 10.

The input is the saved discrete token sequence, not audio or the straight-through
one-hot assignments used during joint training. This script measures the waveform
available to a user of the exported hard tokens through the jointly trained
reconstruction branch.
"""

import argparse
import logging
import sys
from pathlib import Path

import humanfriendly
import soundfile
import torch
import yaml

from espnet2.legacy.utils.cli_utils import get_commandline_args
from espnet2.tasks.gan_tok import GANSpeechTokenizerTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.safe_torch_load import safe_torch_load
from espnet2.utils import config_argparse

TASK_CLASSES = {"gan_tok": GANSpeechTokenizerTask}


def validate_reconstruction_branch_checkpoint(model_file: str) -> None:
    """Require trained parameters for the reconstruction branch."""
    checkpoint = safe_torch_load(model_file, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if isinstance(checkpoint, dict) and "module" in checkpoint:
        checkpoint = checkpoint["module"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(checkpoint)}")
    if not any(
        name.removeprefix("module.").startswith("losses.reconstruction.")
        for name in checkpoint
    ):
        raise RuntimeError(
            "Checkpoint does not contain parameters for the reconstruction branch"
        )


def build_iterator(
    task_class,
    token_file: str,
    spembs_scp: str,
    key_file: str | None,
    dtype: str,
    ngpu: int,
    num_workers: int,
):
    """Build an ESPnet iterator over saved hard token IDs and embeddings."""
    return task_class.build_streaming_iterator(
        data_path_and_name_and_type=[
            (token_file, "speech", "text_int"),
            (spembs_scp, "spembs", "kaldi_ark"),
        ],
        preprocess_fn=None,
        collate_fn=task_class.build_collate_fn(argparse.Namespace(), train=False),
        key_file=key_file,
        batch_size=1,
        dtype=dtype,
        num_workers=num_workers,
        allow_variable_data_keys=True,
        ngpu=ngpu,
        inference=True,
    )


def inference_reconstruction(
    model,
    iterator,
    output: Path,
    device: str,
    train_config: str,
    sample_rate: int | str | None,
) -> None:
    """Resynthesize waveforms from discrete tokens."""
    if sample_rate is None:
        with Path(train_config).open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        sample_rate = config.get("frontend_conf", {}).get("fs")
        if sample_rate is None:
            raise ValueError(
                "sample_rate must be provided when frontend_conf.fs is absent "
                "from the training config"
            )
    if isinstance(sample_rate, str):
        sample_rate = humanfriendly.parse_size(sample_rate)
    wav_dir = output / "wav"
    wav_dir.mkdir(exist_ok=True)
    with (wav_dir / "wav.scp").open("w", encoding="utf-8") as wav_scp:
        for keys, batch in iterator:
            if Path(keys[0]).name != keys[0] or keys[0] in ("", ".", ".."):
                raise ValueError(f"Invalid utterance key for WAV output: {keys[0]}")
            batch = to_device(batch, device)
            ids = batch["speech"]
            lengths = batch["speech_lengths"]
            speaker = batch["spembs"]
            waveform, waveform_lengths = model.synthesize_from_tokens(
                ids, lengths, speaker
            )
            wav = waveform[0, 0, : int(waveform_lengths[0])].float().cpu().numpy()
            wav_path = (wav_dir / f"{keys[0]}.wav").resolve()
            soundfile.write(wav_path, wav, sample_rate)
            wav_scp.write(f"{keys[0]} {wav_path}\n")


@torch.inference_mode()
def inference(
    task: str,
    token_file: str,
    spembs_scp: str | None,
    output_dir: str,
    train_config: str,
    model_file: str,
    key_file: str | None,
    ngpu: int,
    dtype: str,
    num_workers: int,
    sample_rate: str | None,
) -> None:
    """Run the trained reconstruction branch from saved hard token IDs."""
    if ngpu not in (0, 1):
        raise ValueError("ngpu must be 0 or 1")
    if spembs_scp is None:
        raise ValueError("spembs_scp is required for reconstruction inference")
    device = "cuda" if ngpu else "cpu"
    task_class = TASK_CLASSES[task]
    validate_reconstruction_branch_checkpoint(model_file)
    model, _ = task_class.build_model_from_file(train_config, model_file, device)
    model.to(dtype=getattr(torch, dtype)).eval()
    if not hasattr(model, "synthesize_from_tokens"):
        raise ValueError(
            f"Task {task} does not support reconstruction inference from tokens"
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    iterator = build_iterator(
        task_class, token_file, spembs_scp, key_file, dtype, ngpu, num_workers
    )
    inference_reconstruction(model, iterator, output, device, train_config, sample_rate)


def get_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = config_argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", choices=tuple(TASK_CLASSES), default="gan_tok")
    parser.add_argument("--token_file", required=True)
    parser.add_argument("--spembs_scp")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--key_file")
    parser.add_argument("--ngpu", type=int, default=0)
    parser.add_argument(
        "--dtype", choices=("float16", "float32", "float64"), default="float32"
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--sample_rate")
    return parser


def main(cmd=None) -> None:
    """Resynthesize waveforms from saved hard token IDs."""
    print(get_commandline_args(), file=sys.stderr)
    logging.basicConfig(level=logging.INFO)
    args = get_parser().parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
