#!/usr/bin/env python3
"""Decode ASR hypotheses from the hard token IDs written by tokenizer stage 10.

The input is the saved discrete token sequence, not audio or the straight-through
one-hot assignments used during joint training. This script measures the ASR
output available to a user of the exported hard tokens.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

from espnet2.legacy.nets.beam_search import BeamSearch
from espnet2.legacy.nets.scorers.ctc import CTCPrefixScorer
from espnet2.legacy.nets.scorers.length_bonus import LengthBonus
from espnet2.legacy.utils.cli_utils import get_commandline_args
from espnet2.tasks.gan_tok import GANSpeechTokenizerTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.safe_torch_load import safe_torch_load
from espnet2.utils import config_argparse

TASK_CLASSES = {"gan_tok": GANSpeechTokenizerTask}


def validate_asr_branch_checkpoint(model_file: str) -> None:
    """Require trained parameters for the ASR branch."""
    checkpoint = safe_torch_load(model_file, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if isinstance(checkpoint, dict) and "module" in checkpoint:
        checkpoint = checkpoint["module"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(checkpoint)}")
    if not any(
        name.removeprefix("module.").startswith("losses.asr.") for name in checkpoint
    ):
        raise RuntimeError("Checkpoint does not contain parameters for the ASR branch")


def build_iterator(
    task_class,
    token_file: str,
    key_file: str | None,
    dtype: str,
    ngpu: int,
    num_workers: int,
):
    """Build an ESPnet iterator over saved hard token IDs."""
    return task_class.build_streaming_iterator(
        data_path_and_name_and_type=[(token_file, "speech", "text_int")],
        preprocess_fn=None,
        collate_fn=task_class.build_collate_fn(argparse.Namespace(), train=False),
        key_file=key_file,
        batch_size=1,
        dtype=dtype,
        num_workers=num_workers,
        allow_variable_data_keys=False,
        ngpu=ngpu,
        inference=True,
    )


def build_beam_search(asr_model, beam_size: int, ctc_weight: float, penalty: float):
    """Build the standard joint CTC/attention beam search for the ASR branch."""
    token_list = asr_model.token_list
    scorers = {
        "decoder": asr_model.decoder,
        "ctc": CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos),
        "length_bonus": LengthBonus(len(token_list)),
    }
    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "length_bonus": penalty,
    }
    return BeamSearch(
        beam_size=beam_size,
        weights=weights,
        scorers=scorers,
        sos=asr_model.sos,
        eos=asr_model.eos,
        vocab_size=len(token_list),
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "full",
    )


def inference_asr(
    model,
    args,
    iterator,
    output: Path,
    device: str,
    dtype: str,
    beam_size: int,
    penalty: float,
    maxlenratio: float,
    minlenratio: float,
) -> None:
    """Decode ASR hypotheses from discrete tokens."""
    asr_model = model.asr_loss.asr_model
    ctc_weight = asr_model.ctc_weight
    beam_search = build_beam_search(
        asr_model,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        penalty=penalty,
    )
    beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
    text_tokenizer = build_tokenizer(token_type=args.token_type, bpemodel=args.bpemodel)
    with (
        (output / "text").open("w", encoding="utf-8") as text_stream,
        (output / "token").open("w", encoding="utf-8") as token_stream,
        (output / "token_int").open("w", encoding="utf-8") as id_stream,
    ):
        for keys, batch in iterator:
            batch = to_device(batch, device)
            ids = batch["speech"]
            lengths = batch["speech_lengths"]
            encoded, encoded_lengths = model.encode_asr_from_tokens(ids, lengths)
            hypotheses = beam_search(
                x=encoded[0, : int(encoded_lengths[0])],
                maxlenratio=maxlenratio,
                minlenratio=minlenratio,
            )
            hypothesis = hypotheses[0].yseq[1:-1]
            if isinstance(hypothesis, torch.Tensor):
                hypothesis = hypothesis.tolist()
            hypothesis = [index for index in hypothesis if index != 0]
            tokens = [asr_model.token_list[index] for index in hypothesis]
            text = text_tokenizer.tokens2text(tokens) if text_tokenizer else ""
            text_stream.write(f"{keys[0]} {text}\n")
            token_stream.write(f"{keys[0]} {' '.join(tokens)}\n")
            id_stream.write(f"{keys[0]} {' '.join(map(str, hypothesis))}\n")


@torch.inference_mode()
def inference(
    task: str,
    token_file: str,
    output_dir: str,
    train_config: str,
    model_file: str,
    key_file: str | None,
    ngpu: int,
    dtype: str,
    beam_size: int,
    lm_weight: float,
    penalty: float,
    maxlenratio: float,
    minlenratio: float,
    num_workers: int,
) -> None:
    """Run the trained ASR branch from saved hard token IDs."""
    if ngpu not in (0, 1):
        raise ValueError("ngpu must be 0 or 1")
    if lm_weight != 0.0:
        raise ValueError("lm_weight must be 0 because no external LM is loaded")
    device = "cuda" if ngpu else "cpu"
    task_class = TASK_CLASSES[task]
    validate_asr_branch_checkpoint(model_file)
    model, args = task_class.build_model_from_file(train_config, model_file, device)
    model.to(dtype=getattr(torch, dtype)).eval()
    if not hasattr(model, "encode_asr_from_tokens"):
        raise ValueError(f"Task {task} does not support ASR inference from tokens")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    iterator = build_iterator(
        task_class, token_file, key_file, dtype, ngpu, num_workers
    )
    inference_asr(
        model,
        args,
        iterator,
        output,
        device,
        dtype,
        beam_size,
        penalty,
        maxlenratio,
        minlenratio,
    )


def get_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = config_argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", choices=tuple(TASK_CLASSES), default="gan_tok")
    parser.add_argument("--token_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--key_file")
    parser.add_argument("--ngpu", type=int, default=0)
    parser.add_argument(
        "--dtype", choices=("float16", "float32", "float64"), default="float32"
    )
    parser.add_argument("--beam_size", type=int, default=20)
    parser.add_argument("--lm_weight", type=float, default=0.0)
    parser.add_argument("--penalty", type=float, default=0.0)
    parser.add_argument("--maxlenratio", type=float, default=0.0)
    parser.add_argument("--minlenratio", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=1)
    return parser


def main(cmd=None) -> None:
    """Decode ASR hypotheses from saved hard token IDs."""
    print(get_commandline_args(), file=sys.stderr)
    logging.basicConfig(level=logging.INFO)
    args = get_parser().parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
