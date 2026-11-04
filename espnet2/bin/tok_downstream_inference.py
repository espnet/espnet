#!/usr/bin/env python3
"""Run tokenizer downstream branches from saved discrete speech tokens."""

import argparse
import logging
import sys
from pathlib import Path

import humanfriendly
import soundfile
import torch
import yaml

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


def validate_downstream_branch_checkpoint(model_file: str, mode: str) -> None:
    """Require trained parameters for the requested downstream branch."""
    checkpoint = safe_torch_load(model_file, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if isinstance(checkpoint, dict) and "module" in checkpoint:
        checkpoint = checkpoint["module"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(checkpoint)}")

    prefix = f"losses.{mode}."
    if not any(name.removeprefix("module.").startswith(prefix) for name in checkpoint):
        raise RuntimeError(
            f"Checkpoint does not contain parameters for the {mode} downstream branch"
        )


def build_iterator(
    task_class,
    token_file: str,
    spembs_scp: str | None,
    key_file: str | None,
    dtype: str,
    ngpu: int,
    num_workers: int,
):
    """Build an ESPnet iterator over saved tokens and optional embeddings."""
    data = [(token_file, "speech", "text_int")]
    if spembs_scp is not None:
        data.append((spembs_scp, "spembs", "kaldi_ark"))
    return task_class.build_streaming_iterator(
        data_path_and_name_and_type=data,
        preprocess_fn=None,
        collate_fn=task_class.build_collate_fn(argparse.Namespace(), train=False),
        key_file=key_file,
        batch_size=1,
        dtype=dtype,
        num_workers=num_workers,
        allow_variable_data_keys=spembs_scp is not None,
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
    """Run the ASR downstream branch from discrete tokens."""
    asr_model = model.asr_loss.asr_model
    ctc_weight = asr_model.ctc_weight
    beam_search = build_beam_search(
        asr_model, beam_size=beam_size, ctc_weight=ctc_weight, penalty=penalty
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


def inference_reconstruction(
    model,
    iterator,
    output: Path,
    device: str,
    train_config: str,
    sample_rate: int | str | None,
) -> None:
    """Run the reconstruction downstream branch from discrete tokens."""
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
    mode: str,
    token_file: str,
    output_dir: str,
    train_config: str,
    model_file: str,
    key_file: str | None,
    spembs_scp: str | None,
    ngpu: int,
    dtype: str,
    beam_size: int,
    lm_weight: float,
    penalty: float,
    maxlenratio: float,
    minlenratio: float,
    num_workers: int,
    sample_rate: str | None,
) -> None:
    """Run one downstream inference mode from discrete tokens."""
    if ngpu not in (0, 1):
        raise ValueError("ngpu must be 0 or 1")
    if lm_weight != 0.0:
        raise ValueError("lm_weight must be 0 because no external LM is loaded")
    device = "cuda" if ngpu else "cpu"
    task_class = TASK_CLASSES[task]
    validate_downstream_branch_checkpoint(model_file, mode)
    model, args = task_class.build_model_from_file(train_config, model_file, device)
    model.to(dtype=getattr(torch, dtype)).eval()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if mode == "asr":
        if not hasattr(model, "encode_asr_from_tokens"):
            raise ValueError(f"Task {task} does not support ASR inference from tokens")
        iterator = build_iterator(
            task_class, token_file, None, key_file, dtype, ngpu, num_workers
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
    elif mode == "reconstruction":
        if not hasattr(model, "synthesize_from_tokens"):
            raise ValueError(
                f"Task {task} does not support reconstruction inference from tokens"
            )
        if spembs_scp is None:
            raise ValueError("spembs_scp is required for reconstruction")
        iterator = build_iterator(
            task_class,
            token_file,
            spembs_scp,
            key_file,
            dtype,
            ngpu,
            num_workers,
        )
        inference_reconstruction(
            model,
            iterator,
            output,
            device,
            train_config,
            sample_rate,
        )
    else:
        raise ValueError(f"Unsupported downstream inference mode: {mode}")


def get_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = config_argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", choices=tuple(TASK_CLASSES), default="gan_tok")
    parser.add_argument("--mode", choices=("asr", "reconstruction"), required=True)
    parser.add_argument("--token_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--key_file")
    parser.add_argument("--spembs_scp")
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
    parser.add_argument("--sample_rate")
    return parser


def main(cmd=None) -> None:
    """Run downstream branch inference."""
    print(get_commandline_args(), file=sys.stderr)
    logging.basicConfig(level=logging.INFO)
    args = get_parser().parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
