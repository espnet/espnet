#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import os
import sys

import kaldiio
import numpy as np
import torch

from espnet2.speechlm.tokenizer.codec_tokenizer import CodecTokenizer
from espnet2.utils.types import str2bool
from espnet.nets.pytorch_backend.nets_utils import pad_list

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_codec")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec_choice", type=str, required=True)
    parser.add_argument("--codec_fs", type=int, default=16000)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--dump_audio", type=str2bool, default=False)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--vocab_file", type=str, required=True)
    parser.add_argument("--wav_wspecifier", type=str, default=None)
    parser.add_argument(
        "--bias",
        type=int,
        default=0,
        help="bias that reserve slots for other special tokens",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="checkpoint path for Espnet (and potentially other) codec model",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="config path for Espnet (and potentially other) codec model",
    )
    parser.add_argument(
        "rspecifier", type=str, help="Read specifier for feats. e.g. ark:some.ark"
    )
    parser.add_argument(
        "wspecifier", type=str, help="Write specifier for labels. e.g. ark,t:some.txt"
    )

    return parser


def dump_codec(
    rspecifier: str,
    wspecifier: str,
    vocab_file: str,
    wav_wspecifier: str,
    codec_choice: str,
    codec_fs: int,
    batch_size: int,
    bias: int,
    dump_audio: bool,
    rank: int,
    checkpoint_path: str = None,
    config_path: str = None,
):
    # (1) Device
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device_id = rank % torch.cuda.device_count()
        else:
            device_id = 0
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
        logger.warning("Codec tokenization with CPU can be very slow.")
        logger.warning("Change batch_size=1 for CPU tokenization")
        batch_size = 1

    # (2) Codec Tokenizer Implementation
    logger.info(f"build with codec_choice: {codec_choice}")
    tokenizer = CodecTokenizer(
        codec_choice,
        codec_fs,
        device,
        dump_audio,
        checkpoint_path,
        config_path,
    )

    # (3) Tokenizer loop
    codec_writer = kaldiio.WriteHelper(wspecifier)
    wav_reader = kaldiio.ReadHelper(rspecifier)
    if wav_wspecifier is not None and dump_audio:
        wav_ark_file, wav_scp_file = wav_wspecifier.split(":")[1].split(",")
        wav_scp_writer = open(wav_scp_file, "w")
        wav_ark_writer = open(wav_ark_file, "wb")
    else:
        wav_scp_writer, wav_ark_writer = None, None

    buffer, length_buffer, key_buffer = [], [], []
    wav_reader_len = len(open(rspecifier.split(":")[1]).readlines())
    for idx, (key, (sample_rate, wav)) in enumerate(wav_reader):
        if sample_rate != tokenizer.sample_rate:
            raise ValueError("Sample rate mismatch between input audio and codec model")

        if wav.ndim != 1:
            raise ValueError("Multi-Channel audio is not supported so far")

        wav = torch.from_numpy(wav)
        buffer.append(wav)
        length_buffer.append(len(wav))
        key_buffer.append(key)

        if idx == wav_reader_len - 1 or len(buffer) % batch_size == 0:
            wavs = pad_list(buffer, 0.0).to(device).unsqueeze(1).float()
            with torch.no_grad():
                codes, resyn_wavs = tokenizer(wavs)
            codes += bias

            codes = codes.detach().cpu().numpy()
            for code, length, key in zip(codes, length_buffer, key_buffer):
                code = code[: length // tokenizer.subsample * tokenizer.n_codebook]
                codec_writer[key] = code

            if dump_audio:
                resyn_wavs = resyn_wavs.detach().cpu().numpy()
                for wav, length, key in zip(resyn_wavs, length_buffer, key_buffer):
                    wav = wav[:length]
                    kaldiio.save_ark(
                        wav_ark_writer,
                        {key: (wav, sample_rate)},
                        scp=wav_scp_writer,
                        append=True,
                        write_function="soundfile",
                        write_kwargs={"format": "wav", "subtype": None},
                    )

            buffer, length_buffer, key_buffer = [], [], []

    # (4) dump vocabulary file
    if rank == 1:
        vocab_writer = open(vocab_file, "w")
        for codebook_idx in range(tokenizer.n_codebook):
            for code_idx in range(tokenizer.size_codebook):
                vocab_writer.write(f"<codec_layer{codebook_idx}_code{code_idx}>\n")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args = vars(args)
    logger.info(args)
    dump_codec(**args)
