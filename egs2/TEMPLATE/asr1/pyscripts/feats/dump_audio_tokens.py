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

from espnet2.speechlm.tokenizer.audio_tokenizer import AudioTokenizer
from espnet2.utils.types import str2bool
from espnet.nets.pytorch_backend.nets_utils import pad_list

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_audio_tokens")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec_choice", type=str, required=True)
    parser.add_argument("--codec_fs", type=int, default=16000)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--dump_audio", type=str2bool, default=False)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--vocab_file", type=str, required=True)
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


def dump_audio_tokens(
    rspecifier: str,
    wspecifier: str,
    vocab_file: str,
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
    tokenizer = AudioTokenizer(
        codec_choice,
        codec_fs,
        device,
        dump_audio,
        checkpoint_path,
        config_path,
    )

    # (3) Tokenizer loop
    codec_writer = kaldiio.WriteHelper(wspecifier)
    audio_reader = kaldiio.ReadHelper(rspecifier)

    buffer, length_buffer, key_buffer = [], [], []
    wav_reader_len = len(open(rspecifier.split(":")[1]).readlines())
    idx = 0
    for key, data in audio_reader:
        if isinstance(data, tuple):
            sample_rate, wav = data  # raw wav
        else:
            sample_rate, wav = None, data  # wav features

        wav = torch.from_numpy(wav)
        buffer.append(wav)
        length_buffer.append(len(wav))
        key_buffer.append(key)

        if idx == wav_reader_len - 1 or len(buffer) % batch_size == 0:
            wavs = pad_list(buffer, 0.0).to(device).float()
            if wavs.dim() == 2:
                wavs = wavs.unsqueeze(-1)
            if sample_rate:
                assert sample_rate == codec_fs
            # b,t,d (d=1 for raw)
            wav_lens = torch.tensor(length_buffer, dtype=torch.int).to(device)
            with torch.no_grad():
                codes, code_lengths = tokenizer(wavs, wav_lens)
            codes += bias

            codes = codes.detach().cpu().numpy()
            code_lengths = code_lengths.detach().cpu().numpy()
            assert codes.shape[0] == len(length_buffer)
            assert code_lengths.shape[0] == len(length_buffer)
            for code, length, key in zip(codes, code_lengths, key_buffer):
                code = code[:length]
                codec_writer[key] = code

            buffer, length_buffer, key_buffer = [], [], []
        idx += 1
    # (4) dump vocabulary file
    if rank == 1:
        vocab_writer = open(vocab_file, "w")
        for code_idx in range(tokenizer.size_codebook):
            vocab_writer.write(f"{code_idx}\n")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args = vars(args)
    logger.info(args)
    dump_audio_tokens(**args)
