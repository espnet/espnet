#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
from inspect import signature

import numpy as np
import torch
import yaml

from espnet2.speechlm.tokenizer.abs_tokenizer import AbsTokenizer
from espnet2.speechlm.tokenizer.beats_tokenizer import (
    BeatsRandomTokenizer,
    BeatsTokenizer,
)

logger = logging.getLogger(__name__)


def load_beats_config(config_path):
    with open(config_path, "r") as f:
        beats_config = yaml.safe_load(f)
        if "encoder_conf" in beats_config:
            beats_config = beats_config.get("encoder_conf", {})
        else:
            logging.warning(
                f"BEATs config path {config_path} was provided but "
                "no encoder_conf found."
            )
    return beats_config


class AudioTokenizer(AbsTokenizer):
    """Codec Tokenizer implementation

    Use cases:
        - use encode for discrete (de)tokenization
    """

    def __init__(
        self,
        codec_choice: str,
        codec_fs: int,
        device: str = "cpu",
        dump_audio: bool = False,
        checkpoint_path: str = None,
        config_path: str = None,
        max_token_per_frame: int = 32,
        waveform_input: bool = True,
    ):
        """Codec Tokenizer initialization

        Each of the codec implementation should assign all following features:
            self.n_codebook (int): the number of codec codebooks.
            self.size_codebook (int): the dimension of codebooks.
            self.sample_rate (int): the sample rate the model trained on.
            self.subsample (int): the subsample rate, a.k.a., frame shift.
        """

        super().__init__()
        self.codec_choice = codec_choice
        self.device = device
        self.dump_audio = dump_audio
        self.waveform_input = waveform_input

        if self.codec_choice == "beats":
            beats_config = load_beats_config(config_path) if config_path else {}
            valid_args = signature(BeatsTokenizer.__init__).parameters
            filtered_args = {k: v for k, v in beats_config.items() if k in valid_args}
            logging.info(
                f"Setting up tokenization with following args: {filtered_args}"
            )
            self.codec = BeatsTokenizer(
                beats_tokenizer_ckpt_path=checkpoint_path,
                **filtered_args,
            )
            self.codec = self.codec.to(device)
            self.codec.eval()
            self.n_codebook = 1
            self.size_codebook = self.codec.quantize.num_tokens
            self.sample_rate = 16000
            self.subsample = 320

        elif self.codec_choice == "beats_random":
            # Beats like patch-based frontend, with bestrq for quantization
            beats_config = load_beats_config(config_path) if config_path else {}
            valid_args = signature(BeatsRandomTokenizer.__init__).parameters
            filtered_args = {k: v for k, v in beats_config.items() if k in valid_args}
            logging.info(
                f"Setting up tokenization with following args: {filtered_args}"
            )
            self.codec = BeatsRandomTokenizer(**filtered_args)
            self.codec = self.codec.to(device)
            self.codec.eval()
            self.n_codebook = 1
            self.size_codebook = self.codec.config.quant_n
            self.sample_rate = 16000
            self.subsample = 320
            # NOTE(shikhar): Might be greater than 320 if audio does not fit well.
            # Example 10 second audio at 16khz returns 496 codes.
        else:
            raise ValueError(f"Codec {codec_choice} is not supported")

    def encode(self, wavs, wav_lens=None):
        """
        Convert audio waveforms into codec codes
        Input:
            wavs (torch.Tensor): float tensor in shape [B, n_sample, D],
            wav_lens (torch.Tensor): int tensor in shape [B]
        Output:
            codes (torch.Tensor): Int tensor in shape [B, T, n_codebook]
        """
        assert wavs.dim() == 3, "Input wavs should be in shape [B, n_sample, D]"
        if self.codec_choice == "beats" or self.codec_choice == "beats_random":
            wav_in = wavs.squeeze(2)  # [B, n_sample]
            if self.waveform_input and (wav_in.max() > 1.0 or wav_in.min() < -1.0):
                # Beats expects input in range [-1, 1]
                wav_in = wav_in.to(torch.float32)
                wav_in = wav_in / 2**15
                logger.warning(
                    "Input waveform not in range [-1, 1] for BEATs. Normalizing!"
                )
            # Assume no padding, all wavs are full length
            assert wav_lens is not None, "BeatsTokenizer requires wav_lens."
            ret_dict = self.codec.encode(
                xs_pad=wav_in, ilens=wav_lens, waveform_input=self.waveform_input
            )
            codes, code_lengths = ret_dict["codes"], ret_dict["code_lengths"]
            codes = codes.unsqueeze(-1)
        else:
            raise NotImplementedError
        return codes, code_lengths

    def forward(self, wavs, wav_lens=None):
        """
        Convert audio waveforms into flatten codec codes and resynthesis the audio
        Input:
            wavs (torch.Tensor): float tensor in shape [B, n_sample, D],
            wav_lens (torch.Tensor): int tensor in shape [B]
        Output:
            codes (torch.Tensor): Int tensor in shape [B, T * n_codebook],
            code_lengths (torch.Tensor): Int tensor in shape [B]
        """
        codes, code_lengths = self.encode(wavs, wav_lens)
        shift = torch.arange(self.n_codebook).to(self.device)
        codes += shift.view(1, 1, -1) * self.size_codebook
        codes = codes.int().flatten(start_dim=1)
        return codes, code_lengths
