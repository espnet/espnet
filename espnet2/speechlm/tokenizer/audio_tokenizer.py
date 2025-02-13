#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from inspect import signature

import numpy as np
import torch
import yaml
import logging

from espnet2.speechlm.tokenizer.abs_tokenizer import AbsTokenizer
from espnet2.speechlm.tokenizer.beats_tokenizer import (
    BeatsRandomTokenizer,
    BeatsTokenizer,
)

logger = logging.getLogger(__name__)


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

        if self.codec_choice == "beats":
            beats_config = None
            if config_path:
                with open(config_path, "r") as f:
                    beats_config = yaml.safe_load(f)
            valid_args = signature(BeatsTokenizer.__init__).parameters
            remaining_args = (
                {k: v for k, v in beats_config.items() if k in valid_args}
                if beats_config
                else {}
            )
            self.codec = BeatsTokenizer(
                beats_tokenizer_ckpt_path=checkpoint_path,
                tokenizer_config=beats_config,
                **remaining_args,
            )
            self.codec = self.codec.to(device)
            self.codec.eval()
            self.n_codebook = 1
            self.size_codebook = self.codec.quantize.num_tokens
            self.sample_rate = 16000
            self.subsample = 320

        elif self.codec_choice == "beats_random":
            # Beats like patch-based frontend, with bestrq for quantization
            beats_config = None
            if config_path:
                with open(config_path, "r") as f:
                    beats_config = yaml.safe_load(f)
            valid_args = signature(BeatsRandomTokenizer.__init__).parameters
            remaining_args = (
                {k: v for k, v in beats_config.items() if k in valid_args}
                if beats_config
                else {}
            )
            self.codec = BeatsRandomTokenizer(
                tokenizer_config=beats_config,
                **remaining_args,
            )
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
            wavs (torch.Tensor): float tensor in shape [B, 1, n_sample],
            wav_lens (torch.Tensor): int tensor in shape [B]
        Output:
            codes (torch.Tensor): Int tensor in shape [B, T, n_codebook]
        """
        assert wavs.dim() == 3 and wavs.size(1) == 1
        if self.codec_choice == "beats" or self.codec_choice == "beats_random":
            wav_in = wavs.squeeze(1)  # [B, n_sample]
            if wav_in.max() > 1.0 or wav_in.min() < -1.0:
                # Beats expects input in range [-1, 1]
                wav_in = wav_in.to(torch.float32)
                wav_in = wav_in / 2**15
                logger.warning(
                    "Input waveform is not in range [-1, 1] for BEATs. Normalizing!."
                )
            # Assume no padding, all wavs are full length
            assert wav_lens is not None, "BeatsTokenizer requires wav_lens."
            codes, _, _, code_lengths = self.codec.encode(xs_pad=wav_in, ilens=wav_lens)
            codes = codes.unsqueeze(-1)
        else:
            raise NotImplementedError
        return codes, code_lengths

    def forward(self, wavs, wav_lens=None):
        """
        Convert audio waveforms into flatten codec codes and resynthesis the audio
        Input:
            wavs (torch.Tensor): float tensor in shape [B, 1, n_sample],
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
