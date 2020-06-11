import copy
from typing import Optional
from typing import Tuple
from typing import Union

import humanfriendly
import numpy as np
import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet.nets.pytorch_backend.rnn.encoders import RNN
from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft
from espnet2.utils.get_default_kwargs import get_default_kwargs
import pdb



class TFMaskingSeparation(torch.nn.Module):
    """ TF Masking Speech Separation Net

    """
    def __init__(
        self,
        num_bin: int,
        r_type: str = 'lstm',
        r_layer: int = 3,
        r_hidden: int = 512,
        r_dropout: float = 0.0,
        num_spk: int = 2, ):
        super(TFMaskingSeparation, self).__init__()

        self.num_spk = num_spk
        self.rnn = RNN(
                    num_bin,
                    r_layer,
                    r_hidden,
                    num_bin,
                    r_dropout,
                    typ=r_type,)

        self.linear = torch.nn.ModuleList(
            [
                torch.nn.Linear(num_bin, num_bin)
                for _ in range(self.num_spk)
            ]
        )
        self.none_linear = torch.sigmoid

    def forward(
        self, input: torch.Tensor, ilens: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input (torch.Tensor): magnitude spectrum [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
            predcited magnitude masks [Batch, num_speaker, T, F]
            output lengths
        """
        
        x, ilens, _ = self.rnn(input, ilens)

        masks = []
        for linear in self.linear:
            y = linear(x)
            y = self.none_linear(y)
            masks.append(y)
        masks = torch.stack(masks, dim=1)
        return masks, ilens

class EnhFrontend(AbsFrontend):
    """Speech separation frontend 

    STFT -> T-F masking -> [STFT_0, ... , STFT_S]
    """

    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        center: bool = True,
        pad_mode: str = "reflect",
        normalized: bool = False,
        onesided: bool = True,
    ):
        assert check_argument_types()
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=onesided,
        )
        self.bins = n_fft // 2 + 1
        self.sep_net = TFMaskingSeparation(num_bin=self.bins)


    def output_size(self) -> int:
        return self.bins

    def wav_to_stft(
        self, 
        input: torch.Tensor, 
        input_lengths: torch.Tensor) -> ComplexTensor:
        
        input_stft, feats_lens = self.stft(input, input_lengths)
        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # Change torch.Tensor to ComplexTensor
        # input_stft: (..., F, 2) -> (..., F)
        input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])

        return input_stft, feats_lens


    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input (torch.Tensor): raw wave input [batch, samples]
            input_lengths (torch.Tensor): [batch]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            predcited magnitude spectrum [Batch, num_speaker, T, F]
        """

        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        input_stft, feats_lens = self.wav_to_stft(input, input_lengths)

        # for data-parallel
        input_stft = input_stft[:,:max(feats_lens), :]
        
        input_magnitude = abs(input_stft)

        # 2. [Option] Speech enhancement
        assert isinstance(input_magnitude, torch.Tensor), type(input_magnitude)
        # input_magnitude: (Batch, Length, Freq)
        magnitude_mask, _ = self.sep_net(input_magnitude, feats_lens)
        
        try:
            predcited_magnitude = magnitude_mask * torch.unsqueeze(input_magnitude,dim=1)
        except:
            pdb.set_trace()

        return predcited_magnitude, feats_lens
