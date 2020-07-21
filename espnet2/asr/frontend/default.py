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

# from espnet.nets.pytorch_backend.frontends.frontend import Frontend
from espnet2.enh.nets.beamformer_net import BeamformerNet as Frontend
from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft
from espnet2.utils.get_default_kwargs import get_default_kwargs


class DefaultFrontend(AbsFrontend):
    """Conventional frontend structure for ASR.

    Stft -> WPE -> MVDR-Beamformer -> Power-spec -> Mel-Fbank -> CMVN
    """

    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: int = None,
        fmax: int = None,
        htk: bool = False,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
    ):
        assert check_argument_types()
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        # Deepcopy (In general, dict shouldn't be used as default arg)
        frontend_conf = copy.deepcopy(frontend_conf)

        if frontend_conf is not None:
            frontend_conf["n_fft"] = n_fft
            frontend_conf["win_length"] = win_length
            frontend_conf["hop_length"] = hop_length
            frontend_conf["center"] = center
            frontend_conf["window"] = window
            frontend_conf["pad_mode"] = pad_mode
            frontend_conf["normalized"] = normalized
            frontend_conf["onesided"] = onesided
            self.frontend = Frontend(**frontend_conf)
            self.stft = self.frontend.stft
        else:
            self.frontend = None
            self.stft = Stft(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=center,
                window=window,
                pad_mode=pad_mode,
                normalized=normalized,
                onesided=onesided,
            )

        self.logmel = LogMel(
            fs=fs, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk,
        )
        self.n_mels = n_mels

    def output_size(self) -> int:
        return self.n_mels

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. [Optional] Speech enhancement
        if self.frontend is not None:
            # input_stft: (Batch, Length, [Channel], Freq)
            input_stft, feats_lens, mask = self.frontend(input, input_lengths)
        else:
            # 1. Domain-conversion: e.g. Stft: time -> time-freq
            input_stft, feats_lens = self.stft(input, input_lengths)

            assert input_stft.dim() >= 4, input_stft.shape
            # "2" refers to the real/imag parts of Complex
            assert input_stft.shape[-1] == 2, input_stft.shape

        # 2. Change torch.Tensor to ComplexTensor
        # input_stft: (..., F, 2) -> (..., F)
        input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])

        # 3. [Multi channel case]: Select a channel
        if input_stft.dim() == 4:
            # h: (B, T, C, F) -> h: (B, T, F)
            if self.training:
                # Select 1ch randomly
                ch = np.random.randint(input_stft.size(2))
                input_stft = input_stft[:, :, ch, :]
            else:
                # Use the first channel
                input_stft = input_stft[:, :, 0, :]

        # 4. STFT -> Power spectrum
        # h: ComplexTensor(B, T, F) -> torch.Tensor(B, T, F)
        input_power = input_stft.real ** 2 + input_stft.imag ** 2

        # 5. Feature transform e.g. Stft -> Log-Mel-Fbank
        # input_power: (Batch, [Channel,] Length, Freq)
        #       -> input_feats: (Batch, Length, Dim)
        input_feats, _ = self.logmel(input_power, feats_lens)

        return input_feats, feats_lens
