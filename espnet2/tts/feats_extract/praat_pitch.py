# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""F0 extractor from Praat with Parselmouth Python wrapper."""

import logging
from typing import Any, Dict, Tuple, Union

import humanfriendly
import numpy as np
import parselmouth
import torch
import torch.nn.functional as F
from scipy.interpolate import interp1d
from typeguard import check_argument_types

from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet.nets.pytorch_backend.nets_utils import pad_list


class PraatPitch(AbsFeatsExtract):
    """F0 estimation with Praat algorithm.

    https://www.fon.hum.uva.nl/praat/manual/Sound__To_Pitch__ac____.html

    Note:
        This module is based on NumPy implementation. Therefore, the computational graph
        is not connected.

    Todo:
        Replace this module with PyTorch-based implementation.

    """

    def __init__(
        self,
        fs: Union[int, str] = 22050,
        hop_length: int = 256,
        f0min: int = 75,
        f0max: int = 800,
        use_token_averaged_f0: bool = True,
        use_continuous_f0: bool = True,
        use_log_f0: bool = True,
        reduction_factor: int = None,
        enable_warnings: bool = True
    ):
        assert check_argument_types()
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        self.fs = fs
        self.hop_length = hop_length
        self.f0min = f0min
        self.f0max = f0max
        self.use_token_averaged_f0 = use_token_averaged_f0
        self.use_continuous_f0 = use_continuous_f0
        self.use_log_f0 = use_log_f0
        if use_token_averaged_f0:
            assert reduction_factor >= 1
        self.reduction_factor = reduction_factor
        self.enable_warnings = enable_warnings
        self.padding = 2 * hop_length

    def output_size(self) -> int:
        return 1

    def get_parameters(self) -> Dict[str, Any]:
        return dict(
            fs=self.fs,
            hop_length=self.hop_length,
            f0min=self.f0min,
            f0max=self.f0max,
            use_token_averaged_f0=self.use_token_averaged_f0,
            use_continuous_f0=self.use_continuous_f0,
            use_log_f0=self.use_log_f0,
            reduction_factor=self.reduction_factor,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor = None,
        feats_lengths: torch.Tensor = None,
        durations: torch.Tensor = None,
        durations_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # If not provided, we assume that the inputs have the same length
        if input_lengths is None:
            input_lengths = (
                inputs.new_ones(inputs.shape[0], dtype=torch.long) * inputs.shape[1]
            )

        # F0 extraction
        if feats_lengths is None:
            if self.enable_warnings:
                logging.warning('Number of pitch frames will be different from mel frames.')
            pitch = [self._calc_f0(x[:xl]) for x, xl in zip(inputs, input_lengths)]
        else:
            pitch = [
                self._calc_f0_corrected(x[:xl], feats_length)
                for x, xl, feats_length in zip(inputs, input_lengths, feats_lengths)
            ]

        # (Optional): Average by duration to calculate token-wise f0
        if self.use_token_averaged_f0:
            durations = durations * self.reduction_factor
            pitch = [
                self._average_by_duration(p, d).view(-1)
                for p, d in zip(pitch, durations)
            ]
            pitch_lengths = durations_lengths
        else:
            pitch_lengths = inputs.new_tensor([len(p) for p in pitch], dtype=torch.long)

        # Padding
        pitch = pad_list(pitch, 0.0)

        # Return with the shape (B, T, 1)
        return pitch.unsqueeze(-1), pitch_lengths

    def _calc_f0(self, inp: torch.Tensor) -> torch.Tensor:
        x = inp.cpu().numpy().astype(np.double)
        sound = parselmouth.Sound(x, self.fs)
        pitch = sound.to_pitch(pitch_floor=self.f0min, pitch_ceiling=self.f0max)
        f0 = np.array([p[0] for p in pitch.selected_array])
        return self._get_f0_tensor(inp, f0)

    def _calc_f0_corrected(self, inp: torch.Tensor, feats_length) -> torch.Tensor:
        x = inp.cpu().numpy().astype(np.double)
        # Praat uses window length of 3.
        # This padding method gives `floor(input_length / hop_length + 1)` frames, same as LogMelFbank
        padding = self.padding - (len(inp) % self.hop_length) // 2
        x = np.pad(x, (padding, padding))
        sound = parselmouth.Sound(x, self.fs)
        time_step = self.hop_length / self.fs
        pitch = sound.to_pitch(pitch_floor=self.f0min, pitch_ceiling=self.f0max, time_step=time_step)
        f0 = np.array([p[0] for p in pitch.selected_array])
        # len(f0) and feats_length should usually be the same, but could be -1 or 1 due to rounding errors
        diff = feats_length - len(f0)
        if diff > 0:
            if self.enable_warnings:
                logging.warning(f'f0 length ({len(f0)}) shorter than feats length ({feats_length})')
            f0 = np.pad(f0, (0, diff))
        elif diff < 0:
            if self.enable_warnings:
                logging.warning(f'f0 length ({len(f0)}) longer than feats length ({feats_length})')
            f0 = f0[:diff]
        return self._get_f0_tensor(inp, f0)

    def _get_f0_tensor(self, inp: torch.Tensor, f0: np.ndarray):
        if self.use_continuous_f0:
            f0 = self._convert_to_continuous_f0(f0)
        if self.use_log_f0:
            nonzero_idxs = np.where(f0 != 0)[0]
            f0[nonzero_idxs] = np.log(f0[nonzero_idxs])
        return inp.new_tensor(f0.reshape(-1), dtype=torch.float)

    @staticmethod
    def _convert_to_continuous_f0(f0: np.array) -> np.array:
        if (f0 == 0).all():
            logging.warning("All frames seem to be unvoiced.")
            return f0

        # padding start and end of f0 sequence
        start_f0 = f0[f0 != 0][0]
        end_f0 = f0[f0 != 0][-1]
        start_idx = np.where(f0 == start_f0)[0][0]
        end_idx = np.where(f0 == end_f0)[0][-1]
        f0[:start_idx] = start_f0
        f0[end_idx:] = end_f0

        # get non-zero frame index
        nonzero_idxs = np.where(f0 != 0)[0]

        # perform linear interpolation
        interp_fn = interp1d(nonzero_idxs, f0[nonzero_idxs])
        f0 = interp_fn(np.arange(0, f0.shape[0]))

        return f0

    def _average_by_duration(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        assert 0 <= len(x) - d.sum() < self.reduction_factor
        d_cumsum = F.pad(d.cumsum(dim=0), (1, 0))
        x_avg = [
            x[start:end].masked_select(x[start:end].gt(0.0)).mean(dim=0)
            if len(x[start:end].masked_select(x[start:end].gt(0.0))) != 0
            else x.new_tensor(0.0)
            for start, end in zip(d_cumsum[:-1], d_cumsum[1:])
        ]
        return torch.stack(x_avg)
