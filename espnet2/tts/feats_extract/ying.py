# modified from https://github.com/dhchoi99/NANSY
# We have modified the implementation of dhchoi99 to be fully differentiable.
import math
from typing import Any, Dict, Tuple, Union

import torch

from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet2.tts.feats_extract.yin import *
from espnet.nets.pytorch_backend.nets_utils import pad_list


class Ying(AbsFeatsExtract):
    def __init__(
        self,
        fs: int = 22050,
        w_step: int = 256,
        W: int = 2048,
        tau_max: int = 2048,
        midi_start: int = -5,
        midi_end: int = 75,
        octave_range: int = 24,
        use_token_averaged_ying: bool = False,
    ):
        super().__init__()
        self.fs = fs
        self.w_step = w_step
        self.W = W
        self.tau_max = tau_max
        self.use_token_averaged_ying = use_token_averaged_ying
        self.unfold = torch.nn.Unfold((1, self.W), 1, 0, stride=(1, self.w_step))
        midis = list(range(midi_start, midi_end))
        self.len_midis = len(midis)
        c_ms = torch.tensor([self.midi_to_lag(m, octave_range) for m in midis])
        self.register_buffer("c_ms", c_ms)
        self.register_buffer("c_ms_ceil", torch.ceil(self.c_ms).long())
        self.register_buffer("c_ms_floor", torch.floor(self.c_ms).long())

    def output_size(self) -> int:
        return 1

    def get_parameters(self) -> Dict[str, Any]:
        return dict(
            fs=self.fs,
            w_step=self.w_step,
            W=self.W,
            tau_max=self.tau_max,
            use_token_averaged_ying=self.use_token_averaged_ying,
        )

    def midi_to_lag(self, m: int, octave_range: float = 12):
        """converts midi-to-lag, eq. (4)

        Args:
            m: midi
            fs: sample_rate
            octave_range:

        Returns:
            lag: time lag(tau, c(m)) calculated from midi, eq. (4)

        """
        f = 440 * math.pow(2, (m - 69) / octave_range)
        lag = self.fs / f
        return lag

    def yingram_from_cmndf(self, cmndfs: torch.Tensor) -> torch.Tensor:
        """yingram calculator from cMNDFs
        (cumulative Mean Normalized Difference Functions)

        Args:
            cmndfs: torch.Tensor
                calculated cumulative mean normalized difference function
                for details, see models/yin.py or eq. (1) and (2)
            ms: list of midi(int)
            fs: sampling rate

        Returns:
            y:
                calculated batch yingram


        """
        # c_ms = np.asarray([Pitch.midi_to_lag(m, fs) for m in ms])
        # c_ms = torch.from_numpy(c_ms).to(cmndfs.device)

        y = (cmndfs[:, self.c_ms_ceil] - cmndfs[:, self.c_ms_floor]) / (
            self.c_ms_ceil - self.c_ms_floor
        ).unsqueeze(0) * (self.c_ms - self.c_ms_floor).unsqueeze(0) + cmndfs[
            :, self.c_ms_floor
        ]
        return y

    def yingram(self, x: torch.Tensor):
        """calculates yingram from raw audio (multi segment)

        Args:
            x: raw audio, torch.Tensor of shape (t)
            W: yingram Window Size
            tau_max:
            fs: sampling rate
            w_step: yingram bin step size

        Returns:
            yingram: yingram. torch.Tensor of shape (80 x t')

        """
        # x.shape: t -> B,T, B,T = x.shape
        B, T = x.shape
        w_len = self.W

        frames = self.unfold(x.view(B, 1, 1, T))
        frames = frames.permute(0, 2, 1).contiguous().view(-1, self.W)  # [B* frames, W]
        # If not using gpu, or torch not compatible,
        # implemented numpy batch function is still fine
        dfs = differenceFunctionTorch(frames, frames.shape[-1], self.tau_max)
        cmndfs = cumulativeMeanNormalizedDifferenceFunctionTorch(dfs, self.tau_max)
        yingram = self.yingram_from_cmndf(cmndfs)  # [B*frames,F]
        yingram = yingram.view(B, -1, self.len_midis).permute(0, 2, 1)  # [B,F,T]
        return yingram

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

    @staticmethod
    def _adjust_num_frames(x: torch.Tensor, num_frames: torch.Tensor) -> torch.Tensor:
        x_length = x.shape[1]
        if num_frames > x_length:
            x = F.pad(x, (0, num_frames - x_length))
        elif num_frames < x_length:
            x = x[:num_frames]
        return x

    def forward(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor = None,
        feats_lengths: torch.Tensor = None,
        durations: torch.Tensor = None,
        durations_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_lengths is None:
            input_lengths = (
                input.new_ones(input.shape[0], dtype=torch.long) * input.shape[1]
            )
        # Compute the YIN pitch
        # ying = self.yingram(input)
        # ying_lengths = torch.ceil(input_lengths.float() * self.w_step / self.W).long()

        # TODO(yifeng): now we pass batch_size = 1,
        # maybe remove batch_size in self.yingram
        # print("input", input.shape)
        ying = [
            self.yingram(x[:xl].unsqueeze(0)).squeeze(0)
            for x, xl in zip(input, input_lengths)
        ]
        # print("yingram", ying[0].shape)

        # (Optional): Adjust length to match with the mel-spectrogram
        if feats_lengths is not None:
            ying = [
                self._adjust_num_frames(p, fl).transpose(0, 1)
                for p, fl in zip(ying, feats_lengths)
            ]

        # print("yingram2", ying[0].shape)

        # Use token-averaged f0
        if self.use_token_averaged_ying:
            durations = durations * self.reduction_factor
            ying = [
                self._average_by_duration(p, d).view(-1)
                for p, d in zip(ying, durations)
            ]
            ying_lengths = durations_lengths
        else:
            ying_lengths = input.new_tensor([len(p) for p in ying], dtype=torch.long)

        # Padding
        ying = pad_list(ying, 0.0)

        # print("yingram3", ying.shape)

        return (
            ying.float(),
            ying_lengths,
        )  # TODO(yifeng): should float() be here?

    def crop_scope(
        self, x, yin_start, scope_shift
    ):  # x: tensor [B,C,T] #scope_shift: tensor [B]
        return torch.stack(
            [
                x[
                    i,
                    yin_start
                    + scope_shift[i] : yin_start
                    + self.yin_scope
                    + scope_shift[i],
                    :,
                ]
                for i in range(x.shape[0])
            ],
            dim=0,
        )


if __name__ == "__main__":
    import librosa as rosa
    import matplotlib.pyplot as plt
    import torch

    wav = torch.tensor(rosa.load("LJ001-0002.wav", fs=22050, mono=True)[0]).unsqueeze(0)
    #    wav = torch.randn(1,40965)

    wav = torch.nn.functional.pad(wav, (0, (-wav.shape[1]) % 256))
    #    wav = wav[#:,:8096]
    print(wav.shape)
    pitch = Ying()

    with torch.no_grad():
        ps = pitch.yingram(torch.nn.functional.pad(wav, (1024, 1024)))
        ps = torch.nn.functional.pad(ps, (0, 0, 8, 8), mode="replicate")
        print(ps.shape)
        spec = torch.stft(wav, 1024, 256, return_complex=False)
        print(spec.shape)
        plt.subplot(2, 1, 1)
        plt.pcolor(ps[0].numpy(), cmap="magma")
        plt.colorbar()
        plt.subplot(2, 1, 2)
        plt.pcolor(ps[0][15:65, :].numpy(), cmap="magma")
        plt.colorbar()
        plt.show()
