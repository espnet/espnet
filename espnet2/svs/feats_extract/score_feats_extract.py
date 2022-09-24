import logging
from typing import Any, Dict, Optional, Tuple, Union

import humanfriendly
import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.layers.stft import Stft
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract


def ListsToTensor(xs):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        y = x + [0] * (max_len - len(x))
        ys.append(y)
    return ys


class FrameScoreFeats(AbsFeatsExtract):
    def __init__(
        self,
        fs: Union[int, str] = 22050,
        n_fft: int = 1024,
        win_length: int = 512,
        hop_length: int = 128,
        window: str = "hann",
        center: bool = True,
    ):
        assert check_argument_types()
        super().__init__()

        self.fs = fs
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = window
        self.center = center

    def extra_repr(self):
        return (
            f"win_length={self.win_length}, "
            f"hop_length={self.hop_length}, "
            f"center={self.center}, "
        )

    def output_size(self) -> int:
        return 1

    def get_parameters(self) -> Dict[str, Any]:
        return dict(
            fs=self.fs,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            win_length=self.win_length,
            center=self.stft.center,
        )

    def label_aggregate(
        self, input: torch.Tensor, input_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """lage_aggregate function.
        Args:
            input: (Batch, Nsamples, Label_dim)
            input_lengths: (Batch)
        Returns:
            output: (Batch, Frames, Label_dim)
        """
        bs = input.size(0)
        max_length = input.size(1)
        label_dim = input.size(2)

        # NOTE(jiatong):
        #   The default behaviour of label aggregation is compatible with
        #   torch.stft about framing and padding.

        # Step1: center padding
        if self.center:
            pad = self.win_length // 2
            max_length = max_length + 2 * pad
            input = torch.nn.functional.pad(input, (0, 0, pad, pad), "constant", 0)
            input[:, :pad, :] = input[:, pad : (2 * pad), :]
            input[:, (max_length - pad) : max_length, :] = input[
                :, (max_length - 2 * pad) : (max_length - pad), :
            ]
            nframe = (max_length - self.win_length) // self.hop_length + 1

        # Step2: framing
        output = input.as_strided(
            (bs, nframe, self.win_length, label_dim),
            (max_length * label_dim, self.hop_length * label_dim, label_dim, 1),
        )

        # Step3: aggregate label
        _tmp = output.sum(dim=-1, keepdim=False).float()
        output = _tmp[:, :, self.win_length // 2]

        # Step4: process lengths
        if input_lengths is not None:
            if self.center:
                pad = self.win_length // 2
                input_lengths = input_lengths + 2 * pad

            olens = (input_lengths - self.win_length) // self.hop_length + 1
            output.masked_fill_(make_pad_mask(olens, output, 1), 0.0)
        else:
            olens = None

        return output, olens

    def forward(
        self,
        durations: Optional[torch.Tensor] = None,
        durations_lengths: Optional[torch.Tensor] = None,
        score: Optional[torch.Tensor] = None,
        score_lengths: Optional[torch.Tensor] = None,
        tempo: Optional[torch.Tensor] = None,
        tempo_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """FrameScoreFeats forward function.

        Args:
            durations: (Batch, Nsamples)
            durations_lengths: (Batch)
            score: (Batch, Nsamples)
            score_lengths: (Batch)
            tempo: (Batch, Nsamples)
            tempo_lengths: (Batch)

        Returns:
            output: (Batch, Frames)
        """
        durations, durations_lengths = self.label_aggregate(
            durations, durations_lengths
        )
        score, score_lengths = self.label_aggregate(score, score_lengths)
        tempo, tempo_lengths = self.label_aggregate(tempo, tempo_lengths)
        return (
            durations,
            durations_lengths,
            score,
            score_lengths,
            tempo,
            tempo_lengths,
        )


class SyllableScoreFeats(AbsFeatsExtract):
    def __init__(
        self,
        fs: Union[int, str] = 22050,
        n_fft: int = 1024,
        win_length: int = 512,
        hop_length: int = 128,
        window: str = "hann",
        center: bool = True,
    ):
        assert check_argument_types()
        super().__init__()

        self.fs = fs
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = window
        self.center = center

    def extra_repr(self):
        return (
            f"win_length={self.win_length}, "
            f"hop_length={self.hop_length}, "
            f"center={self.center}, "
        )

    def output_size(self) -> int:
        return 1

    def get_parameters(self) -> Dict[str, Any]:
        return dict(
            fs=self.fs,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            win_length=self.win_length,
            center=self.stft.center,
        )

    def get_segments(
        self,
        durations: Optional[torch.Tensor] = None,
        durations_lengths: Optional[torch.Tensor] = None,
        score: Optional[torch.Tensor] = None,
        score_lengths: Optional[torch.Tensor] = None,
        tempo: Optional[torch.Tensor] = None,
        tempo_lengths: Optional[torch.Tensor] = None,
    ):
        seq = [0]
        for i in range(durations_lengths):
            if durations[seq[-1]] != durations[i]:
                seq.append(i)
        seq.append(durations_lengths.item())

        seq.append(0)
        for i in range(score_lengths):
            if score[seq[-1]] != score[i]:
                seq.append(i)
        seq.append(score_lengths.item())
        seq = list(set(seq))
        seq.sort()

        lengths = len(seq) - 1
        seg_duartion = []
        seg_score = []
        seg_tempo = []
        for i in range(lengths):
            l, r = seq[i], seq[i + 1]

            tmp_duartion = durations[l:r][(r - l) // 2]
            tmp_score = score[l:r][(r - l) // 2]
            tmp_tempo = tempo[l:r][(r - l) // 2]

            seg_duartion.append(tmp_duartion.item())
            seg_score.append(tmp_score.item())
            seg_tempo.append(tmp_tempo.item())

        return seg_duartion, lengths, seg_score, lengths, seg_tempo, lengths

    def forward(
        self,
        durations: Optional[torch.Tensor] = None,
        durations_lengths: Optional[torch.Tensor] = None,
        score: Optional[torch.Tensor] = None,
        score_lengths: Optional[torch.Tensor] = None,
        tempo: Optional[torch.Tensor] = None,
        tempo_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """SyllableScoreFeats forward function.

        Args:
            durations: (Batch, Nsamples)
            durations_lengths: (Batch)
            score: (Batch, Nsamples)
            score_lengths: (Batch)
            tempo: (Batch, Nsamples)
            tempo_lengths: (Batch)

        Returns:
            output: (Batch, Frames)
        """
        assert durations.shape == score.shape and score.shape == tempo.shape
        assert (
            durations_lengths.shape == score_lengths.shape
            and score_lengths.shape == tempo_lengths.shape
        )

        bs = durations.size(0)
        seg_durations, seg_durations_lengths = [], []
        seg_score, seg_score_lengths = [], []
        seg_tempo, seg_tempo_lengths = [], []

        for i in range(bs):
            seg = self.get_segments(
                durations=durations[i],
                durations_lengths=durations_lengths[i],
                score=score[i],
                score_lengths=score_lengths[i],
                tempo=tempo[i],
                tempo_lengths=tempo_lengths[i],
            )
            seg_durations.append(seg[0])
            seg_durations_lengths.append(seg[1])
            seg_score.append(seg[2])
            seg_score_lengths.append(seg[3])
            seg_tempo.append(seg[4])
            seg_tempo_lengths.append(seg[5])

        seg_durations = torch.LongTensor(ListsToTensor(seg_durations)).to(
            durations.device
        )
        seg_durations_lengths = torch.LongTensor(seg_durations_lengths).to(
            durations.device
        )
        seg_score = torch.LongTensor(ListsToTensor(seg_score)).to(durations.device)
        seg_score_lengths = torch.LongTensor(seg_score_lengths).to(durations.device)
        seg_tempo = torch.LongTensor(ListsToTensor(seg_tempo)).to(durations.device)
        seg_tempo_lengths = torch.LongTensor(seg_tempo_lengths).to(durations.device)

        return (
            seg_durations,
            seg_durations_lengths,
            seg_score,
            seg_score_lengths,
            seg_tempo,
            seg_tempo_lengths,
        )
