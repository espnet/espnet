from typing import Any, Dict, Optional, Tuple, Union

import torch
from typeguard import typechecked

from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


def ListsToTensor(xs):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        y = x + [0] * (max_len - len(x))
        ys.append(y)
    return ys


class FrameScoreFeats(AbsFeatsExtract):
    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 22050,
        n_fft: int = 1024,
        win_length: int = 512,
        hop_length: int = 128,
        window: str = "hann",
        center: bool = True,
    ):
        if win_length is None:
            win_length = n_fft
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
        label: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None,
        midi: Optional[torch.Tensor] = None,
        midi_lengths: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
        duration_lengths: Optional[torch.Tensor] = None,
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
            label: (Batch, Nsamples)
            label_lengths: (Batch)
            midi: (Batch, Nsamples)
            midi_lengths: (Batch)
            duration: (Batch, Nsamples)
            duration_lengths: (Batch)

        Returns:
            output: (Batch, Frames)
        """
        label, label_lengths = self.label_aggregate(label, label_lengths)
        midi, midi_lengths = self.label_aggregate(midi, midi_lengths)
        duration, duration_lengths = self.label_aggregate(duration, duration_lengths)
        return (
            label,
            label_lengths,
            midi,
            midi_lengths,
            duration,
            duration_lengths,
        )


class SyllableScoreFeats(AbsFeatsExtract):
    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 22050,
        n_fft: int = 1024,
        win_length: int = 512,
        hop_length: int = 128,
        window: str = "hann",
        center: bool = True,
    ):
        if win_length is None:
            win_length = n_fft
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
        label: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None,
        midi: Optional[torch.Tensor] = None,
        midi_lengths: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
        duration_lengths: Optional[torch.Tensor] = None,
    ):
        seq = [0]
        for i in range(label_lengths):
            if label[seq[-1]] != label[i]:
                seq.append(i)
        seq.append(label_lengths.item())

        seq.append(0)
        for i in range(midi_lengths):
            if midi[seq[-1]] != midi[i]:
                seq.append(i)
        seq.append(midi_lengths.item())
        seq = list(set(seq))
        seq.sort()

        lengths = len(seq) - 1
        seg_label = []
        seg_midi = []
        seg_duration = []
        for i in range(lengths):
            l, r = seq[i], seq[i + 1]

            tmp_label = label[l:r][(r - l) // 2]
            tmp_midi = midi[l:r][(r - l) // 2]
            tmp_duration = duration[l:r][(r - l) // 2]

            seg_label.append(tmp_label.item())
            seg_midi.append(tmp_midi.item())
            seg_duration.append(tmp_duration.item())

        return (
            seg_label,
            lengths,
            seg_midi,
            lengths,
            seg_duration,
            lengths,
        )

    def forward(
        self,
        label: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None,
        midi: Optional[torch.Tensor] = None,
        midi_lengths: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
        duration_lengths: Optional[torch.Tensor] = None,
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
            label: (Batch, Nsamples)
            label_lengths: (Batch)
            midi: (Batch, Nsamples)
            midi_lengths: (Batch)
            duration: (Batch, Nsamples)
            duration_lengths: (Batch)

        Returns:
            output: (Batch, Frames)
        """
        assert label.shape == midi.shape and midi.shape == duration.shape
        assert (
            label_lengths.shape == midi_lengths.shape
            and midi_lengths.shape == duration_lengths.shape
        )

        bs = label.size(0)
        seg_label, seg_label_lengths = [], []
        seg_midi, seg_midi_lengths = [], []
        seg_duration, seg_duration_lengths = [], []

        for i in range(bs):
            seg = self.get_segments(
                label=label[i],
                label_lengths=label_lengths[i],
                midi=midi[i],
                midi_lengths=midi_lengths[i],
                duration=duration[i],
                duration_lengths=duration_lengths[i],
            )
            seg_label.append(seg[0])
            seg_label_lengths.append(seg[1])
            seg_midi.append(seg[2])
            seg_midi_lengths.append(seg[3])
            seg_duration.append(seg[6])
            seg_duration_lengths.append(seg[7])

        seg_label = torch.LongTensor(ListsToTensor(seg_label)).to(label.device)
        seg_label_lengths = torch.LongTensor(seg_label_lengths).to(label.device)
        seg_midi = torch.LongTensor(ListsToTensor(seg_midi)).to(label.device)
        seg_midi_lengths = torch.LongTensor(seg_midi_lengths).to(label.device)
        seg_duration = torch.LongTensor(ListsToTensor(seg_duration)).to(label.device)
        seg_duration_lengths = torch.LongTensor(seg_duration_lengths).to(label.device)

        return (
            seg_label,
            seg_label_lengths,
            seg_midi,
            seg_midi_lengths,
            seg_duration,
            seg_duration_lengths,
        )


def expand_to_frame(expand_len, len_size, label, midi, duration):
    # expand phone to frame level
    bs = label.size(0)
    seq_label, seq_label_lengths = [], []
    seq_midi, seq_midi_lengths = [], []
    seq_duration, seq_duration_lengths = [], []

    for i in range(bs):
        length = sum(expand_len[i])
        seq_label_lengths.append(length)
        seq_midi_lengths.append(length)
        seq_duration_lengths.append(length)

        seq_label.append(
            [
                label[i][j]
                for j in range(len_size[i])
                for k in range(int(expand_len[i][j]))
            ]
        )
        seq_midi.append(
            [
                midi[i][j]
                for j in range(len_size[i])
                for k in range(int(expand_len[i][j]))
            ]
        )
        seq_duration.append(
            [
                duration[i][j]
                for j in range(len_size[i])
                for k in range(int(expand_len[i][j]))
            ]
        )

    seq_label = torch.LongTensor(ListsToTensor(seq_label)).to(label.device)
    seq_label_lengths = torch.LongTensor(seq_label_lengths).to(label.device)
    seq_midi = torch.LongTensor(ListsToTensor(seq_midi)).to(label.device)
    seq_midi_lengths = torch.LongTensor(seq_midi_lengths).to(label.device)
    seq_duration = torch.LongTensor(ListsToTensor(seq_duration)).to(label.device)
    seq_duration_lengths = torch.LongTensor(seq_duration_lengths).to(label.device)

    return (
        seq_label,
        seq_label_lengths,
        seq_midi,
        seq_midi_lengths,
        seq_duration,
        seq_duration_lengths,
    )
