#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
# Copyright 2022 Yifeng Yu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Length regulator related modules."""

import logging

import torch

from espnet.nets.pytorch_backend.nets_utils import pad_list


class LengthRegulator(torch.nn.Module):
    """Length Regulator"""

    def __init__(self, pad_value=0.0):
        """Initilize length regulator module.
        Args:
            pad_value (float, optional): Value used for padding.
        """
        super().__init__()
        self.pad_value = pad_value
        self.winlen = 1024
        self.hoplen = 256
        self.sr = 24000

    def LR(self, xs, notepitch, ds):
        output = list()
        frame_pitch = list()
        mel_len = list()
        xs = torch.transpose(xs, 1, -1)
        frame_lengths = list()

        for batch, expand_target in zip(xs, ds):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            frame_lengths.append(expanded.shape[0])

        for batch, expand_target in zip(notepitch, ds):
            expanded_pitch = self.expand_pitch(batch, expand_target)
            frame_pitch.append(expanded_pitch)

        max_len = max(frame_lengths)
        output_padded = torch.FloatTensor(xs.size(0), max_len, xs.size(2))
        output_padded.zero_()
        frame_pitch_padded = torch.FloatTensor(notepitch.size(0), max_len)
        frame_pitch_padded.zero_()
        for i in range(output_padded.size(0)):
            output_padded[i, : frame_lengths[i], :] = output[i]
        for i in range(frame_pitch_padded.size(0)):
            length = len(frame_pitch[i])
            frame_pitch[i].extend([0] * (max_len - length))
            frame_pitch_tensor = torch.LongTensor(frame_pitch[i])
            frame_pitch_padded[i] = frame_pitch_tensor
        output_padded = torch.transpose(output_padded, 1, -1)

        return output_padded, frame_pitch_padded, torch.LongTensor(frame_lengths)

    def expand_pitch(self, batch, predicted):
        out = list()
        predicted = predicted.squeeze()
        for i, vec in enumerate(batch):
            duration = predicted[i].item()
            if self.sr * duration - self.winlen > 0:
                expand_size = max((self.sr * duration - self.winlen) / self.hoplen, 1)
            elif duration == 0:
                expand_size = 0
            else:
                expand_size = 1
            vec_expand = (
                vec.expand(max(int(expand_size), 0), 1).squeeze(1).cpu().numpy()
            )
            out.extend(vec_expand)

        torch.LongTensor(out).to(vec.device)
        return out

    def expand(self, batch, predicted):
        out = list()
        predicted = predicted.squeeze()
        for i, vec in enumerate(batch):

            duration = predicted[i].item()
            if self.sr * duration - self.winlen > 0:
                expand_size = max((self.sr * duration - self.winlen) / self.hoplen, 1)
            elif duration == 0:
                expand_size = 0
            else:
                expand_size = 1
            vec_expand = vec.expand(max(int(expand_size), 0), -1)
            out.append(vec_expand)

        out = torch.cat(out, 0)
        return out

    def forward(self, xs, notepitch, ds, x_lengths):
        if ds.sum() == 0:
            logging.warning(
                "predicted durations includes all 0 sequences. "
                "fill the first element with 1."
            )
            ds[ds.sum(dim=1).eq(0)] = 1

        # expand xs
        xs = torch.transpose(xs, 1, 2)
        # print("ds", ds)
        phn_repeat = [torch.repeat_interleave(x, d, dim=0) for x, d in zip(xs, ds)]
        output = pad_list(phn_repeat, self.pad_value)  # (B, D_frame, dim)
        output = torch.transpose(output, 1, 2)

        # expand pitch
        notepitch = torch.detach(notepitch)
        pitch_repeat = [
            torch.repeat_interleave(n, d, dim=0) for n, d in zip(notepitch, ds)
        ]
        frame_pitch = pad_list(pitch_repeat, self.pad_value)  # (B, D_frame)

        x_lengths = torch.LongTensor([len(i) for i in pitch_repeat]).to(
            frame_pitch.device
        )
        return output, frame_pitch, x_lengths
