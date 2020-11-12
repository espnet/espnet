import skvideo.io
from torch.utils.data import Dataset
import glob
import skvideo.io
import torch
import numpy as np
import os
import sys


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.shape[dim]
    return torch.cat([torch.Tensor(vec), torch.zeros(*pad_size)], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        batch = [x for x in batch if x is not None]

        if not batch:
            return torch.Tensor([]), [], []

        max_len = max([x[0].shape[self.dim] for x in batch])

        # pad according to max_len
        pad_xs = [pad_tensor(x[0], pad=max_len, dim=self.dim) for x in batch]

        # stack all
        xs = torch.stack(pad_xs, dim=0)
        len = [x[1] for x in batch]
        dirs = [x[2] for x in batch]
        return xs, len, dirs

    def __call__(self, batch):
        return self.pad_collate(batch)


class Voceleb2Raw(Dataset):
    def __init__(self, data_path, fold=''):
        super(Dataset, self).__init__()
        if fold:
            video_prefix = data_path + '/Videos/' + fold + '/mp4'
        else:
            video_prefix = data_path + '/Videos/*/mp4'
        self.video_list = glob.glob(video_prefix + '/id*/*/*.mp4')

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, item):

        try:
            data = skvideo.io.vread(self.video_list[item])
            r, g, b = data[..., 0], data[..., 1], data[..., 2]
            data = (0.2989 * r + 0.5870 * g + 0.1140 * b) / 255
        except:
            print("ERROR When loading: ", self.video_list[item], file=sys.stderr)
            return None

        return data, len(data), self.video_list[item]

class LRS2Raw(Dataset):
    def __init__(self, data_path):
        super(Dataset, self).__init__()
        self.video_list = glob.glob(data_path + '/*/*/*.mp4')
        print(data_path)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, item):

        try:
            data = skvideo.io.vread(self.video_list[item])
            r, g, b = data[..., 0], data[..., 1], data[..., 2]
            data = (0.2989 * r + 0.5870 * g + 0.1140 * b) / 255
        except:
            print("ERROR When loading: ", self.video_list[item], file=sys.stderr)
            return None

        return data, len(data), self.video_list[item]
