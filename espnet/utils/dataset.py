#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""This module contains pytorch dataset and dataloader implementation for chainer training."""

import torch
import torch.utils.data


class TransformDataset(torch.utils.data.Dataset):
    """Transform Dataset for pytorch backend.

    Args:
        data: list object from make_batchset
        transfrom: transform function

    """

    def __init__(self, data, transform):
        """Init function."""
        super(TransformDataset).__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        """Len function."""
        return len(self.data)

    def __getitem__(self, idx):
        """[] operator."""
        return self.transform(self.data[idx])


class ChainerDataLoader(object):
    """Pytorch dataloader in chainer style.

    Args:
        all args for torch.utils.data.dataloader.Dataloader

    """

    def __init__(self, **kwargs):
        """Init function."""
        self.loader = torch.utils.data.dataloader.DataLoader(**kwargs)
        self.len = len(kwargs['dataset'])
        self.current_position = 0
        self.epoch = 0
        self.iter = None
        self.kwargs = kwargs

    def next(self):
        """Implement next function."""
        if self.iter is None:
            self.iter = iter(self.loader)
        try:
            ret = next(self.iter)
        except StopIteration:
            self.iter = None
            return self.next()
        self.current_position += 1
        if self.current_position == self.len:
            self.epoch = self.epoch + 1
            self.current_position = 0
        return ret

    def __iter__(self):
        """Implement iter function."""
        for batch in self.loader:
            yield batch

    @property
    def epoch_detail(self):
        """Epoch_detail required by chainer."""
        return self.epoch + self.current_position / self.len

    def serialize(self, serializer):
        """Serialize and deserialize function."""
        epoch = serializer('epoch', self.epoch)
        current_position = serializer('current_position', self.current_position)
        self.epoch = epoch
        self.current_position = current_position

    def start_shuffle(self):
        """Shuffle function for sortagrad."""
        self.kwargs['shuffle'] = True
        self.loader = torch.utils.data.dataloader.DataLoader(**self.kwargs)

    def finalize(self):
        """Implement finalize function."""
        del self.loader
