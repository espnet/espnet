#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""pytorch dataset and dataloader implementation for chainer training."""

import torch
import torch.utils.data


class Transform:
    """Transform function container.

    lambda can't work well when using DDP because
    lambda is not pickable in the case of multi process.
    This class is required for DDP use case.

    Args:
        converter: batch converter
        load: function object to load data and create minibatch
    """

    def __init__(self, converter, load):
        """Initialize."""
        self._converter = converter
        self._load = load

    def __call__(self, data):
        """Apply a given converter and a given loader."""
        return self._converter([self._load(data)])


class TransformDataset(torch.utils.data.Dataset):
    """Transform Dataset for pytorch backend.

    Args:
        data: list object from make_batchset
        transform: transform function

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

    @staticmethod
    def get_first_element(x):
        """Get first element of a given array-like object."""
        return x[0]

    def __init__(self, **kwargs):
        """Init function."""
        self.loader = torch.utils.data.dataloader.DataLoader(**kwargs)
        if hasattr(self.loader, "__len__"):
            # To support DistribtedSampler.
            # When using DDP, the size of dataset itself is different from
            # the size returned by DataLoader.
            # Unless using length of dataloader, at the end of iterations,
            # this loader class can't recognize the end of each epoch.
            self.len = len(self.loader)
        else:
            self.len = len(kwargs["dataset"])
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
        epoch = serializer("epoch", self.epoch)
        current_position = serializer("current_position", self.current_position)
        self.epoch = epoch
        self.current_position = current_position

    def start_shuffle(self):
        """Shuffle function for sortagrad."""
        self.kwargs["shuffle"] = True
        self.loader = torch.utils.data.dataloader.DataLoader(**self.kwargs)

    def finalize(self):
        """Implement finalize function."""
        del self.loader
