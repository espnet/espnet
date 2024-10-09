#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import random

import cv2
import numpy as np
import torch
from scipy import signal

__all__ = [
    "Compose",
    "Normalize",
    "CenterCrop",
    "AddNoise",
    "NormalizeUtterance",
    "Identity",
    "SpeedRate",
    "ExpandDims",
    "HorizontalFlip",
    "TimeMask",
    "CutoutHole",
    "RandomCrop",
]


class Compose(object):
    """Compose several preprocess together.
    Args:
        preprocess (list of ``Preprocess`` objects): list of preprocess to compose.
    """

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, img):
        for t in self.preprocess:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.preprocess:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        return (img - self.mean) / self.std

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class CutoutHole(object):
    """Normalize a ndarray image with mean and standard deviation."""

    def __init__(self, min_hole_length, max_hole_length):
        self.min_hole_length = min_hole_length
        self.max_hole_length = max_hole_length

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        frames, h, w = img.shape

        hole_length = random.randint(self.min_hole_length, self.max_hole_length)

        start_h = random.randint(0, h - hole_length)
        start_w = random.randint(0, w - hole_length)

        img[:, start_h : start_h + hole_length, start_w : start_w + hole_length] = 0.0
        return img


class NormalizeUtterance:
    """Normalize per raw audio by removing the
    mean and divided by the standard deviation
    """

    def __call__(self, signal):
        signal_std = 0.0 if np.std(signal) == 0.0 else np.std(signal)
        signal_mean = np.mean(signal)
        return (signal - signal_mean) / signal_std


class CenterCrop(object):
    """Crop the given image at the center"""

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw)) / 2.0)
        delta_h = int(round((h - th)) / 2.0)
        frames = frames[:, delta_h : delta_h + th, delta_w : delta_w + tw]
        return frames


class RandomCrop(object):
    """Crop the given image at the center"""

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w - tw)
        delta_h = random.randint(0, h - th)
        frames = frames[:, delta_h : delta_h + th, delta_w : delta_w + tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)


class HorizontalFlip(object):
    """Flip image horizontally."""

    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be flipped with a probability flip_ratio
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                frames[index] = cv2.flip(frames[index], 1)
        return frames


class TimeMask:
    """time mask"""

    def __init__(self, max_mask_length):
        self.max_mask_length = max_mask_length

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Images to be masked.
        Returns:
            numpy.ndarray: Masked image.
        """
        frames, h, w = img.shape
        max_mask_length = min(frames, self.max_mask_length)

        mask_length = random.randint(0, max_mask_length)
        start = random.randint(0, frames - mask_length)

        img[start : start + mask_length] = 0.0
        return img


class AddNoise(object):
    """Add SNR noise [-1, 1]"""

    def __init__(self, noise, snr_target=None, snr_levels=[-5, 0, 5, 10, 15]):
        assert noise.dtype in [
            np.float32,
            np.float64,
        ], "noise only supports float data type"
        self.noise = noise
        self.snr_levels = snr_levels
        self.snr_target = snr_target

    def get_power(self, clip):
        clip2 = clip.copy()
        clip2 = clip2**2
        return np.sum(clip2) / (len(clip2) * 1.0)

    def __call__(self, signal):
        assert signal.dtype in [
            np.float32,
            np.float64,
        ], "signal only supports float32 data type"
        snr_target = (
            random.choice(self.snr_levels) if not self.snr_target else self.snr_target
        )
        if snr_target == 9999:
            return signal
        else:
            # -- get noise
            start_idx = random.randint(0, len(self.noise) - len(signal))
            noise_clip = self.noise[start_idx : start_idx + len(signal)]
            sig_power = self.get_power(signal)
            noise_clip_power = self.get_power(noise_clip)
            factor = (sig_power / noise_clip_power) / (10 ** (snr_target / 10.0))
            desired_signal = (signal + noise_clip * np.sqrt(factor)).astype(np.float32)
            if random.random() < 0.5:
                max_len = len(desired_signal) // 8
                start_idx = random.sample(range(len(desired_signal)), k=4)
                length = random.choices(range(max_len), k=4)
                for i in range(4):
                    desired_signal[start_idx[i] : start_idx[i] + length[i]] = 0
                return desired_signal
            else:
                return desired_signal


class Identity(object):
    """Identity"""

    def __init__(
        self,
    ):
        pass

    def __call__(self, array):
        return array


class SpeedRate(object):
    """Subsample/Upsample the number of frames in a sequence."""

    def __init__(self, speed_rate=1.0):
        """__init__.

        :param speed_rate: float, the speed rate between the frame rate of \
            the input video and the frame rate used for training.
        """
        self._speed_rate = speed_rate

    def __call__(self, x):
        """
        Args:
            img (numpy.ndarray): sequence to be sampled.
        Returns:
            numpy.ndarray: sampled sequence.
        """
        if self._speed_rate <= 0:
            raise ValueError("speed_rate should be greater than zero.")
        if self._speed_rate == 1.0:
            return x
        old_length = x.shape[0]
        new_length = int(old_length / self._speed_rate)
        old_indices = np.arange(old_length)
        new_indices = np.linspace(
            start=0, stop=old_length, num=new_length, endpoint=False
        )
        new_indices = list(map(int, new_indices))
        x = x[new_indices]
        return x


class ExpandDims(object):
    """ExpandDims."""

    def __init__(
        self,
    ):
        """__init__."""

    def __call__(self, x):
        """__call__.

        :param x: numpy.ndarray, Expand the shape of an array.
        """
        return np.expand_dims(x, axis=1) if x.ndim == 1 else x
