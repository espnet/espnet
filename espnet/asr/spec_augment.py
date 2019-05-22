"""
MIT License

Copyright (c) 2019 Zach Caceres, modified by Shigeki Karita

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import random

import numpy
from PIL import Image
from PIL.Image import BICUBIC


def time_warp(x, window=80):
    """time warp for spec agument
    :param numpy.ndarray x: (time, freq)
    :param int window: warp center ~ uniform(window, time - window) for uniform(-window, window) frames
    :returns numpy.ndarray: (time, freq)
    """
    # avoid randint range error
    if x.shape[0] - window > window:
        return x
    center = random.randint(window, x.shape[0] - window)
    w = random.randint(-window, window)
    left = Image.fromarray(x[:center]).resize((x.shape[1], center + w), BICUBIC)
    right = Image.fromarray(x[center:]).resize((x.shape[1], x.shape[0] - center - w), BICUBIC)
    return numpy.concatenate((left, right), 0)


def freq_mask(x, F=30, num_masks=2, replace_with_zero=True, inplace=False):
    """freq mask for spec agument
    :param numpy.ndarray x: (time, freq)
    :param int num_masks: the number of masks
    :param bool inplace: overwrite
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    if inplace:
        cloned = x
    else:
        cloned = x.copy()

    num_mel_channels = cloned.shape[1]
    fs = numpy.random.randint(0, F, size=(num_masks, 2))

    for f, mask_end in fs:
        f_zero = random.randrange(0, num_mel_channels - f)
        mask_end += f_zero

        # avoids randrange error if values are equal and range is empty
        if f_zero == f_zero + f:
            continue

        if replace_with_zero:
            cloned[:, f_zero:mask_end] = 0
        else:
            cloned[:, f_zero:mask_end] = cloned.mean()
    return cloned


def time_mask(spec, T=40, num_masks=2, replace_with_zero=True, inplace=False):
    """freq mask for spec agument
    :param numpy.ndarray spec: (time, freq)
    :param int num_masks: the number of masks
    :param bool inplace: overwrite
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    if inplace:
        cloned = spec
    else:
        cloned = spec.copy()
    len_spectro = cloned.shape[0]
    ts = numpy.random.randint(0, T, size=(num_masks, 2))
    for t, mask_end in ts:
        t_zero = random.randint(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if t_zero == t_zero + t:
            continue

        mask_end += t_zero
        if replace_with_zero:
            cloned[t_zero:mask_end] = 0
        else:
            cloned[t_zero:mask_end] = cloned.mean()
    return cloned


def spec_augment(x, max_time_warp=40, max_freq_width=27, n_freq_mask=2, max_time_width=100, n_time_mask=2):
    """spec agument
    :param numpy.ndarray x: (time, freq)
    :param int num_masks: the number of masks
    :param bool inplace: overwrite
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    assert isinstance(x, numpy.ndarray)
    assert x.ndim == 2
    x = time_warp(x, max_time_warp)
    x = freq_mask(x, max_freq_width, n_freq_mask, inplace=True)
    x = time_mask(x, max_time_width, n_time_mask, inplace=True)
    return x
