import pytest
import torch

from espnet2.layers.augmentation import (
    DataAugmentation,
    bandpass_filtering,
    bandreject_filtering,
    clipping,
    contrast,
    corrupt_phase,
    deemphasis,
    equalization_filtering,
    highpass_filtering,
    lowpass_filtering,
    pitch_shift,
    polarity_inverse,
    preemphasis,
    reverse,
    speed_perturb,
    time_stretch,
)


def test_lowpass_filtering():
    audio = torch.randn(1000)
    sr = 8000
    ret = lowpass_filtering(audio, sr, cutoff_freq=1000, Q=0.707)


def test_highpass_filtering():
    audio = torch.randn(1000)
    sr = 8000
    ret = highpass_filtering(audio, sr, cutoff_freq=3000, Q=0.707)


@pytest.mark.parametrize("const_skirt_gain", [True, False])
def test_bandpass_filtering(const_skirt_gain):
    audio = torch.randn(1000)
    sr = 8000
    ret = bandpass_filtering(
        audio, sr, center_freq=2000, Q=0.707, const_skirt_gain=const_skirt_gain
    )


def test_bandreject_filtering():
    audio = torch.randn(2000)
    sr = 8000
    ret = bandreject_filtering(audio, sr, center_freq=2000, Q=0.707)


def test_contrast():
    audio = torch.randn(1000)
    sr = 8000
    ret = contrast(audio, sr, enhancement_amount=75)


def test_equalization_filtering():
    audio = torch.randn(1000)
    sr = 8000
    ret = equalization_filtering(audio, sr, center_freq=2000, gain=0, Q=0.707)


@pytest.mark.parametrize("n_steps", [-4, 5])
def test_pitch_shift(n_steps):
    audio = torch.randn(1000)
    sr = 8000
    ret = pitch_shift(audio, sr, n_steps=n_steps, bins_per_octave=12)


@pytest.mark.parametrize("factor", [0.9, 1.1])
def test_speed_perturb(factor):
    audio = torch.randn(1000)
    sr = 8000
    ret = speed_perturb(audio, sr, factor=factor)


@pytest.mark.parametrize("factor", [0.9, 1.1])
def test_time_stretch(factor):
    audio = torch.randn(1000)
    sr = 8000
    ret = time_stretch(audio, sr, factor=factor)


def test_preemphasis():
    audio = torch.randn(1000)
    sr = 8000
    ret = preemphasis(audio, sr, coeff=0.97)


def test_deemphasis():
    audio = torch.randn(1000)
    sr = 8000
    ret = deemphasis(audio, sr, coeff=0.97)


def test_clipping():
    audio = torch.randn(1000)
    sr = 8000
    ret = clipping(audio, sr, min_quantile=0.1, max_quantile=0.9)


def test_polarity_inverse():
    audio = torch.randn(1000)
    sr = 8000
    ret = polarity_inverse(audio, sr)


def test_reverse():
    audio = torch.randn(1000)
    sr = 8000
    ret = reverse(audio, sr)


def test_phase_corruption():
    audio = torch.randn(1000)
    sr = 8000
    ret = corrupt_phase(audio, sr)


@pytest.mark.parametrize("apply_n", [[1, 1], [1, 4]])
def test_data_augmentation(apply_n):
    effects = [
        [0.1, "lowpass", {"cutoff_freq": 1000, "Q": 0.707}],
        [0.1, "highpass", {"cutoff_freq": 3000, "Q": 0.707}],
        [0.1, "equalization", {"center_freq": 1000, "gain": 0, "Q": 0.707}],
        [
            0.1,
            [
                [0.3, "speed_perturb", {"factor": 0.9}],
                [0.4, "speed_perturb", {"factor": 1.1}],
            ],
        ],
        [
            0.5,
            [
                [0.5, "clipping", {"min_quantile": 0.1, "max_quantile": 0.9}],
                [0.5, "reverse", {}],
            ],
        ],
    ]
    data_aug = DataAugmentation(effects, apply_n)
    audio = torch.randn(1000)
    sr = 8000
    ret = data_aug(audio, sr)
