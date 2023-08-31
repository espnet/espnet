import math
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio


def weighted_sample_without_replacement(population, weights, k, rng=random):
    if k > len(population):
        raise ValueError(
            "Cannot take a larger sample than population when without replacement"
        )
    v = [rng.random() ** (1 / w) for w in weights]
    order = sorted(range(len(population)), key=lambda i: v[i])
    return [population[i] for i in order[-k:]]


class DataAugmentation:
    """A series of data augmentation effects that can be applied to a given waveform.

    Note: Currently we only support single-channel waveforms.

    Args:
        effects (list): a list of effects to be applied to the waveform.
            Example:
                [
                    [0.1, "lowpass", {"cutoff_freq": 1000, "Q": 0.707}],
                    [0.1, "highpass", {"cutoff_freq": 3000, "Q": 0.707}],
                    [0.1, "equalization", {"center_freq": 1000, "gain": 0, "Q": 0.707}],
                    [
                        0.1,
                        [
                            [0.3, "speed_perturb", {"factor": 0.9}],
                            [0.3, "speed_perturb", {"factor": 1.1}],
                        ]
                    ],
                ]
            Description:
                - The above list defines a series of data augmentation effects that will
                  be randomly sampled to apply to a given waveform.
                - The data structure of each element can be either
                  type1=Tuple[float, str, Dict] or type2=Tuple[float, type1].
                - In type1, the three values are the weight of sampling this effect, the
                  name (key) of the effect, and the keyword arguments for the effect.
                - In type2, the first value is the weight of sampling this effect.
                  The second value is a list of type1 elements which are similarly
                  defined as above.
                - Note that he effects defined in each type2 data are mutually exclusive
                  (i.e., only one of them can be applied each time).
                  This can be useful when you want to avoid applying some specific
                  effects at the same time.
        apply_n (list): range of the number of effects to be applied to the waveform.
    """

    def __init__(
        self,
        effects: List[
            Union[
                Tuple[float, List[Tuple[float, str, Dict]]],
                Tuple[float, str, Dict],
            ]
        ],
        apply_n: Tuple[int, int] = [1, 1],
    ):
        self.effects = tuple(
            [tup[1] if isinstance(tup[1], list) else tup[1:] for tup in effects]
        )
        self.effect_probs = tuple([tup[0] for tup in effects])
        assert apply_n[0] <= apply_n[1], apply_n
        assert apply_n[1] > 0, apply_n
        self.apply_n = tuple(apply_n)

    def __call__(self, waveform, sample_rate):
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        assert waveform.ndim == 1, waveform.shape
        if self.apply_n[1] > self.apply_n[0]:
            apply_n = np.random.randint(self.apply_n[0], self.apply_n[1] + 1)
        else:
            apply_n = self.apply_n[0]
        for effect in weighted_sample_without_replacement(
            self.effects, weights=self.effect_probs, k=apply_n
        ):
            if isinstance(effect[1], list):
                probs = [tup[0] for tup in effect]
                _, eff, eff_args = weighted_sample_without_replacement(
                    effect, weights=probs, k=1
                )[0]
            else:
                eff, eff_args = effect

            waveform = self._apply_effect(waveform, sample_rate, eff, eff_args)
        return waveform.cpu().numpy()

    def _apply_effect(self, waveform, sample_rate, eff, eff_args):
        eff_args.pop("sample_rate", None)
        return effects_dict[eff](waveform, sample_rate, **eff_args)


def lowpass_filtering(
    waveform, sample_rate: int, cutoff_freq: int = 1000, Q: float = 0.707
):
    """Lowpass filter the input signal.

    Args:
        waveform (torch.Tensor): audio signal (..., time)
        sample_rate (int): sampling rate in Hz
        cutoff_freq (int): filter cutoff frequency
        Q (float or torch.Tensor): https://en.wikipedia.org/wiki/Q_factor

    Returns:
        ret (torch.Tensor): filtered signal (..., time)
    """
    ret = torchaudio.functional.lowpass_biquad(waveform, sample_rate, cutoff_freq, Q=Q)
    return ret


def highpass_filtering(
    waveform, sample_rate: int, cutoff_freq: int = 3000, Q: float = 0.707
):
    """Highpass filter the input signal.

    Args:
        waveform (torch.Tensor): audio signal (..., time)
        sample_rate (int): sampling rate in Hz
        cutoff_freq (int): filter cutoff frequency
        Q (float or torch.Tensor): https://en.wikipedia.org/wiki/Q_factor

    Returns:
        ret (torch.Tensor): filtered signal (..., time)
    """
    ret = torchaudio.functional.highpass_biquad(waveform, sample_rate, cutoff_freq, Q=Q)
    return ret


def bandpass_filtering(
    waveform,
    sample_rate: int,
    center_freq: int = 3000,
    Q: float = 0.707,
    const_skirt_gain: bool = False,
):
    """Bandpass filter the input signal.

    Args:
        waveform (torch.Tensor): audio signal (..., time)
        sample_rate (int): sampling rate in Hz
        center_freq_freq (int): filter's center_freq frequency
        Q (float or torch.Tensor): https://en.wikipedia.org/wiki/Q_factor
        const_skirt_gain (bool): If True, uses a constant skirt gain (peak gain = Q).
            If False, uses a constant 0dB peak gain.

    Returns:
        ret (torch.Tensor): filtered signal (..., time)
    """
    ret = torchaudio.functional.bandpass_biquad(
        waveform, sample_rate, center_freq, Q=Q, const_skirt_gain=const_skirt_gain
    )
    return ret


def bandreject_filtering(
    waveform, sample_rate: int, center_freq: int = 3000, Q: float = 0.707
):
    """Two-pole band-reject filter the input signal.

    Args:
        waveform (torch.Tensor): audio signal (..., time)
        sample_rate (int): sampling rate in Hz
        center_freq_freq (int): filter's center_freq frequency
        Q (float or torch.Tensor): https://en.wikipedia.org/wiki/Q_factor

    Returns:
        ret (torch.Tensor): filtered signal (..., time)
    """
    ret = torchaudio.functional.bandreject_biquad(
        waveform, sample_rate, center_freq, Q=Q
    )
    return ret


def contrast(waveform, sample_rate: int = 16000, enhancement_amount: float = 75.0):
    """Apply contrast effect to the input signal to make it sound louder.

    Args:
        waveform (torch.Tensor): audio signal (..., time)
        sample_rate (int): sampling rate in Hz (not used)
        enhancement_amount (float): controls the amount of the enhancement
            Allowed range of values for enhancement_amount : 0-100
            Note that enhancement_amount = 0 still gives a significant
            contrast enhancement.

    Returns:
        ret (torch.Tensor): filtered signal (..., time)
    """
    ret = torchaudio.functional.contrast(waveform, enhancement_amount)
    return ret


def equalization_filtering(
    waveform,
    sample_rate: int,
    center_freq: int = 1000,
    gain: float = 0.0,
    Q: float = 0.707,
):
    """Equalization filter the input signal.

    Args:
        waveform (torch.Tensor): audio signal (..., time)
        sample_rate (int): sampling rate in Hz
        center_freq (int): filter's center frequency
        gain (float or torch.Tensor): desired gain at the boost (or attenuation) in dB
        Q (float or torch.Tensor): https://en.wikipedia.org/wiki/Q_factor

    Returns:
        ret (torch.Tensor): filtered signal (..., time)
    """
    ret = torchaudio.functional.equalizer_biquad(
        waveform, sample_rate, center_freq, gain, Q=Q
    )
    return ret


def pitch_shift(
    waveform,
    sample_rate: int,
    n_steps: int,
    bins_per_octave: int = 12,
    n_fft: float = 0.032,
    win_length: Optional[float] = None,
    hop_length: float = 0.008,
    window: Optional[str] = "hann",
):
    """Shift the pitch of a waveform by `n_steps` steps.

    Note: this function is slow.

    Args:
        waveform (torch.Tensor): audio signal (..., time)
        sample_rate (int): sampling rate in Hz
        n_steps (int): the (fractional) steps to shift the pitch
            -4 for shifting pitch down by 4/`bins_per_octave` octaves
            4 for shifting pitch up by 4/`bins_per_octave` octaves
        bins_per_octave (int): number of steps per octave
        n_fft (float): length of FFT (in second)
        win_length (float or None): The window length (in second) used for STFT
            If None, it is treated as equal to n_fft
        hop_length (float): The hop size (in second) used for STFT
        window (str or None): The windowing function applied to the signal after
            padding with zeros

    Returns:
        ret (torch.Tensor): filtered signal (..., time)
    """
    n_fft = int(sample_rate * n_fft)
    if hop_length is None:
        hop_length = n_fft // 4
    else:
        hop_length = int(sample_rate * hop_length)
    if win_length is None:
        win_length = n_fft
    if window is not None:
        window_func = getattr(torch, f"{window}_window")
        window = window_func(win_length, dtype=waveform.dtype, device=waveform.device)
    ret = torchaudio.functional.pitch_shift(
        waveform,
        sample_rate,
        n_steps,
        bins_per_octave=bins_per_octave,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
    )
    return ret


def speed_perturb(waveform, sample_rate: int, factor: float):
    """Speed perturbation which also changes the pitch.

    Note: This function should be used with caution as it changes the signal duration.

    Args:
        waveform (torch.Tensor): audio signal (..., time)
        sample_rate (int): sampling rate in Hz
        factor (float): speed factor (e.g., 0.9 for 90% speed)
        lengths (torch.Tensor): lengths of the input signals

    Returns:
        ret (torch.Tensor): perturbed signal (..., time)
    """
    orig_freq = sample_rate
    source_sample_rate = int(factor * orig_freq)
    target_sample_rate = int(orig_freq)

    gcd = math.gcd(source_sample_rate, target_sample_rate)
    source_sample_rate = source_sample_rate // gcd
    target_sample_rate = target_sample_rate // gcd

    ret = torchaudio.functional.resample(
        waveform, source_sample_rate, target_sample_rate
    )
    return ret


def time_stretch(
    waveform,
    sample_rate: int,
    factor: float,
    n_fft: float = 0.032,
    win_length: Optional[float] = None,
    hop_length: float = 0.008,
    window: Optional[str] = "hann",
):
    """Time scaling (speed up in time without modifying pitch) via phase vocoder.

    Note: This function should be used with caution as it changes the signal duration.

    Args:
        waveform (torch.Tensor): audio signal (..., time)
        sample_rate (int): sampling rate in Hz
        factor (float): speed-up factor (e.g., 0.9 for 90% speed and 1.3 for 130% speed)
        n_fft (float): length of FFT (in second)
        win_length (float or None): The window length (in second) used for STFT
            If None, it is treated as equal to n_fft
        hop_length (float): The hop size (in second) used for STFT
        window (str or None): The windowing function applied to the signal after
            padding with zeros

    Returns:
        ret (torch.Tensor): perturbed signal (..., time)
    """
    n_fft = int(sample_rate * n_fft)
    if hop_length is None:
        hop_length = n_fft // 4
    else:
        hop_length = int(sample_rate * hop_length)
    if win_length is None:
        win_length = n_fft
    if window is not None:
        window_func = getattr(torch, f"{window}_window")
        window = window_func(win_length, dtype=waveform.dtype, device=waveform.device)
    spec = torch.stft(
        waveform, n_fft, hop_length, win_length, window=window, return_complex=True
    )
    freq = spec.size(-2)
    phase_advance = torch.linspace(0, math.pi * hop_length, freq)[..., None]
    spec_sp = torchaudio.functional.phase_vocoder(spec, factor, phase_advance)
    len_stretch = int(round(waveform.size(-1) / factor))
    ret = torch.functional.istft(
        spec_sp, n_fft, hop_length, win_length, window=window, length=len_stretch
    )
    return ret


def codecs(
    waveform,
    sample_rate: int,
    format: str,
    compression: Optional[float] = None,
    encoding: Optional[str] = None,
    bits_per_sample: Optional[int] = None,
):
    """Apply the specified codecs to the input signal.

    Warning: Wait until torchaudio 2.1 for this function to work.

    Note:
        1. This function only supports CPU backend.
        2. The GSM codec can be used to emulate phone line channel effects.

    Args:
        waveform (torch.Tensor): audio signal (..., time)
        sample_rate (int): sampling rate in Hz
        format (str): file format.
            Valid values are "wav", "mp3", "ogg", "vorbis", "amr-nb", "amb",
            "flac", "sph", "gsm", and "htk".
        compression (float or None, optional): used for formats other than WAV

            For more details see torchaudio.backend.sox_io_backend.save().
        encoding (str or None, optional): change the encoding for the supported formats
            Valid values are "PCM_S" (signed integer Linear PCM),
            "PCM_U" (unsigned integer Linear PCM), "PCM_F" (floating point PCM),
            "ULAW" (mu-law), and "ALAW" (a-law).
            For more details see torchaudio.backend.sox_io_backend.save().
        bits_per_sample (int or None, optional): change the bit depth
            for the supported formats
            For more details see torchaudio.backend.sox_io_backend.save().

    Returns:
        ret (torch.Tensor): compressed signal (..., time)
    """
    raise NotImplementedError
    ret = torchaudio.functional.apply_codec(
        waveform.unsqueeze(0),
        sample_rate,
        format,
        channels_first=False,
        compression=compression,
        encoding=encoding,
        bits_per_sample=bits_per_sample,
    )
    return ret.squeeze(0)


def preemphasis(waveform, sample_rate: int, coeff: float = 0.97):
    """Pre-emphasize a waveform along the time dimension.

    y[i] = x[i] - coeff * x[i - 1]

    Args:
        waveform (torch.Tensor): audio signal (..., time)
        sample_rate (int): sampling rate in Hz (not used)
        coeff (float): pre-emphasis coefficient. Typically between 0.0 and 1.0.

    Returns:
        ret (torch.Tensor): pre-emphasized signal (..., time)
    """
    waveform = waveform.clone()
    waveform[..., 1:] -= coeff * waveform[..., :-1]
    return waveform


def deemphasis(waveform, sample_rate: int, coeff: float = 0.97):
    """De-emphasize a waveform along the time dimension.

    y[i] = x[i] + coeff * y[i - 1]

    Args:
        waveform (torch.Tensor): audio signal (..., time)
        sample_rate (int): sampling rate in Hz (not used)
        coeff (float): de-emphasis coefficient. Typically between 0.0 and 1.0.

    Returns:
        ret (torch.Tensor): de-emphasized signal (..., time)
    """
    a_coeffs = waveform.new_tensor([1.0, -coeff])
    b_coeffs = waveform.new_tensor([1.0, 0.0])
    return torchaudio.functional.lfilter(waveform, a_coeffs=a_coeffs, b_coeffs=b_coeffs)


def clipping(
    waveform, sample_rate: int, min_quantile: float = 0.0, max_quantile: float = 0.9
):
    """Apply the clipping distortion to the input signal.

    Args:
        waveform (torch.Tensor): audio signal (..., time)
        sample_rate (int): sampling rate in Hz (not used)
        min_quantile (float): lower bound on the total percent of samples to be clipped
        max_quantile (float): upper bound on the total percent of samples to be clipped

    Returns:
        ret (torch.Tensor): clipped signal (..., time)
    """
    q = waveform.new_tensor([min_quantile, max_quantile])
    min_, max_ = torch.quantile(waveform, q, dim=-1, keepdim=True)
    ret = torch.clamp(waveform, min_, max_)
    return ret


def polarity_inverse(waveform, sample_rate):
    return -waveform


def reverse(waveform, sample_rate):
    return torch.flip(waveform, [-1])


def corrupt_phase(
    waveform,
    sample_rate,
    scale: float = 0.5,
    n_fft: float = 0.032,
    win_length: Optional[float] = None,
    hop_length: float = 0.008,
    window: Optional[str] = "hann",
):
    """Adding random noise to the phase of input waveform.

    Args:
        waveform (torch.Tensor): audio signal (..., time)
        sample_rate (int): sampling rate in Hz
        scale (float): scale factor for the phase noise
        n_fft (float): length of FFT (in second)
        win_length (float or None): The window length (in second) used for STFT
            If None, it is treated as equal to n_fft
        hop_length (float): The hop size (in second) used for STFT
        window (str or None): The windowing function applied to the signal after
            padding with zeros

    Returns:
        ret (torch.Tensor): phase-corrupted signal (..., time)
    """
    n_fft = int(sample_rate * n_fft)
    if hop_length is None:
        hop_length = n_fft // 4
    else:
        hop_length = int(sample_rate * hop_length)
    if win_length is None:
        win_length = n_fft
    if window is not None:
        window_func = getattr(torch, f"{window}_window")
        window = window_func(win_length, dtype=waveform.dtype, device=waveform.device)
    spec = torch.stft(
        waveform, n_fft, hop_length, win_length, window=window, return_complex=True
    )
    phase = torch.angle(spec)
    phase = torch.randn_like(phase) * scale + phase
    spec = torch.abs(spec) * torch.exp(1j * phase)
    ret = torch.functional.istft(
        spec, n_fft, hop_length, win_length, window=window, length=waveform.size(-1)
    )
    return ret


effects_dict = {
    "lowpass": lowpass_filtering,
    "highpass": highpass_filtering,
    "bandpass": bandpass_filtering,
    "bandreject": bandreject_filtering,
    "contrast": contrast,
    "equalization": equalization_filtering,
    "pitch_shift": pitch_shift,
    "speed_perturb": speed_perturb,
    "time_stretch": time_stretch,
    "preemphasis": preemphasis,
    "deemphasis": deemphasis,
    "clipping": clipping,
    "polarity_inverse": polarity_inverse,
    "reverse": reverse,
    "corrupt_phase": corrupt_phase,
}
