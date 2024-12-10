import math
import random
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torchaudio

# Avaiable sampling rates for bandwidth limitation
SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)

RESAMPLE_METHODS = (
    "kaiser_best",
    "kaiser_fast",
    "scipy",
    "polyphase",
)


def weighted_sample_without_replacement(population, weights, k, rng=random):
    """
        Sample `k` unique elements from a population based on provided weights.

    This function samples `k` elements from a given population without replacement,
    ensuring that each element is sampled according to its specified weight. It uses
    a probabilistic method to select elements, which means that elements with higher
    weights have a greater chance of being selected.

    Args:
        population (list): A list of elements from which to sample.
        weights (list): A list of weights corresponding to each element in the
            population. The weights must be positive and have the same length as
            the population.
        k (int): The number of unique elements to sample. Must be less than or
            equal to the length of the population.
        rng (random.Random, optional): An optional random number generator. If
            not provided, the default `random` module will be used.

    Returns:
        list: A list of `k` unique elements sampled from the population.

    Raises:
        ValueError: If `k` is greater than the length of the population.

    Examples:
        >>> population = ['apple', 'banana', 'cherry', 'date']
        >>> weights = [0.1, 0.2, 0.3, 0.4]
        >>> weighted_sample_without_replacement(population, weights, 2)
        ['cherry', 'date']  # Example output may vary due to randomness.

        >>> weighted_sample_without_replacement(population, weights, 0)
        []  # Sampling 0 elements returns an empty list.

    Note:
        The function ensures that no element is selected more than once.
    """
    if k == 0:
        return []
    if k > len(population):
        raise ValueError(
            "Cannot take a larger sample than population when without replacement"
        )
    v = [rng.random() ** (1 / w) for w in weights]
    order = sorted(range(len(population)), key=lambda i: v[i])
    return [population[i] for i in order[-k:]]


class DataAugmentation:
    """
    A series of data augmentation effects that can be applied to a given waveform.

    Note: Currently we only support single-channel waveforms.

    Args:
        effects (list): A list of effects to be applied to the waveform.
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
                - Note that the effects defined in each type2 data are mutually exclusive
                  (i.e., only one of them can be applied each time).
                  This can be useful when you want to avoid applying some specific
                  effects at the same time.
        apply_n (list): Range of the number of effects to be applied to the waveform.

    Attributes:
        effects (tuple): A tuple containing the effects to be applied.
        effect_probs (tuple): A tuple containing the probabilities of sampling each effect.
        apply_n (tuple): A tuple indicating the range of the number of effects to apply.

    Examples:
        >>> augmentation = DataAugmentation(
        ...     effects=[
        ...         [0.5, "lowpass", {"cutoff_freq": 1000}],
        ...         [0.5, "highpass", {"cutoff_freq": 3000}],
        ...     ],
        ...     apply_n=(1, 2)
        ... )
        >>> augmented_waveform = augmentation(waveform, sample_rate)

    Raises:
        AssertionError: If the apply_n values are not valid (e.g., if the lower bound
        is greater than the upper bound).
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
    """
        Lowpass filter the input signal.

    This function applies a lowpass filter to the provided audio waveform, allowing
    frequencies below the cutoff frequency to pass while attenuating frequencies above
    it. The filter is defined using the Q factor, which describes the filter's bandwidth
    relative to its center frequency.

    Args:
        waveform (torch.Tensor): audio signal (..., time)
        sample_rate (int): sampling rate in Hz
        cutoff_freq (int): filter cutoff frequency
        Q (float or torch.Tensor): quality factor, defines the filter's bandwidth
            https://en.wikipedia.org/wiki/Q_factor

    Returns:
        ret (torch.Tensor): filtered signal (..., time)

    Examples:
        >>> import torch
        >>> sample_rate = 16000
        >>> waveform = torch.randn(1, sample_rate)  # Simulated audio signal
        >>> filtered_waveform = lowpass_filtering(waveform, sample_rate, 1000, 0.707)
    """
    ret = torchaudio.functional.lowpass_biquad(waveform, sample_rate, cutoff_freq, Q=Q)
    return ret


def highpass_filtering(
    waveform, sample_rate: int, cutoff_freq: int = 3000, Q: float = 0.707
):
    """
        Highpass filter the input signal.

    This function applies a highpass filter to the given audio waveform,
    removing frequencies below a specified cutoff frequency.

    Args:
        waveform (torch.Tensor): Audio signal with shape (..., time).
        sample_rate (int): Sampling rate in Hz.
        cutoff_freq (int): Filter cutoff frequency in Hz. Default is 3000.
        Q (float): Quality factor of the filter, where a higher Q indicates
            a narrower bandwidth. Default is 0.707.

    Returns:
        torch.Tensor: Filtered signal with shape (..., time).

    Examples:
        >>> import torch
        >>> waveform = torch.randn(1, 16000)  # Simulated audio waveform
        >>> sample_rate = 16000
        >>> filtered_waveform = highpass_filtering(waveform, sample_rate)
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
    """
        Bandpass filter the input signal.

    This function applies a bandpass filter to the given audio waveform, allowing
    frequencies within a specified range to pass through while attenuating frequencies
    outside this range. The filter's characteristics can be adjusted using the center
    frequency and quality factor (Q).

    Args:
        waveform (torch.Tensor): audio signal (..., time)
        sample_rate (int): sampling rate in Hz
        center_freq (int): filter's center frequency
        Q (float or torch.Tensor): quality factor, which affects the bandwidth
            of the filter. For more details, refer to:
            https://en.wikipedia.org/wiki/Q_factor
        const_skirt_gain (bool): If True, uses a constant skirt gain (peak gain = Q).
            If False, uses a constant 0dB peak gain.

    Returns:
        ret (torch.Tensor): filtered signal (..., time)

    Examples:
        >>> import torch
        >>> sample_rate = 16000
        >>> waveform = torch.randn(1, sample_rate)  # 1 second of random noise
        >>> filtered_waveform = bandpass_filtering(waveform, sample_rate,
        ...                                         center_freq=3000, Q=0.707)
    """
    ret = torchaudio.functional.bandpass_biquad(
        waveform, sample_rate, center_freq, Q=Q, const_skirt_gain=const_skirt_gain
    )
    return ret


def bandreject_filtering(
    waveform, sample_rate: int, center_freq: int = 3000, Q: float = 0.707
):
    """
        Two-pole band-reject filter the input signal.

    This filter attenuates frequencies around a specified center frequency while
    allowing other frequencies to pass. It is useful for removing specific
    interference or noise from a signal.

    Args:
        waveform (torch.Tensor): audio signal (..., time)
        sample_rate (int): sampling rate in Hz
        center_freq (int): filter's center frequency
        Q (float or torch.Tensor): https://en.wikipedia.org/wiki/Q_factor

    Returns:
        ret (torch.Tensor): filtered signal (..., time)

    Examples:
        >>> waveform = torch.randn(1, 16000)  # Example waveform
        >>> sample_rate = 16000
        >>> filtered_waveform = bandreject_filtering(waveform, sample_rate,
        ...                                            center_freq=3000, Q=0.707)
    """
    ret = torchaudio.functional.bandreject_biquad(
        waveform, sample_rate, center_freq, Q=Q
    )
    return ret


def contrast(waveform, sample_rate: int = 16000, enhancement_amount: float = 75.0):
    """
        Apply contrast effect to the input signal to make it sound louder.

    This function applies a contrast enhancement effect to the given audio
    waveform. The enhancement amount controls the degree of contrast applied,
    making the signal sound louder. The enhancement_amount parameter allows
    values from 0 to 100, with 0 still providing a noticeable effect.

    Args:
        waveform (torch.Tensor): audio signal (..., time).
        sample_rate (int): sampling rate in Hz (not used).
        enhancement_amount (float): controls the amount of the enhancement.
            Allowed range of values for enhancement_amount: 0-100.
            Note that enhancement_amount = 0 still gives a significant
            contrast enhancement.

    Returns:
        ret (torch.Tensor): filtered signal (..., time).

    Examples:
        >>> import torch
        >>> waveform = torch.randn(1, 16000)  # Example waveform
        >>> enhanced_waveform = contrast(waveform, 16000, 75.0)
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
    """
        Equalization filter the input signal.

    This function applies an equalization filter to the input audio waveform,
    allowing for control over the center frequency, gain, and quality factor (Q).
    The equalization can boost or attenuate specific frequency ranges based on the
    given parameters.

    Args:
        waveform (torch.Tensor): audio signal (..., time).
        sample_rate (int): sampling rate in Hz.
        center_freq (int): filter's center frequency.
        gain (float or torch.Tensor): desired gain at the boost (or attenuation) in dB.
        Q (float or torch.Tensor): https://en.wikipedia.org/wiki/Q_factor.

    Returns:
        ret (torch.Tensor): filtered signal (..., time).

    Examples:
        >>> import torch
        >>> waveform = torch.randn(1, 16000)  # Simulated audio signal
        >>> filtered_waveform = equalization_filtering(waveform, 16000, 1000, 5.0, 0.707)
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
    """
        Shift the pitch of a waveform by `n_steps` steps.

    Note: this function is slow.

    Args:
        waveform (torch.Tensor): audio signal (..., time).
        sample_rate (int): sampling rate in Hz.
        n_steps (int): the (fractional) steps to shift the pitch.
            -4 for shifting pitch down by 4/`bins_per_octave` octaves,
            4 for shifting pitch up by 4/`bins_per_octave` octaves.
        bins_per_octave (int): number of steps per octave.
        n_fft (float): length of FFT (in second).
        win_length (float or None): The window length (in second) used for STFT.
            If None, it is treated as equal to n_fft.
        hop_length (float): The hop size (in second) used for STFT.
        window (str or None): The windowing function applied to the signal after
            padding with zeros.

    Returns:
        ret (torch.Tensor): filtered signal (..., time).

    Examples:
        >>> waveform = torch.randn(1, 16000)  # example waveform
        >>> sample_rate = 16000
        >>> n_steps = 4
        >>> shifted_waveform = pitch_shift(waveform, sample_rate, n_steps)
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
    """
    Speed perturbation which also changes the pitch.

    Note: This function should be used with caution as it changes the signal
    duration.

    Args:
        waveform (torch.Tensor): audio signal (..., time).
        sample_rate (int): sampling rate in Hz.
        factor (float): speed factor (e.g., 0.9 for 90% speed).

    Returns:
        ret (torch.Tensor): perturbed signal (..., time).

    Examples:
        >>> import torch
        >>> waveform = torch.randn(16000)  # Example waveform of 1 second
        >>> sample_rate = 16000
        >>> perturbed_waveform = speed_perturb(waveform, sample_rate, factor=0.9)
        >>> perturbed_waveform.shape
        torch.Size([14400])  # Duration will be shorter due to speed perturbation
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
    """
        Time scaling (speed up in time without modifying pitch) via phase vocoder.

    Note: This function should be used with caution as it changes the signal duration.

    Args:
        waveform (torch.Tensor): audio signal (..., time)
        sample_rate (int): sampling rate in Hz
        factor (float): speed-up factor (e.g., 0.9 for 90% speed and
            1.3 for 130% speed)
        n_fft (float): length of FFT (in seconds)
        win_length (float or None): The window length (in seconds) used for STFT
            If None, it is treated as equal to n_fft
        hop_length (float): The hop size (in seconds) used for STFT
        window (str or None): The windowing function applied to the signal after
            padding with zeros

    Returns:
        ret (torch.Tensor): perturbed signal (..., time)

    Examples:
        >>> waveform = torch.randn(1, 16000)  # Simulated audio waveform
        >>> sample_rate = 16000
        >>> factor = 1.5  # Speed up by 50%
        >>> stretched_waveform = time_stretch(waveform, sample_rate, factor)
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
    """
    Pre-emphasize a waveform along the time dimension.

    The pre-emphasis operation is defined as:
        y[i] = x[i] - coeff * x[i - 1]

    This process enhances the high-frequency components of the audio signal, which
    can be beneficial for certain types of processing, such as speech recognition.

    Args:
        waveform (torch.Tensor): audio signal (..., time)
        sample_rate (int): sampling rate in Hz (not used)
        coeff (float): pre-emphasis coefficient. Typically between 0.0 and 1.0.

    Returns:
        ret (torch.Tensor): pre-emphasized signal (..., time)

    Examples:
        >>> import torch
        >>> waveform = torch.tensor([0.0, 1.0, 0.5, 0.2])
        >>> sample_rate = 16000
        >>> preemphasized_waveform = preemphasis(waveform, sample_rate)
        >>> print(preemphasized_waveform)
    """
    waveform = waveform.clone()
    waveform[..., 1:] -= coeff * waveform[..., :-1]
    return waveform


def deemphasis(waveform, sample_rate: int, coeff: float = 0.97):
    """
        De-emphasize a waveform along the time dimension.

    This function applies a de-emphasis filter to the input waveform, which is
    typically used in speech processing to reduce the high-frequency content of
    the signal. The de-emphasis operation is defined as:

        y[i] = x[i] + coeff * y[i - 1]

    where `x` is the input waveform, `y` is the output waveform, and `coeff` is
    the de-emphasis coefficient.

    Args:
        waveform (torch.Tensor): audio signal (..., time).
        sample_rate (int): sampling rate in Hz (not used).
        coeff (float): de-emphasis coefficient. Typically between 0.0 and 1.0.

    Returns:
        ret (torch.Tensor): de-emphasized signal (..., time).

    Examples:
        >>> import torch
        >>> waveform = torch.tensor([0.5, 0.3, 0.1, -0.1, -0.3])
        >>> sample_rate = 16000
        >>> coeff = 0.97
        >>> de_emphasized_waveform = deemphasis(waveform, sample_rate, coeff)
    """
    a_coeffs = waveform.new_tensor([1.0, -coeff])
    b_coeffs = waveform.new_tensor([1.0, 0.0])
    return torchaudio.functional.lfilter(waveform, a_coeffs=a_coeffs, b_coeffs=b_coeffs)


def clipping(
    waveform, sample_rate: int, min_quantile: float = 0.0, max_quantile: float = 0.9
):
    """
        Apply the clipping distortion to the input signal.

    This function clips the waveform to the specified quantile limits. The
    clipping is performed based on the provided minimum and maximum quantiles,
    which determine the lower and upper bounds of the signal values that will
    be retained. Any values outside these bounds will be clipped.

    Args:
        waveform (torch.Tensor): audio signal (..., time)
        sample_rate (int): sampling rate in Hz (not used)
        min_quantile (float): lower bound on the total percent of samples to be clipped
        max_quantile (float): upper bound on the total percent of samples to be clipped

    Returns:
        ret (torch.Tensor): clipped signal (..., time)

    Examples:
        >>> waveform = torch.tensor([0.1, 0.5, 0.9, 1.0, 1.5, 2.0])
        >>> clipped_waveform = clipping(waveform, sample_rate=16000,
        ...                              min_quantile=0.0, max_quantile=0.9)
        >>> print(clipped_waveform)
        tensor([0.1, 0.5, 0.9, 0.9, 0.9, 0.9])
    """
    q = waveform.new_tensor([min_quantile, max_quantile])
    min_, max_ = torch.quantile(waveform, q, dim=-1, keepdim=True)
    ret = torch.clamp(waveform, min_, max_)
    return ret


def polarity_inverse(waveform, sample_rate):
    """
        Invert the polarity of the input waveform.

    This function takes an audio waveform as input and returns the waveform
    with its polarity inverted, effectively multiplying the waveform by -1.
    This can be useful in audio processing to create effects or mitigate
    phase issues.

    Args:
        waveform (torch.Tensor): The audio signal to be processed. It should
            have the shape (..., time).
        sample_rate (int): The sampling rate of the audio signal (not used
            in this function).

    Returns:
        torch.Tensor: The waveform with inverted polarity, having the same
            shape as the input waveform.

    Examples:
        >>> import torch
        >>> waveform = torch.tensor([0.5, -0.2, 0.3, -0.1])
        >>> inverted_waveform = polarity_inverse(waveform, sample_rate=16000)
        >>> print(inverted_waveform)
        tensor([-0.5000,  0.2000, -0.3000,  0.1000])
    """
    return -waveform


def reverse(waveform, sample_rate):
    """
        Reverse the input waveform along the time dimension.

    Args:
        waveform (torch.Tensor): audio signal (..., time).
        sample_rate (int): sampling rate in Hz (not used).

    Returns:
        torch.Tensor: the reversed audio signal (..., time).

    Examples:
        >>> waveform = torch.tensor([1.0, 2.0, 3.0])
        >>> reversed_waveform = reverse(waveform, sample_rate=16000)
        >>> print(reversed_waveform)
        tensor([3.0, 2.0, 1.0])
    """
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
    """
    Add random noise to the phase of the input waveform.

    This function modifies the phase of the input audio signal by adding random
    noise, which can help in data augmentation tasks. It is particularly useful
    in scenarios where robustness to phase variations is desired.

    Args:
        waveform (torch.Tensor): Audio signal (..., time).
        sample_rate (int): Sampling rate in Hz.
        scale (float): Scale factor for the phase noise. Default is 0.5.
        n_fft (float): Length of FFT in seconds. Default is 0.032.
        win_length (float or None): The window length (in seconds) used for STFT.
            If None, it is treated as equal to n_fft.
        hop_length (float): The hop size (in seconds) used for STFT. Default is 0.008.
        window (str or None): The windowing function applied to the signal after
            padding with zeros. Default is "hann".

    Returns:
        ret (torch.Tensor): Phase-corrupted signal (..., time).

    Examples:
        >>> waveform = torch.randn(1, 16000)  # Simulated waveform
        >>> sample_rate = 16000
        >>> noisy_waveform = corrupt_phase(waveform, sample_rate)

    Note:
        The random noise added to the phase may alter the perceived audio quality.
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


def bandwidth_limitation(waveform, sample_rate: int, res_type="random"):
    """
    Apply the bandwidth limitation distortion to the input signal.

    This function reduces the effective sampling rate of the input waveform
    by randomly selecting from available lower sampling rates and resampling
    the signal accordingly. It is particularly useful for simulating
    bandwidth-limited conditions in audio processing.

    Args:
        waveform (np.ndarray): A single speech sample (..., Time).
        sample_rate (int): Input sampling rate in Hz.
        res_type (str): Resampling method. Defaults to "random".
            Valid options include:
                - "kaiser_best"
                - "kaiser_fast"
                - "scipy"
                - "polyphase"

    Returns:
        ret (np.ndarray): Bandwidth-limited speech sample (..., Time).

    Examples:
        >>> import numpy as np
        >>> sample_rate = 16000
        >>> waveform = np.random.rand(sample_rate)  # Example waveform
        >>> limited_waveform = bandwidth_limitation(waveform, sample_rate)

    Note:
        If no valid lower sampling rates are found, the original waveform
        will be returned without any changes.

    Raises:
        ValueError: If the input waveform is not a 1D or 2D numpy array.
    """
    fs = sample_rate
    fs_opts = [fs_new for fs_new in SAMPLE_RATES if fs_new < fs]
    if fs_opts:
        fs_new = np.random.choice(fs_opts)
    else:
        return waveform
    if res_type == "random":
        res_type = np.random.choice(RESAMPLE_METHODS)
    opts = {"res_type": res_type}
    if waveform.ndim == 1:
        length = waveform.size(0)
    else:
        length = waveform.size(1)
    ret = librosa.resample(waveform.cpu().numpy(), orig_sr=fs, target_sr=fs_new, **opts)
    # resample back to the original sampling rate
    ret = librosa.resample(ret, orig_sr=fs_new, target_sr=fs, **opts)
    return torch.from_numpy(ret[:length]).to(device=waveform.device)


effects_dict = {
    "lowpass": lowpass_filtering,
    "highpass": highpass_filtering,
    "bandpass": bandpass_filtering,
    "bandreject": bandreject_filtering,
    "bandwidth_limitation": bandwidth_limitation,
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
