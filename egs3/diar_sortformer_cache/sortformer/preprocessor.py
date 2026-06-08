"""NeMo-free port of ``AudioToMelSpectrogramPreprocessor`` (FilterbankFeatures).

Faithful reimplementation of NVIDIA NeMo's log-mel feature extractor as
configured for Sortformer (``normalize="per_feature"``, 80 mel bins, 25 ms
window / 10 ms stride, n_fft 512, pre-emphasis 0.97, slaney mel filters,
power spectrogram, natural-log compression). Matching these features is
required for the released encoder weights to behave correctly.

Reference (Apache-2.0): NVIDIA/NeMo
    nemo/collections/asr/parts/preprocessing/features.py
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

CONSTANT = 1e-5


def normalize_batch(x: torch.Tensor, seq_len: torch.Tensor):
    """Normalize each mel bin to zero mean / unit variance over valid frames.

    Statistics are computed per sample and per feature (per mel bin), using only
    the frames within ``seq_len``; padding frames are excluded and then zeroed.

    Args:
        x: Log-mel features of shape ``(B, features, T)``.
        seq_len: Valid frame count per sample, shape ``(B,)``.

    Returns:
        Normalized features of shape ``(B, features, T)``.
    """
    batch_size, _, max_time = x.shape
    time_steps = (
        torch.arange(max_time, device=x.device)
        .unsqueeze(0)
        .expand(batch_size, max_time)
    )
    valid_mask = time_steps < seq_len.unsqueeze(1)
    x_mean_num = torch.where(valid_mask.unsqueeze(1), x, 0.0).sum(axis=2)
    x_mean_den = valid_mask.sum(axis=1)
    x_mean = x_mean_num / x_mean_den.unsqueeze(1)
    x_std = torch.sqrt(
        torch.sum(
            torch.where(valid_mask.unsqueeze(1), x - x_mean.unsqueeze(2), 0.0) ** 2,
            axis=2,
        )
        / (x_mean_den.unsqueeze(1) - 1.0)
    )
    x_std = x_std.masked_fill(x_std.isnan(), 0.0)
    x_std = x_std + CONSTANT
    normalized = (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)
    normalized.masked_fill_(~valid_mask.unsqueeze(1), 0.0)
    return normalized


def _mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax):
    """Build a Slaney-normalized mel filterbank.

    Prefers ``librosa.filters.mel`` (the NeMo reference); falls back to
    ``torchaudio`` Slaney filters when librosa is unavailable. Returns a NumPy
    array of shape ``(n_mels, n_fft // 2 + 1)``.
    """
    try:
        import librosa

        return librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            norm="slaney",
        )
    except ImportError:  # pragma: no cover - librosa is an espnet dependency
        from torchaudio.functional import melscale_fbanks

        fb = melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=fmin,
            f_max=fmax,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        return fb.T.numpy()


class MelSpectrogramPreprocessor(nn.Module):
    """NeMo-faithful log-mel feature extractor for Sortformer.

    Turns a batch of raw waveforms into log-mel spectrogram features that match
    NVIDIA NeMo's ``AudioToMelSpectrogramPreprocessor`` exactly, which is required
    for the released encoder weights to behave correctly. The pipeline is:
    optional dither (train only), pre-emphasis, STFT (Hann window, ``center=True``),
    power spectrogram, Slaney mel projection, natural-log compression, optional
    per-feature normalization, padding-frame masking, and padding the time axis up
    to a multiple of ``pad_to``.

    Defaults follow the Sortformer config: 16 kHz, 25 ms window / 10 ms stride
    (giving 100 fps features), ``n_fft=512``, 80 mel bins, pre-emphasis 0.97,
    power spectrogram (``mag_power=2.0``), and ``normalize="per_feature"``.

    Args:
        sample_rate: Input sample rate in Hz.
        window_size: STFT window length in seconds.
        window_stride: STFT hop length in seconds.
        n_fft: FFT size; if falsy, the next power of two above the window.
        features: Number of mel bins.
        lowfreq: Lowest mel filter frequency in Hz.
        highfreq: Highest mel filter frequency in Hz (defaults to Nyquist).
        preemph: Pre-emphasis coefficient, or ``None`` to disable.
        dither: Gaussian noise std added during training only.
        mag_power: Spectrogram magnitude power (2.0 = power spectrogram).
        log_zero_guard_value: Constant added before the log.
        pad_to: Pad the time axis up to a multiple of this many frames.
        pad_value: Fill value for padded frames.
        normalize: ``"per_feature"`` to z-normalize each mel bin over valid
            frames, or any other value to skip normalization.

    Example:
        >>> prep = MelSpectrogramPreprocessor()
        >>> wav = torch.randn(2, 16000)          # 2 x 1 s of audio
        >>> lengths = torch.tensor([16000, 12000])
        >>> feats, feat_lens = prep(wav, lengths)
        >>> feats.shape                          # (B, 80, T)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_size: float = 0.025,
        window_stride: float = 0.01,
        n_fft: int = 512,
        features: int = 80,
        lowfreq: int = 0,
        highfreq: Optional[int] = None,
        preemph: float = 0.97,
        dither: float = 1e-5,
        mag_power: float = 2.0,
        log_zero_guard_value: float = 2**-24,
        pad_to: int = 16,
        pad_value: float = 0.0,
        normalize: str = "per_feature",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.win_length = int(round(window_size * sample_rate))
        self.hop_length = int(round(window_stride * sample_rate))
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.preemph = preemph
        self.dither = dither
        self.mag_power = mag_power
        self.log_zero_guard_value = log_zero_guard_value
        self.pad_to = pad_to
        self.pad_value = pad_value
        self.normalize = normalize

        window_tensor = torch.hann_window(self.win_length, periodic=False)
        self.register_buffer("window", window_tensor)

        highfreq = highfreq or sample_rate / 2
        fb = torch.tensor(
            _mel_filterbank(sample_rate, self.n_fft, features, lowfreq, highfreq),
            dtype=torch.float,
        )
        self.register_buffer("fb", fb)

    def get_seq_len(self, seq_len: torch.Tensor) -> torch.Tensor:
        """Convert sample counts to output frame counts (matching the STFT)."""
        pad_amount = self.n_fft // 2 * 2
        seq_len = torch.floor_divide(
            (seq_len + pad_amount - self.n_fft), self.hop_length
        )
        return seq_len.to(dtype=torch.long)

    def stft(self, x):
        """Centered STFT with the registered Hann window. Returns complex output."""
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            window=self.window.to(dtype=torch.float, device=x.device),
            return_complex=True,
            pad_mode="constant",
        )

    def forward(
        self, input_signal: torch.Tensor, length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract log-mel features from a batch of waveforms.

        Args:
            input_signal: Raw audio of shape ``(B, num_samples)``.
            length: Valid sample count per waveform, shape ``(B,)``.

        Returns:
            Tuple ``(features, feat_lens)`` where ``features`` has shape
            ``(B, features, T)`` and ``feat_lens`` is the valid frame count per
            sample, shape ``(B,)``. ``T`` is padded up to a multiple of ``pad_to``.
        """
        seq_len_time = length
        seq_len = self.get_seq_len(length)

        x = input_signal
        if self.training and self.dither > 0:
            x = x + self.dither * torch.randn_like(x)

        if self.preemph is not None:
            timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(
                0
            ) < seq_len_time.unsqueeze(1)
            x = torch.cat(
                (x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1
            )
            x = x.masked_fill(~timemask, 0.0)

        with torch.amp.autocast(x.device.type, enabled=False):
            x = self.stft(x)
        x = torch.view_as_real(x)
        x = torch.sqrt(x.pow(2).sum(-1))
        if self.mag_power != 1.0:
            x = x.pow(self.mag_power)

        with torch.amp.autocast(x.device.type, enabled=False):
            x = torch.matmul(self.fb.to(x.dtype), x)
        x = torch.log(x + self.log_zero_guard_value)

        if self.normalize == "per_feature":
            x = normalize_batch(x, seq_len)

        max_len = x.size(-1)
        mask = torch.arange(max_len, device=x.device).repeat(
            x.size(0), 1
        ) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).to(torch.bool), self.pad_value)
        if self.pad_to > 0:
            pad_amt = x.size(-1) % self.pad_to
            if pad_amt != 0:
                x = nn.functional.pad(
                    x, (0, self.pad_to - pad_amt), value=self.pad_value
                )
        return x, seq_len
