"""Additive noise augmentation of a padded batch of waveforms, on its device.

The same procedure as ``CommonPreprocessor`` with ``noise_scp`` (per utterance:
a random noise from the list at a random offset, repeated if it is shorter than
the utterance, mixed at an SNR drawn uniformly from ``noise_db_range`` relative
to the power of the non-silent part of the speech, then the result divided by
its peak if that exceeds 1), but done for the whole batch on the device of the
batch, typically the GPU, instead of per utterance on the CPU in the data loader
workers. All noise recordings are loaded once, as int16, into one buffer on that
device.

Used in training only: in eval mode, and when ``apply_prob`` is 0, the input is
returned unchanged and the noise is never loaded. The buffer is not saved in
the state dict.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import soundfile
import torch

from espnet2.fileio.read_text import read_2columns_text


class BatchNoiseAugment(torch.nn.Module):
    """Add noise to a batch of single-channel waveforms on their device.

    Args:
        noise_scp: Kaldi-style list ``<id> <path>`` of noise recordings. Each
            channel of a multi-channel recording is used as a separate noise.
        noise_apply_prob: Probability of adding noise to each utterance.
        noise_db_range: SNR range in dB, ``"low_high"`` or a single value.
        fs: Sampling rate of the speech; noise at another rate is resampled
            when it is loaded.
        silence_threshold: A frame is non-silent when its power exceeds this
            fraction of the mean frame power (``detect_non_silence``).
        frame_length: Frame length (samples) of the silence detection.
        frame_shift: Frame shift (samples) of the silence detection.
    """

    def __init__(
        self,
        noise_scp: str,
        noise_apply_prob: float = 1.0,
        noise_db_range: str = "13_15",
        fs: int = 16000,
        silence_threshold: float = 0.01,
        frame_length: int = 1024,
        frame_shift: int = 512,
    ):
        super().__init__()
        self.noise_scp = noise_scp
        self.noise_apply_prob = noise_apply_prob
        bounds = [float(v) for v in str(noise_db_range).split("_")]
        if len(bounds) == 1:
            bounds = bounds * 2
        if len(bounds) != 2 or bounds[0] > bounds[1]:
            raise ValueError(f"noise_db_range must be 'low_high': {noise_db_range}")
        self.noise_db_low, self.noise_db_high = bounds
        self.fs = fs
        self.silence_threshold = silence_threshold
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        # filled on the first forward pass in training mode
        self.register_buffer(
            "bank", torch.empty(0, dtype=torch.int16), persistent=False
        )
        self.register_buffer(
            "starts", torch.empty(0, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "lengths", torch.empty(0, dtype=torch.long), persistent=False
        )

    def extra_repr(self) -> str:
        return (
            f"noise_scp={self.noise_scp}, noise_apply_prob={self.noise_apply_prob}, "
            f"noise_db_range=({self.noise_db_low}, {self.noise_db_high})"
        )

    def _load(self, device: torch.device):
        chunks, lengths = [], []
        for path in read_2columns_text(self.noise_scp).values():
            wav, fs = soundfile.read(path, dtype="int16", always_2d=True)
            if fs != self.fs:
                import librosa

                wav = librosa.resample(
                    wav.T.astype(np.float32) / 32768,
                    orig_sr=fs,
                    target_sr=self.fs,
                    res_type="kaiser_fast",
                )
                wav = np.clip(np.round(wav * 32768), -32768, 32767).astype(np.int16).T
            for ch in range(wav.shape[1]):
                if wav.shape[0] > 0:
                    chunks.append(np.ascontiguousarray(wav[:, ch]))
                    lengths.append(wav.shape[0])
        if not chunks:
            raise RuntimeError(f"no noise in {self.noise_scp}")
        lengths = torch.tensor(lengths, dtype=torch.long)
        self.bank = torch.from_numpy(np.concatenate(chunks)).to(device)
        self.lengths = lengths.to(device)
        self.starts = (torch.cumsum(lengths, 0) - lengths).to(device)
        logging.info(
            f"BatchNoiseAugment: {len(chunks)} noises, "
            f"{lengths.sum().item() / self.fs / 3600:.1f} h, "
            f"{self.bank.numel() * 2 / 1e9:.1f} GB on {device}"
        )

    def _non_silent_power(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Mean power of the non-silent samples of each row, as detect_non_silence.

        x: (B, T) float, valid: (B, T) bool. Returns (B,).
        """
        b, t = x.shape
        fl, fsh = self.frame_length, self.frame_shift
        if t < fl:  # detect_non_silence treats a short signal as all non-silent
            n = valid.sum(1).clamp(min=1)
            return (x.pow(2) * valid).sum(1) / n
        # frames padded at the end with zeros (framing(..., padded=True))
        n_frames = -(-(t - fl) // fsh) + 1
        padded = torch.nn.functional.pad(x * valid, (0, (n_frames - 1) * fsh + fl - t))
        frame_power = padded.unfold(1, fl, fsh).pow(2).mean(-1)  # (B, F)
        # frames that start inside each row
        length = valid.sum(1, keepdim=True)
        frame_valid = (torch.arange(n_frames, device=x.device) * fsh) < length
        mean_power = (frame_power * frame_valid).sum(1, keepdim=True) / frame_valid.sum(
            1, keepdim=True
        ).clamp(min=1)
        active = frame_power / mean_power.clamp(min=1e-30) > self.silence_threshold
        active = active & frame_valid
        # all-zero rows: every sample counts (and their power is 0)
        active = torch.where(mean_power > 0, active, frame_valid)
        # sample i takes the decision of frame i // shift; the tail repeats the last
        idx = (torch.arange(t, device=x.device) // fsh).clamp(max=n_frames - 1)
        mask = active[:, idx] & valid
        n = mask.sum(1).clamp(min=1)
        return (x.pow(2) * mask).sum(1) / n

    def forward(
        self, speech: torch.Tensor, speech_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Add noise to speech (B, T) in training mode.

        Returns the augmented speech (same shape and dtype) and speech_lengths.
        """
        if not self.training or self.noise_apply_prob <= 0:
            return speech, speech_lengths
        if speech.dim() != 2:
            raise ValueError(f"speech must be (Batch, Time): {tuple(speech.shape)}")
        if self.bank.numel() == 0:
            self._load(speech.device)

        b, t = speech.shape
        device = speech.device
        x = speech.float()
        if speech_lengths is None:
            valid = torch.ones(b, t, dtype=torch.bool, device=device)
        else:
            valid = torch.arange(t, device=device) < speech_lengths.to(device)[:, None]

        power = self._non_silent_power(x, valid)

        # a noise, an offset and an SNR per utterance
        which = torch.randint(len(self.lengths), (b,), device=device)
        n_len = self.lengths[which]
        # long noise: an offset where the whole utterance fits;
        # short noise: any offset, repeated (np.pad(mode="wrap") in the preprocessor)
        span = torch.clamp(n_len - t, min=0) + (n_len < t) * n_len
        offset = (torch.rand(b, device=device) * span.clamp(min=1)).long()
        pos = (offset[:, None] + torch.arange(t, device=device)) % n_len[:, None]
        noise = self.bank[self.starts[which][:, None] + pos].float() / 32768
        noise = noise * valid

        n = valid.sum(1).clamp(min=1)
        noise_power = noise.pow(2).sum(1) / n
        snr = torch.empty(b, device=device).uniform_(
            self.noise_db_low, self.noise_db_high
        )
        scale = 10 ** (-snr / 20) * power.sqrt() / noise_power.clamp(min=1e-10).sqrt()
        apply = torch.rand(b, device=device) < self.noise_apply_prob
        x = x + (scale * apply)[:, None] * noise

        # as the preprocessor: divide by the peak when it exceeds 1
        peak = (x.abs() * valid).amax(1, keepdim=True)
        x = torch.where(peak > 1.0, x / peak.clamp(min=1e-30), x)
        return x.to(speech.dtype), speech_lengths
