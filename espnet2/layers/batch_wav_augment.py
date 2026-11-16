"""Reverberation and additive noise for a padded batch of waveforms, on its device.

The same procedure as ``CommonPreprocessor`` with ``rir_scp`` and ``noise_scp``,
for the whole batch on the device of the batch (typically the GPU) instead of
per utterance on the CPU in the data loader workers. For each utterance:

1. Compute the power of its non-silent part (``detect_non_silence``).
2. With probability ``rir_apply_prob``, convolve it with a random room impulse
   response, keep its first samples (the length does not change), and scale it
   back to the power of step 1.
3. With probability ``noise_apply_prob``, add a random noise at a random offset,
   repeated when it is shorter than the utterance, at an SNR drawn uniformly from
   ``noise_db_range`` relative to the power of step 1.
4. With probability ``music_apply_prob``, add a random recording of a second
   list (``music_scp``) the same way, at an SNR drawn from its own
   ``music_db_range``.
5. With probability ``telephone_apply_prob``, pass it through a telephone
   channel: band-limited to a cut-off drawn from ``telephone_cutoff_range``,
   resampled to 8 kHz, optionally high-passed at ``telephone_highpass_hz``,
   optionally companded to 8 bits (mu-law or A-law, 50/50) and resampled back.
6. Divide the utterance by its peak when the peak exceeds 1.

The labels are not changed: none of the steps moves the speech in time.

All recordings of each list are loaded once into one bank in host memory: the
noise as int16, the impulse responses as float32. By default each step copies
only what the batch needs to its device. With ``keep_on_device=True`` the
whole banks are copied once to the device instead.

The banks are a cache, not buffers: they are not in the state dict, not
broadcast by DistributedDataParallel, and shared by the replicas of
DataParallel. The module works in training mode only: in eval mode it returns
the input unchanged and loads nothing.
"""

import logging
import threading
from typing import Dict, Optional, Tuple

import numpy as np
import soundfile
import torch

from espnet2.fileio.read_text import read_2columns_text

# This lock makes the replicas of DataParallel, which run in threads and share
# the cache, load each list once. It is a module-level object so that the module
# can still be deep-copied.
_LOAD_LOCK = threading.Lock()


def _db_range(name: str, value) -> Tuple[float, float]:
    """Parse ``"low_high"`` (or a single value) into (low, high)."""
    if not isinstance(value, str):
        # YAML reads an unquoted 13_15 as the integer 1315
        logging.warning(
            f"{name}={value!r} is not a string: quote a range in YAML, e.g. '13_15'"
        )
    text = str(value)
    # a leading minus belongs to the first bound: "-10_20" is (-10, 20)
    bounds = [float(v) for v in text.split("_")] if text else []
    if len(bounds) == 1:
        bounds = bounds * 2
    if len(bounds) != 2 or bounds[0] > bounds[1]:
        raise ValueError(f"{name} must be 'low_high': {value}")
    return bounds[0], bounds[1]


def mu_law_8bit(x: torch.Tensor, mu: float = 255.0) -> torch.Tensor:
    """Compress x (in [-1, 1]) with the mu-law, quantise to 8 bits, and expand."""
    y = torch.sign(x) * torch.log1p(mu * x.abs()) / np.log1p(mu)
    y = torch.round(y * 127.5 - 0.5) + 0.5  # 256 levels, midrise
    y = (y / 127.5).clamp(-1, 1)
    return torch.sign(y) * torch.expm1(y.abs() * np.log1p(mu)) / mu


def a_law_8bit(x: torch.Tensor, a: float = 87.6) -> torch.Tensor:
    """Compress x (in [-1, 1]) with the A-law, quantise to 8 bits, and expand."""
    ax = x.abs()
    log_a = 1 + np.log(a)
    y = torch.where(
        ax < 1 / a, a * ax / log_a, (1 + torch.log((a * ax).clamp(min=1))) / log_a
    )
    y = torch.sign(x) * y
    y = torch.round(y * 127.5 - 0.5) + 0.5
    y = (y / 127.5).clamp(-1, 1)
    ay = y.abs()
    e = torch.where(ay < 1 / log_a, ay * log_a / a, torch.exp(ay * log_a - 1) / a)
    return torch.sign(y) * e


class BatchWavAugment(torch.nn.Module):
    """Add reverberation and noise to a batch of single-channel waveforms.

    Args:
        noise_scp: Kaldi-style list ``<id> <path>`` (or ``<path>``) of noise
            recordings, or None for no noise. Each channel of a multi-channel
            recording is used as a separate noise.
        noise_apply_prob: Probability of adding noise to each utterance.
        noise_db_range: SNR range in dB, ``"low_high"`` or a single value.
            Quote a range in YAML (``'13_15'``): unquoted, it is the integer 1315.
        rir_scp: Kaldi-style list ``<id> <path>`` (or ``<path>``) of room impulse
            responses, or None for no reverberation. Each channel of a
            multi-channel file is used as a separate impulse response.
        rir_apply_prob: Probability of convolving each utterance with an
            impulse response.
        rir_shift: Shift the reverberated speech back by the position of the
            impulse response's peak (``argmax |rir|``, the largest absolute value,
            not an estimate of the direct path; each channel of a multi-channel
            file is its own impulse response with its own peak), so that
            it stays aligned with the dry speech and its labels, as Kaldi's
            ``wav-reverberate --shift-output`` (default), Lhotse and SpeechBrain
            do. False keeps the first samples of the convolution, as
            CommonPreprocessor does, which delays the speech by the peak
            position (e.g. the propagation delay of a measured or simulated
            impulse response).
        snr_reference: The speech power the SNR refers to. "dry": the
            non-silent power of the speech before reverberation (the
            reverberated speech is scaled back to it), as CommonPreprocessor
            does. "early": the non-silent power of the early reverberation, the
            speech convolved with the impulse response from 1 ms before its
            peak to 50 ms after it, as Kaldi's wav-reverberate does; the late
            reverberation then counts as distortion, not as speech.
        noise_rir_apply_prob: Probability of convolving the noise of each
            utterance with an impulse response of its own (drawn independently
            of the speech's), so that a point-source noise is reverberated as
            Kaldi's reverberate_data_dir.py does. Needs rir_scp.
        context: Number of leading samples of each row that only give the
            reverberation its history: the row is augmented as a whole, the
            powers and the peak are measured after the context, and the first
            ``context`` samples are removed (also in eval mode, without
            augmentation), so the labels cover the rest. Give the data loader
            windows that start ``context`` samples early (zeros before the start
            of a recording); without it the first samples of a window are
            reverberated as if nothing preceded them. 0 for none.
        fs: Sampling rate of the speech; noise and impulse responses at another
            rate are resampled when they are loaded.
        silence_threshold: A frame is non-silent when its power exceeds this
            fraction of the mean frame power (``detect_non_silence``).
        frame_length: Frame length (samples) of the silence detection.
        frame_shift: Frame shift (samples) of the silence detection.
        keep_on_device: Copy the whole banks to the device of the batch once,
            instead of copying what each batch needs from host memory. Faster,
            but uses 2 bytes of device memory per noise sample (11.4 GB for 99 h
            at 16 kHz) and 4 bytes per impulse-response sample.
        music_scp: Kaldi-style list of a second set of additive recordings
            (e.g. music), added independently of noise_scp with its own
            probability and SNR range, or None.
        music_apply_prob: Probability of adding a recording of music_scp to
            each utterance.
        music_db_range: SNR range in dB of music_scp, ``"low_high"``.
        telephone_apply_prob: Probability of passing each utterance through the
            telephone channel (after reverberation and the additive signals).
        telephone_cutoff_range: Low-pass cut-off range in Hz, ``"low_high"``,
            drawn per utterance; at most 4000 (the 8 kHz Nyquist frequency).
        telephone_highpass_hz: Cut-off of the optional high-pass in Hz.
        telephone_highpass_prob: Probability of the high-pass, per telephone
            utterance.
        telephone_companding_prob: Probability of 8-bit companding at 8 kHz,
            per telephone utterance; mu-law (G.711 mu, 255) or A-law (G.711 A,
            87.6) with equal probability.
        telephone_transition_hz: Width in Hz of the raised-cosine transition of
            the low-pass and the high-pass.
    """

    def __init__(
        self,
        noise_scp: Optional[str] = None,
        noise_apply_prob: float = 1.0,
        noise_db_range: str = "13_15",
        rir_scp: Optional[str] = None,
        rir_apply_prob: float = 1.0,
        rir_shift: bool = True,
        snr_reference: str = "dry",
        noise_rir_apply_prob: float = 0.0,
        context: int = 0,
        fs: int = 16000,
        silence_threshold: float = 0.01,
        frame_length: int = 1024,
        frame_shift: int = 512,
        keep_on_device: bool = False,
        music_scp: Optional[str] = None,
        music_apply_prob: float = 0.0,
        music_db_range: str = "-10_20",
        telephone_apply_prob: float = 0.0,
        telephone_cutoff_range: str = "3400_3950",
        telephone_highpass_hz: float = 300.0,
        telephone_highpass_prob: float = 0.5,
        telephone_companding_prob: float = 0.5,
        telephone_transition_hz: float = 100.0,
    ):
        super().__init__()
        if (
            noise_scp is None
            and rir_scp is None
            and music_scp is None
            and telephone_apply_prob <= 0
        ):
            raise ValueError(
                "give noise_scp, rir_scp, music_scp or telephone_apply_prob > 0"
            )
        self.noise_scp = noise_scp
        self.noise_apply_prob = noise_apply_prob if noise_scp else 0.0
        self.noise_db_low, self.noise_db_high = _db_range(
            "noise_db_range", noise_db_range
        )
        self.music_scp = music_scp
        self.music_apply_prob = music_apply_prob if music_scp else 0.0
        self.music_db_low, self.music_db_high = _db_range(
            "music_db_range", music_db_range
        )
        self.telephone_apply_prob = telephone_apply_prob
        low, high = _db_range("telephone_cutoff_range", telephone_cutoff_range)
        if not 0 < low <= high <= 4000:
            raise ValueError(
                "telephone_cutoff_range must lie in (0, 4000]: "
                f"{telephone_cutoff_range}"
            )
        if telephone_apply_prob > 0 and fs % 8000 != 0:
            raise ValueError(f"the telephone channel needs fs a multiple of 8000: {fs}")
        self.telephone_cutoff_low, self.telephone_cutoff_high = low, high
        self.telephone_highpass_hz = float(telephone_highpass_hz)
        self.telephone_highpass_prob = telephone_highpass_prob
        self.telephone_companding_prob = telephone_companding_prob
        self.telephone_transition_hz = float(telephone_transition_hz)
        self.rir_scp = rir_scp
        self.rir_apply_prob = rir_apply_prob if rir_scp else 0.0
        self.rir_shift = rir_shift
        if snr_reference not in ("dry", "early"):
            raise ValueError(f"snr_reference must be dry or early: {snr_reference}")
        self.snr_reference = snr_reference
        if noise_rir_apply_prob > 0 and not rir_scp:
            raise ValueError("noise_rir_apply_prob > 0 needs rir_scp")
        self.noise_rir_apply_prob = noise_rir_apply_prob if noise_scp else 0.0
        if context < 0:
            raise ValueError(f"context must be >= 0: {context}")
        self.context = int(context)
        self.fs = int(fs)
        self.silence_threshold = silence_threshold
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.keep_on_device = keep_on_device
        # The first training step fills this cache: (kind, "host") holds (bank,
        # starts, lengths) on the CPU, and (kind, device) a copy on that device;
        # kind is "noise" or "rir". It is a plain dict, not buffers: the replicas
        # of DataParallel share it (they copy the module's attributes shallowly),
        # and DistributedDataParallel does not broadcast it.
        self._cache: Dict[Tuple[str, str], Tuple[torch.Tensor, ...]] = {}

    def extra_repr(self) -> str:
        return (
            f"noise_scp={self.noise_scp}, noise_apply_prob={self.noise_apply_prob}, "
            f"noise_db_range=({self.noise_db_low}, {self.noise_db_high}), "
            f"rir_scp={self.rir_scp}, rir_apply_prob={self.rir_apply_prob}, "
            f"rir_shift={self.rir_shift}, snr_reference={self.snr_reference}, "
            f"noise_rir_apply_prob={self.noise_rir_apply_prob}, "
            f"context={self.context}, "
            f"fs={self.fs}, keep_on_device={self.keep_on_device}, "
            f"music_scp={self.music_scp}, music_apply_prob={self.music_apply_prob}, "
            f"music_db_range=({self.music_db_low}, {self.music_db_high}), "
            f"telephone_apply_prob={self.telephone_apply_prob}, "
            f"telephone_cutoff_range=({self.telephone_cutoff_low}, "
            f"{self.telephone_cutoff_high}), "
            f"telephone_highpass_hz={self.telephone_highpass_hz}, "
            f"telephone_highpass_prob={self.telephone_highpass_prob}, "
            f"telephone_companding_prob={self.telephone_companding_prob}"
        )

    def _load(self, kind: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load the noise or music (int16) or impulse-response (float32) list."""
        scp = {"noise": self.noise_scp, "music": self.music_scp, "rir": self.rir_scp}[
            kind
        ]
        chunks, lengths = [], []
        for key, path in read_2columns_text(scp).items():
            # A one-column line holds the path only, as in CommonPreprocessor
            path = path or key
            # Read every file as float. Reading a floating-point file as int16
            # does not scale it.
            wav, fs = soundfile.read(path, dtype="float32", always_2d=True)
            if fs != self.fs:
                import librosa

                # Use librosa's default resampler (soxr), which librosa installs.
                wav = librosa.resample(wav.T, orig_sr=fs, target_sr=self.fs).T
            for ch in range(wav.shape[1]):
                x = wav[:, ch]
                if x.size == 0:
                    continue
                if kind in ("noise", "music"):
                    # Scale the noise to the full int16 range. The mixing
                    # rescales it to the SNR, so its level does not change the
                    # result.
                    peak = np.abs(x).max()
                    if peak > 0:
                        x = x * (32767.0 / peak)
                    x = np.round(x).astype(np.int16)
                chunks.append(np.ascontiguousarray(x))
                lengths.append(x.size)
        if not chunks:
            raise RuntimeError(f"no {kind} in {scp}")
        lengths = torch.tensor(lengths, dtype=torch.long)
        bank = torch.from_numpy(np.concatenate(chunks))
        starts = torch.cumsum(lengths, 0) - lengths
        logging.info(
            f"BatchWavAugment: {len(chunks)} {kind} signals, "
            f"{lengths.sum().item() / self.fs / 3600:.2f} h, "
            f"{bank.numel() * bank.element_size() / 1e9:.2f} GB"
        )
        return bank, starts, lengths

    def bank(self, kind: str, device: torch.device) -> Tuple[torch.Tensor, ...]:
        """Return (bank, starts, lengths) of "noise", "music" or "rir" on ``device``.

        The first call loads the list; each device gets one copy.
        """
        key = (kind, str(device))
        bank = self._cache.get(key)
        if bank is None:
            with _LOAD_LOCK:
                if key not in self._cache:
                    host = (kind, "host")
                    if host not in self._cache:
                        self._cache[host] = self._load(kind)
                    self._cache[key] = tuple(t.to(device) for t in self._cache[host])
                bank = self._cache[key]
        return bank

    def _rir_peaks(self) -> torch.Tensor:
        """Return the index of the peak (``argmax |rir|``) of each impulse response.

        The first call computes them from the host bank; later calls reuse them.
        """
        key = ("rir_peak", "host")
        peaks = self._cache.get(key)
        if peaks is None:
            bank, starts, lengths = self.bank("rir", torch.device("cpu"))
            with _LOAD_LOCK:
                if key not in self._cache:
                    self._cache[key] = torch.stack(
                        [
                            bank[s : s + n].abs().argmax()
                            for s, n in zip(starts.tolist(), lengths.tolist())
                        ]
                    )
                peaks = self._cache[key]
        return peaks

    def _non_silent_power(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Return the mean power of the non-silent samples of each row.

        This function computes the same power as detect_non_silence does for the
        utterance alone: it uses the length of each row, not the batch width.

        x: (B, T) float, valid: (B, T) bool (a prefix of each row). Returns (B,).
        """
        b, t = x.shape
        fl, fsh = self.frame_length, self.frame_shift
        length = valid.sum(1)  # (B,)
        x = x * valid
        # Count the frames of each utterance as framing(centered=False,
        # padded=True) does.
        n_frames = torch.where(
            length >= fl,
            torch.div(length - fl + fsh - 1, fsh, rounding_mode="floor") + 1,
            torch.zeros_like(length),
        )
        f_max = -(-(t - fl) // fsh) + 1 if t >= fl else 1
        padded = torch.nn.functional.pad(x, (0, (f_max - 1) * fsh + fl - t))
        frame_power = padded.unfold(1, fl, fsh).pow(2).mean(-1)[:, :f_max]  # (B, F)
        frame_valid = torch.arange(f_max, device=x.device) < n_frames[:, None]
        mean_power = (frame_power * frame_valid).sum(1, keepdim=True) / n_frames[
            :, None
        ].clamp(min=1)
        active = (frame_power / mean_power.clamp(min=1e-30)) > self.silence_threshold
        # Give sample i the decision of frame i // shift, and the samples after
        # the last frame the decision of the last frame (np.pad(mode="edge")).
        idx = torch.div(torch.arange(t, device=x.device), fsh, rounding_mode="floor")
        idx = torch.minimum(idx[None, :], (n_frames - 1).clamp(min=0)[:, None])
        mask = torch.gather(active, 1, idx)
        # Count every sample of a row that is shorter than one frame or all zero.
        whole = (n_frames == 0)[:, None] | (mean_power == 0)
        mask = torch.where(whole, torch.ones_like(mask), mask) & valid
        return (x.pow(2) * mask).sum(1) / mask.sum(1).clamp(min=1)

    @staticmethod
    def _cut_on_host(bank, start, n_len, offset, t, device):
        """Cut the noise segments of a batch from the host bank and copy them.

        Row i is bank[start[i] + (offset[i] + j) % n_len[i]] for j < t. The rows
        are written into one pinned int16 tensor, which is copied to ``device``
        asynchronously. A row whose noise from the offset is shorter than t
        gets one period of the noise (from the offset) and then copies of what
        is already written, doubling each time, so every row is a few memory
        copies.
        """
        pinned = device.type == "cuda"
        out = torch.empty((len(start), t), dtype=torch.int16, pin_memory=pinned)
        bank_np, out_np = bank.numpy(), out.numpy()
        for row, s, n, o in zip(
            out_np, start.tolist(), n_len.tolist(), offset.tolist()
        ):
            noise = bank_np[s : s + n]
            # row[j] = noise[(o + j) % n]: the rest of the noise from the offset,
            # then its start, then copies of the first period
            head = min(n - o, t)
            row[:head] = noise[o : o + head]
            tail = min(o, t - head)
            row[head : head + tail] = noise[:tail]
            filled = head + tail
            while filled < t:
                c = min(filled, t - filled)
                row[filled : filled + c] = row[:c]
                filled += c
        return out.to(device, non_blocking=pinned)

    def _impulse_responses(self, which: torch.Tensor, device: torch.device):
        """Return the impulse responses ``which`` (CPU) as a padded (B, L) batch."""
        # The batch width comes from the host copy of the lengths, so it does not
        # wait for the GPU.
        host_bank, host_starts, host_lengths = self.bank("rir", torch.device("cpu"))
        width = int(host_lengths[which].max())
        if self.keep_on_device:
            bank, starts, lengths = self.bank("rir", device)
            which = which.to(device)
            n_len = lengths[which]
            pos = torch.arange(width, device=device)
            idx = (starts[which][:, None] + pos).clamp(max=len(bank) - 1)
            return bank[idx] * (pos < n_len[:, None])
        # From host memory: copy each impulse response into one zeroed (pinned)
        # batch, a slice copy per row, and send it to the device asynchronously.
        pinned = device.type == "cuda"
        out = torch.zeros((len(which), width), dtype=host_bank.dtype, pin_memory=pinned)
        bank_np, out_np = host_bank.numpy(), out.numpy()
        for row, s, n in zip(
            out_np, host_starts[which].tolist(), host_lengths[which].tolist()
        ):
            row[:n] = bank_np[s : s + n]
        return out.to(device, non_blocking=pinned)

    def _convolve(
        self,
        spectrum: torch.Tensor,
        rirs: torch.Tensor,
        peak: torch.Tensor,
        t: int,
        n_fft: int,
    ) -> torch.Tensor:
        """Convolve rows given by their spectrum (rfft, n_fft) with rirs (B, L).

        The result keeps t samples of the full convolution: from the impulse
        response's peak with rir_shift, else the first ones.
        """
        y = torch.fft.irfft(spectrum * torch.fft.rfft(rirs, n_fft), n_fft)
        if self.rir_shift:
            # Samples peak .. peak + t - 1 of each row: the signal shifted back
            # by its impulse response's peak. peak + t - 1 <= L + t - 2 < n_fft.
            return y.gather(1, peak[:, None] + torch.arange(t, device=y.device))
        return y[:, :t]

    def _reverberate(
        self, x: torch.Tensor, which: torch.Tensor, early: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Convolve each row of x (B, T) with the impulse response ``which`` (CPU).

        The full convolution is computed through the FFT; see _convolve for the
        samples kept. With early=True, also returns x convolved with the early
        part of each impulse response, 1 ms before the peak to 50 ms after it
        (Kaldi's ComputeEarlyReverbEnergy), aligned as the full one; both reuse
        one FFT of x.
        """
        t = x.size(1)
        device = x.device
        rirs = self._impulse_responses(which, device)
        peak = self._rir_peaks()[which].to(device)
        n_fft = 1 << (t + rirs.size(1) - 2).bit_length()
        spectrum = torch.fft.rfft(x, n_fft)
        full = self._convolve(spectrum, rirs, peak, t, n_fft)
        if not early:
            return full, None
        pos = torch.arange(rirs.size(1), device=device)
        # As Kaldi: the float bounds are truncated to integers, so at a rate
        # where 1 ms or 50 ms is not a whole number of samples (22.05 kHz) the
        # window is the one Kaldi takes.
        lo = (peak.double() - 0.001 * self.fs).long()
        hi = (peak.double() + 0.05 * self.fs).long()
        rirs = rirs * ((pos >= lo[:, None]) & (pos < hi[:, None]))
        return full, self._convolve(spectrum, rirs, peak, t, n_fft)

    def draw(self, lengths: torch.Tensor) -> Dict:
        """Draw the random choices of a batch: what to apply, which and where.

        ``lengths`` (B,) holds the number of samples of each utterance. Returns a
        dict of (B,) tensors: rir_apply, rir_which, noise_apply, noise_which and
        snr (dB) on the CPU, and noise_offset on the device of ``lengths``; the
        same for music (music_snr), and the telephone choices (telephone_apply,
        telephone_cutoff, telephone_highpass, telephone_compand,
        telephone_alaw) on the CPU. As in
        CommonPreprocessor, the noise is cut for each utterance by its own
        length: a noise at least as long as the utterance starts at any offset
        where the whole utterance fits (n_len - length + 1 offsets); a shorter
        noise starts at any offset and repeats.
        """
        batch_size = len(lengths)
        draws = {}
        if self.rir_apply_prob > 0:
            _, _, rir_lengths = self.bank("rir", torch.device("cpu"))
            draws["rir_apply"] = torch.rand(batch_size) < self.rir_apply_prob
            draws["rir_which"] = torch.randint(len(rir_lengths), (batch_size,))
        for kind, prob, low, high in (
            ("noise", self.noise_apply_prob, self.noise_db_low, self.noise_db_high),
            ("music", self.music_apply_prob, self.music_db_low, self.music_db_high),
        ):
            if prob <= 0:
                continue
            _, _, noise_lengths = self.bank(kind, torch.device("cpu"))
            which = torch.randint(len(noise_lengths), (batch_size,))
            u = torch.rand(batch_size)
            draws[f"{kind}_apply"] = torch.rand(batch_size) < prob
            draws[f"{kind}_which"] = which
            draws[f"{kind}_snr"] = torch.empty(batch_size).uniform_(low, high)
            # The offset is computed where the lengths are, so lengths on the
            # GPU are not copied back to the host.
            n_len = noise_lengths[which].to(lengths.device)
            span = torch.where(n_len >= lengths, n_len - lengths + 1, n_len)
            offset = (u.to(lengths.device) * span).long()
            draws[f"{kind}_offset"] = torch.minimum(offset, span - 1)
            if self.noise_rir_apply_prob > 0:
                _, _, rir_lengths = self.bank("rir", torch.device("cpu"))
                draws[f"{kind}_rir_apply"] = (
                    torch.rand(batch_size) < self.noise_rir_apply_prob
                )
                draws[f"{kind}_rir_which"] = torch.randint(
                    len(rir_lengths), (batch_size,)
                )
        if "noise_snr" in draws:
            draws["snr"] = draws.pop("noise_snr")
        if self.telephone_apply_prob > 0:
            draws["telephone_apply"] = (
                torch.rand(batch_size) < self.telephone_apply_prob
            )
            draws["telephone_cutoff"] = torch.empty(batch_size).uniform_(
                self.telephone_cutoff_low, self.telephone_cutoff_high
            )
            draws["telephone_highpass"] = (
                torch.rand(batch_size) < self.telephone_highpass_prob
            )
            draws["telephone_compand"] = (
                torch.rand(batch_size) < self.telephone_companding_prob
            )
            draws["telephone_alaw"] = torch.rand(batch_size) < 0.5
        return draws

    def _additive(self, kind, draws, snr, speech_power, valid, t):
        """Return the noise or music (``kind``) of each row, scaled to its SNR.

        Rows that do not get it are zero.
        """
        device = valid.device
        c = self.context
        bank_device = device if self.keep_on_device else torch.device("cpu")
        bank, starts, lengths = self.bank(kind, bank_device)
        which = draws[f"{kind}_which"].to(bank_device)
        offset = draws[f"{kind}_offset"].to(bank_device)
        n_len = lengths[which]
        if self.keep_on_device:
            pos = torch.arange(t, device=device)
            pos = (offset[:, None] + pos) % n_len[:, None]
            noise = bank[starts[which][:, None] + pos]
        else:
            noise = self._cut_on_host(bank, starts[which], n_len, offset, t, device)
        noise = noise.float() / 32768 * valid
        noise_rir = draws.get(f"{kind}_rir_apply")
        if noise_rir is not None:
            # A point-source noise: reverberated by an impulse response of
            # its own; its power is measured after the reverberation. Only
            # the rows that get both noise and its reverberation.
            noise_rir = noise_rir.cpu() & draws[f"{kind}_apply"].cpu()
        if noise_rir is not None and bool(noise_rir.any()):
            rows = noise_rir.nonzero().squeeze(1)
            rows_d = rows.to(device)
            noisy, _ = self._reverberate(
                noise.index_select(0, rows_d), draws[f"{kind}_rir_which"].cpu()[rows]
            )
            noise = noise.index_copy(0, rows_d, noisy * valid.index_select(0, rows_d))
        n = valid[:, c:].sum(1).clamp(min=1)
        noise_power = (noise[:, c:] * valid[:, c:]).pow(2).sum(1) / n
        snr = snr.to(device)
        scale = (
            10 ** (-snr / 20)
            * speech_power.sqrt()
            / noise_power.clamp(min=1e-10).sqrt()
        )
        return (scale * draws[f"{kind}_apply"].to(device))[:, None] * noise

    def _telephone(self, x, cutoff, highpass, compand, alaw):
        """Pass rows of x (B, T) through a telephone channel.

        In the frequency domain (one FFT of the row): a raised-cosine low-pass
        at ``cutoff`` (B,) Hz and, where ``highpass``, a high-pass at
        telephone_highpass_hz; the band up to 4 kHz is then taken as an 8 kHz
        signal, companded to 8 bits where ``compand`` (A-law where ``alaw``,
        else mu-law; each row scaled to its peak first and back afterwards),
        and resampled back to fs by zero-padding its spectrum. Returns (B, T).
        """
        b, t = x.shape
        device = x.device
        # Zero-pad by 0.1 s, so the circular filtering does not carry the end of
        # the row into its start, to an even number of samples at 8 kHz.
        ratio = self.fs // 8000
        n = -(-(t + self.fs // 10) // (2 * ratio)) * 2 * ratio
        n8 = n // ratio
        spec = torch.fft.rfft(x, n)  # (B, n // 2 + 1)
        freq = torch.fft.rfftfreq(n, 1 / self.fs).to(device)
        w = self.telephone_transition_hz

        def ramp(f):  # 0 below -w/2, 1 above w/2, raised cosine between
            return 0.5 - 0.5 * torch.cos(np.pi * ((f / w) + 0.5).clamp(0, 1))

        gain = ramp(cutoff.to(device)[:, None] - freq)
        hp = ramp(freq - self.telephone_highpass_hz)
        gain = gain * torch.where(highpass.to(device)[:, None], hp, torch.ones_like(hp))
        # the 8 kHz signal: the bins up to 4 kHz, with irfft's 1/n scaling kept
        x8 = torch.fft.irfft(spec[:, : n8 // 2 + 1] * gain[:, : n8 // 2 + 1], n8)
        x8 = x8 / ratio
        if bool(compand.any()):
            peak = x8.abs().amax(1, keepdim=True).clamp(min=1e-10)
            y = x8 / peak
            y = torch.where(alaw.to(device)[:, None], a_law_8bit(y), mu_law_8bit(y))
            x8 = torch.where(compand.to(device)[:, None], y * peak, x8)
        out = torch.fft.rfft(x8, n8) * ratio
        # the 4 kHz bin (the Nyquist bin of the 8 kHz signal) is dropped: the
        # pass band ends below it, and companding noise there has no image
        out[:, -1] = 0
        out = torch.nn.functional.pad(out, (0, n // 2 + 1 - out.size(1)))
        return torch.fft.irfft(out, n)[:, :t]

    def apply(
        self, speech: torch.Tensor, valid: torch.Tensor, draws: Dict
    ) -> torch.Tensor:
        """Apply the choices of ``draws`` to speech (B, T); valid marks the samples.

        This is the deterministic part of forward(): given the same choices it
        gives the same result as CommonPreprocessor.
        """
        b, t = speech.shape
        device = speech.device
        x = speech.float() * valid
        # The powers and the peak are measured on the samples after the leading
        # context: the signal and its mask are cropped first, so the frames of
        # the silence detection start where the window starts.
        c = self.context

        def window_power(signal, mask):
            return self._non_silent_power(signal[:, c:], mask[:, c:])

        power = window_power(x, valid)
        speech_power = power  # the power the SNR refers to

        # The apply flags are on the CPU: testing them does not wait for the GPU.
        if "rir_which" in draws and bool(draws["rir_apply"].any()):
            # Only the rows that get reverberation are convolved: their indices
            # come from the flags on the CPU, so no wait for the GPU.
            rows = draws["rir_apply"].cpu().nonzero().squeeze(1)
            rows_d = rows.to(device)
            early = self.snr_reference == "early"
            y, e = self._reverberate(
                x.index_select(0, rows_d), draws["rir_which"].cpu()[rows], early
            )
            v = valid.index_select(0, rows_d)
            y = y * v
            # Scale back to the power before the reverberation.
            p = power.index_select(0, rows_d)
            gain = (p / window_power(y, v).clamp(min=1e-10)).sqrt()
            x = x.index_copy(0, rows_d, y * gain[:, None])
            if early:
                # The SNR refers to the early reverberation, with the same gain.
                e = e * v * gain[:, None]
                speech_power = speech_power.index_copy(0, rows_d, window_power(e, v))

        for kind in ("noise", "music"):
            if f"{kind}_which" in draws and bool(draws[f"{kind}_apply"].any()):
                snr = draws["snr" if kind == "noise" else f"{kind}_snr"]
                x = x + self._additive(kind, draws, snr, speech_power, valid, t)

        if "telephone_apply" in draws and bool(draws["telephone_apply"].any()):
            rows = draws["telephone_apply"].cpu().nonzero().squeeze(1)
            rows_d = rows.to(device)
            y = self._telephone(
                x.index_select(0, rows_d),
                draws["telephone_cutoff"].cpu()[rows],
                draws["telephone_highpass"].cpu()[rows],
                draws["telephone_compand"].cpu()[rows],
                draws["telephone_alaw"].cpu()[rows],
            )
            x = x.index_copy(0, rows_d, y * valid.index_select(0, rows_d))

        # As the preprocessor: divide by the peak when it exceeds 1.
        peak = (x[:, c:].abs() * valid[:, c:]).amax(1, keepdim=True)
        return torch.where(peak > 1.0, x / peak.clamp(min=1e-30), x)

    def forward(
        self, speech: torch.Tensor, speech_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Add reverberation and noise to speech (B, T) in training mode.

        Returns the augmented speech and speech_lengths. With ``context`` the
        first ``context`` samples of each row are removed afterwards, in
        training and in eval mode.
        """
        augment = self.training and (
            self.noise_apply_prob > 0
            or self.rir_apply_prob > 0
            or self.music_apply_prob > 0
            or self.telephone_apply_prob > 0
        )
        if augment:
            if speech.dim() != 2:
                raise ValueError(f"speech must be (Batch, Time): {tuple(speech.shape)}")
            b, t = speech.shape
            device = speech.device
            if speech_lengths is None:
                lengths = torch.full((b,), t, dtype=torch.long, device=device)
            else:
                lengths = speech_lengths.to(device).long()
            valid = torch.arange(t, device=device) < lengths[:, None]
            if not self.keep_on_device:
                # The noise is cut on the host from offsets that depend on the
                # lengths: one copy of the lengths to the host per batch.
                lengths = lengths.cpu()
            speech = self.apply(speech, valid, self.draw(lengths)).to(speech.dtype)
        return self.remove_context(speech, speech_lengths)

    def remove_context(
        self, speech: torch.Tensor, speech_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Remove the first ``context`` samples of each row (nothing if 0).

        Raises ValueError when the batch has no sample after the context; a row
        whose length does not exceed the context fails an assertion on the
        device (checked without waiting for it).
        """
        if self.context <= 0:
            return speech, speech_lengths
        if speech.size(1) <= self.context:
            raise ValueError(
                f"a row of {speech.size(1)} samples has no samples after the "
                f"context of {self.context}"
            )
        speech = speech[:, self.context :]
        if speech_lengths is not None:
            torch._assert_async(
                (speech_lengths > self.context).all(),
                "a row has no samples after the context",
            )
            speech_lengths = speech_lengths - self.context
        return speech, speech_lengths
