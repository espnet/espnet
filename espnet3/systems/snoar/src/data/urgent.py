"""Supervised seed data: URGENT 2025 speech + noise (+ RIRs), mixed on the fly.

The URGENT 2025 training pool is read straight from an OMNIIO audio store
(one ``metadata.parquet`` per subset; each row is one encoded file at a byte
range of a blob, whose ``path`` may be in a shared store such as
``OMNIIO/audio/dns5_speech``), so nothing is extracted to per-file audio:

.. code-block:: text

    <root>/speech_{commonvoice,dns5,ears,libritts,mls,vctk}/
    <root>/noise_{dns5,fsd50k,fma_novocal,wham,wind}/
    <root>/rir_dns5/

Each crop is a random speech segment, convolved with a random RIR with
probability ``reverb_prob``, plus a random noise segment at an SNR drawn from
``snr_range``. Everything is resampled to ``fs`` (the store mixes 16-48 kHz)
and multichannel files contribute channel 0. The speech target is the speech
exactly as it is in the input (reverberant or dry), so a model trained on it
denoises but does not dereverberate, which is what mixture consistency
(speech + noise = mixture) asks for. Files are split into train/valid by a
stable hash of their id.
"""

from __future__ import annotations

import hashlib
import io
import os
from math import gcd
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

from espnet3.components.data.base_dataset import BaseDataset
from espnet3.components.data.dataset_builder import DatasetBuilder as _BaseBuilder
from espnet3.contract import Field
from espnet3.systems.snoar.src.data.noisy import _rng_for

SPEECH_SUBSETS = (
    "speech_commonvoice",
    "speech_dns5",
    "speech_ears",
    "speech_libritts",
    "speech_mls",
    "speech_vctk",
)
NOISE_SUBSETS = (
    "noise_dns5",
    "noise_fsd50k",
    "noise_fma_novocal",
    "noise_wham",
    "noise_wind",
)
RIR_SUBSETS = ("rir_dns5",)
_COLUMNS = ["id", "sample_rate", "samples", "start_byte", "end_byte", "path"]


def is_valid_id(key: str, valid_fraction: float) -> bool:
    """Return whether ``key`` falls in the held-out part (stable hash)."""
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < valid_fraction


def read_allow_list(path: Optional[str]) -> Optional[set]:
    """Read ``<subset>/<id>`` lines (e.g. a license allow-list), or ``None``."""
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def read_file_list(path: Optional[str]) -> List[str]:
    """Read one audio path per line (empty for ``None``)."""
    if not path:
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


class _Pool:
    """Compact index of the files of some OMNIIO subsets (one split).

    ``allow`` (a set of ``<subset>/<id>`` keys) restricts the subsets named in
    ``allow_subsets``; ``files`` adds plain audio files (key ``file/<path>``).
    """

    def __init__(
        self,
        root: Path,
        subsets: Sequence[str],
        split: Optional[str],
        valid_fraction: float,
        min_samples_at_16k: int = 0,
        allow: Optional[set] = None,
        allow_subsets: Sequence[str] = (),
        files: Sequence[str] = (),
    ) -> None:
        import pyarrow.parquet as pq

        self.blobs: List[str] = []
        blob_id: Dict[str, int] = {}
        cols = {k: [] for k in ("sr", "n", "s0", "s1", "bin")}

        def add(key, sr, n, s0, s1, path):
            if split is not None and is_valid_id(key, valid_fraction) != (
                split == "valid"
            ):
                return
            if n * 16000 < min_samples_at_16k * sr:
                return
            if path not in blob_id:
                blob_id[path] = len(self.blobs)
                self.blobs.append(path)
            for c, v in zip(cols, (sr, n, s0, s1, blob_id[path])):
                cols[c].append(v)

        for sub in subsets:
            restrict = allow is not None and sub in allow_subsets
            t = pq.read_table(
                root / sub / "metadata.parquet", columns=_COLUMNS
            ).to_pydict()
            for key, sr, n, s0, s1, path in zip(*(t[c] for c in _COLUMNS)):
                if restrict and f"{sub}/{key}" not in allow:
                    continue
                add(f"{sub}/{key}", sr, n, s0, s1, path)
        for path in files:
            info = sf.info(path)
            add(
                f"file/{os.path.basename(path)}",
                info.samplerate,
                info.frames,
                0,
                os.path.getsize(path),
                path,
            )
        if not cols["sr"]:
            raise ValueError(f"no file of {list(subsets)} in {root} (split {split})")
        missing = [b for b in self.blobs if not os.path.isfile(b)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} blobs missing, e.g. {missing[0]}")
        self.sr = np.asarray(cols["sr"], dtype=np.int32)
        self.n = np.asarray(cols["n"], dtype=np.int64)
        self.s0 = np.asarray(cols["s0"], dtype=np.int64)
        self.s1 = np.asarray(cols["s1"], dtype=np.int64)
        self.blob = np.asarray(cols["bin"], dtype=np.int32)

    def __len__(self) -> int:
        return len(self.sr)

    def read(
        self,
        i: int,
        fs: int,
        length: Optional[int],
        rng: Optional[np.random.Generator],
    ) -> np.ndarray:
        """Read file ``i`` (a random ``length``-sample window at ``fs``)."""
        sr = int(self.sr[i])
        with open(self.blobs[self.blob[i]], "rb") as f:
            f.seek(int(self.s0[i]))
            data = f.read(int(self.s1[i] - self.s0[i]))
        with sf.SoundFile(io.BytesIO(data)) as snd:
            frames = snd.frames
            want = frames if length is None else int(np.ceil(length * sr / fs)) + 16
            if rng is not None and frames > want:
                snd.seek(int(rng.integers(0, frames - want + 1)))
            x = snd.read(min(frames, want), dtype="float32", always_2d=True)[:, 0]
        if sr != fs:
            g = gcd(fs, sr)
            x = resample_poly(x, fs // g, sr // g).astype(np.float32)
        return x if length is None else x[:length]


def _power(x: np.ndarray) -> float:
    return float(np.mean(np.square(x, dtype=np.float64)))


class UrgentMixDataset(BaseDataset):
    """Random crops of URGENT 2025 speech + noise with both references.

    Args:
        root: OMNIIO directory of the URGENT 2025 pool.
        split: ``train`` / ``valid`` (stable hash per file) or ``None``.
        valid_fraction: Share of the files held out as ``valid``.
        speech_subsets / noise_subsets / rir_subsets: Subset directories.
        segment_seconds: Crop length.
        fs: Output sampling rate.
        reverb_prob: Probability that the speech is convolved with an RIR.
        snr_range: ``[low, high]`` SNR (dB) of the speech over the noise.
        gain_range: ``[low, high]`` mixture gain in dB (peak kept < 1).
        min_speech_seconds: Shortest speech file used.
        noise_allow_list: Text file of ``<subset>/<id>`` noise files that may
            be used (e.g. a commercial-license allow-list); it restricts the
            subsets in ``noise_allow_subsets`` and leaves the others whole.
        noise_allow_subsets: Noise subsets the allow-list applies to.
        extra_noise_list: Text file of extra noise audio paths (any rate).
        num_items: Items per epoch.
        deterministic: Fixed crops per index (validation).
    """

    fields = (
        Field("speech_mix", "audio"),
        Field("speech_ref", "audio"),
        Field("noise_ref", "audio"),
    )

    def __init__(
        self,
        root: str,
        split: Optional[str] = "train",
        valid_fraction: float = 0.005,
        speech_subsets: Sequence[str] = SPEECH_SUBSETS,
        noise_subsets: Sequence[str] = NOISE_SUBSETS,
        rir_subsets: Sequence[str] = RIR_SUBSETS,
        segment_seconds: float = 3.0,
        fs: int = 16000,
        reverb_prob: float = 0.5,
        snr_range: Sequence[float] = (-5.0, 20.0),
        gain_range: Sequence[float] = (-10.0, 6.0),
        min_speech_seconds: float = 1.0,
        noise_allow_list: Optional[str] = None,
        noise_allow_subsets: Sequence[str] = (),
        extra_noise_list: Optional[str] = None,
        num_items: int = 20000,
        deterministic: bool = False,
    ) -> None:
        """Index the parquet metadata of every subset."""
        if split not in (None, "train", "valid"):
            raise ValueError(f"split {split!r}: expected train, valid or null")
        root = Path(root)
        self.fs = int(fs)
        self.seg = int(round(segment_seconds * fs))
        self.speech = _Pool(
            root,
            speech_subsets,
            split,
            valid_fraction,
            int(min_speech_seconds * 16000),
        )
        self.noise = _Pool(
            root,
            noise_subsets,
            split,
            valid_fraction,
            allow=read_allow_list(noise_allow_list),
            allow_subsets=noise_allow_subsets,
            files=read_file_list(extra_noise_list),
        )
        self.rir = (
            _Pool(root, rir_subsets, split, valid_fraction) if reverb_prob > 0 else None
        )
        self.reverb_prob = float(reverb_prob)
        self.snr_range = tuple(float(v) for v in snr_range)
        self.gain_range = tuple(float(v) for v in gain_range)
        self.num_items = int(num_items)
        self.deterministic = deterministic

    def __len__(self) -> int:
        """Return the number of items per epoch."""
        return self.num_items

    def _speech(self, rng) -> np.ndarray:
        for _ in range(10):
            s = self.speech.read(
                int(rng.integers(len(self.speech))), self.fs, self.seg, rng
            )
            if _power(s) > 1e-8:
                break
        out = np.zeros(self.seg, dtype=np.float32)
        off = int(rng.integers(0, self.seg - len(s) + 1)) if len(s) < self.seg else 0
        out[off : off + len(s)] = s
        return out

    def _reverb(self, s: np.ndarray, rng) -> np.ndarray:
        h = self.rir.read(int(rng.integers(len(self.rir))), self.fs, None, None)
        h = h / max(float(np.sqrt(np.sum(h.astype(np.float64) ** 2))), 1e-8)
        return fftconvolve(s, h)[: self.seg].astype(np.float32)

    def _noise(self, rng) -> np.ndarray:
        for _ in range(10):
            n = self.noise.read(
                int(rng.integers(len(self.noise))), self.fs, self.seg, rng
            )
            if _power(n) > 1e-10:
                break
        if len(n) < self.seg:
            n = np.tile(n, int(np.ceil(self.seg / max(len(n), 1))))[: self.seg]
        return n.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one mixed crop with its speech and noise references."""
        rng = _rng_for(idx, self.deterministic)
        s = self._speech(rng)
        if self.rir is not None and rng.random() < self.reverb_prob:
            s = self._reverb(s, rng)
        n = self._noise(rng)
        snr = rng.uniform(*self.snr_range)
        n = n * np.float32(
            np.sqrt(_power(s) / (max(_power(n), 1e-12) * 10 ** (snr / 10)))
        )
        gain = 10 ** (rng.uniform(*self.gain_range) / 20)
        peak = float(np.max(np.abs(s + n))) * gain
        if peak > 0.99:
            gain *= 0.99 / peak
        s, n = s * np.float32(gain), n * np.float32(gain)
        return {"speech_mix": s + n, "speech_ref": s, "noise_ref": n}


class UrgentBuilder(_BaseBuilder):
    """Nothing to build: the OMNIIO store is read in place."""

    def is_source_prepared(self, root: Optional[str] = None, **kwargs) -> bool:
        """Return whether ``root`` holds the subsets' metadata."""
        return root is not None and any(Path(root).glob("*/metadata.parquet"))

    def prepare_source(self, root: Optional[str] = None, **kwargs) -> None:
        """Raise: the store has to exist.

        Raises:
            FileNotFoundError: Always (the source is not prepared).
        """
        raise FileNotFoundError(f"no URGENT 2025 OMNIIO store at {root}")

    def is_built(self, **kwargs) -> bool:
        """Return True: the store is the built form."""
        return True

    def build(self, **kwargs) -> None:
        """Do nothing."""


Dataset = UrgentMixDataset
DatasetBuilder = UrgentBuilder
