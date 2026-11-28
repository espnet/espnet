"""Datasets of noisy recordings: training crops and full inference utterances.

Select them from a config with the module-path form of ``data_src``:

.. code-block:: yaml

    dataset:
      train:
        - data_src: espnet3.systems.snoar.src.data.noisy
          data_src_args:
            manifest: ${data_dir}/manifest/train.jsonl
            noise_pool: ${data_dir}/manifest/noise_train.jsonl  # dnf / mixit
            segment_seconds: 3.0
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from espnet3.components.data.base_dataset import BaseDataset
from espnet3.components.data.dataset_builder import DatasetBuilder as _BaseBuilder
from espnet3.contract import Field
from espnet3.systems.snoar.src.data.manifest import (
    crossfade_concat,
    fit_length,
    read_audio,
    read_manifest,
)


def _rng_for(idx: int, deterministic: bool) -> np.random.Generator:
    """Seed per item: fixed for validation crops, per worker for training."""
    if deterministic:
        return np.random.default_rng(int(idx))
    info = torch.utils.data.get_worker_info()
    base = info.seed if info is not None else torch.initial_seed()
    return np.random.default_rng([int(base) % (2**32), int(idx)])


class NoisyCropDataset(BaseDataset):
    """Fixed-length crops of noisy recordings, with optional noise-only crops.

    Each item holds ``speech_mix`` (the noisy crop ``x1``). With
    ``noise_pool`` set it also holds ``noise2``, a noise-only crop for DNF
    and MixIT, built by concatenating pool regions with 10 ms crossfades.
    With ``with_refs`` it holds ``speech_ref`` cropped like ``speech_mix``,
    for the validation diagnostic only.

    Args:
        manifest: Recording manifest (JSONL).
        segment_seconds: Crop length.
        fs: Expected sampling rate; files at another rate raise.
        noise_pool: Noise-pool manifest (JSONL), or ``None``.
        noise_level: ``native`` keeps the pool's own level (right for gaps
            harvested from the same recordings); ``snr`` rescales ``noise2``
            to ``noise_snr`` dB below ``speech_mix`` (for external corpora).
        noise_snr: ``[low, high]`` dB range for ``noise_level: snr``.
        deterministic: Fixed crops per index (validation).
        with_refs: Also return ``speech_ref``.
        num_items: Items per epoch; defaults to the number of recordings.

    Raises:
        ValueError: For an unknown ``noise_level`` or a recording manifest
            without references when ``with_refs`` is set.
    """

    def __init__(
        self,
        manifest: str,
        segment_seconds: float = 3.0,
        fs: int = 16000,
        noise_pool: Optional[str] = None,
        noise_level: str = "native",
        noise_snr: Sequence[float] = (0.0, 10.0),
        deterministic: bool = False,
        with_refs: bool = False,
        num_items: Optional[int] = None,
    ) -> None:
        """Read the manifests."""
        if noise_level not in ("native", "snr"):
            raise ValueError(f"noise_level {noise_level!r} not in native|snr")
        self.rows = read_manifest(manifest)
        self.fs = int(fs)
        self.seg = int(round(segment_seconds * fs))
        self.pool = read_manifest(noise_pool) if noise_pool else None
        self.noise_level = noise_level
        self.noise_snr = tuple(float(v) for v in noise_snr)
        self.deterministic = deterministic
        self.with_refs = with_refs
        self.num_items = int(num_items) if num_items else len(self.rows)
        if with_refs and "speech_ref" not in self.rows[0]:
            raise ValueError(f"with_refs is set but {manifest} has no `speech_ref`")
        fields = [Field("speech_mix", "audio")]
        if self.pool is not None:
            fields.append(Field("noise2", "audio"))
        if with_refs:
            fields.append(Field("speech_ref", "audio"))
        self.fields = tuple(fields)

    def __len__(self) -> int:
        """Return the number of items per epoch."""
        return self.num_items

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one crop."""
        rng = _rng_for(idx, self.deterministic)
        row = self.rows[int(idx) % len(self.rows)]
        n = int(row.get("num_samples") or 0)
        start = int(rng.integers(0, n - self.seg + 1)) if n > self.seg else 0
        item = {
            "speech_mix": fit_length(
                read_audio(row["audio"], start, start + self.seg, self.fs), self.seg
            )
        }
        if self.with_refs:
            item["speech_ref"] = fit_length(
                read_audio(row["speech_ref"], start, start + self.seg, self.fs),
                self.seg,
            )
        if self.pool is not None:
            item["noise2"] = self._noise(rng, item["speech_mix"])
        return item

    def _noise(self, rng: np.random.Generator, x1: np.ndarray) -> np.ndarray:
        pieces, total = [], 0
        for _ in range(64):  # bounded: a pool of tiny regions cannot loop forever
            r = self.pool[int(rng.integers(len(self.pool)))]
            s0, s1 = int(r.get("start", 0)), r.get("end")
            s1 = int(s1) if s1 not in (None, -1) else None
            span = (s1 - s0) if s1 is not None else None
            need = self.seg - total
            if span is not None and span > need:
                s0 = s0 + int(rng.integers(0, span - need + 1))
                s1 = s0 + need
            elif span is None:
                s1 = None
            piece = read_audio(r["audio"], s0, s1, self.fs)
            if span is None and len(piece) > need:
                off = int(rng.integers(0, len(piece) - need + 1))
                piece = piece[off : off + need]
            pieces.append(piece)
            total += len(piece)
            if total >= self.seg:
                break
        n2 = crossfade_concat(pieces, self.seg, fade=self.fs // 100)
        if self.noise_level == "snr":
            snr = rng.uniform(*self.noise_snr)
            e1 = float(np.sum(x1.astype(np.float64) ** 2)) + 1e-8
            e2 = float(np.sum(n2.astype(np.float64) ** 2)) + 1e-8
            n2 = n2 * np.float32(np.sqrt(e1 / (e2 * 10 ** (snr / 10))))
        return n2.astype(np.float32)


class UtteranceDataset(BaseDataset):
    """Whole recordings for the ``infer`` stage.

    Items hold ``speech`` (the noisy recording) and, when the manifest has
    one, ``speech_ref``: the path of the clean reference, which only the
    metrics read (``dataset:speech_ref``). ``utt_id`` names the output file.

    Args:
        manifest: Recording manifest (JSONL).
        fs: Expected sampling rate.
    """

    def __init__(self, manifest: str, fs: int = 16000) -> None:
        """Read the manifest."""
        self.rows = read_manifest(manifest)
        self.fs = int(fs)
        fields = [Field("utt_id", "text"), Field("speech", "audio")]
        if "speech_ref" in self.rows[0]:
            fields.append(Field("speech_ref", "text"))
        self.fields = tuple(fields)

    def __len__(self) -> int:
        """Return the number of recordings."""
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one whole recording."""
        row = self.rows[int(idx)]
        item = {
            "utt_id": str(row.get("id", idx)),
            "speech": read_audio(row["audio"], fs=self.fs),
        }
        if "speech_ref" in row:
            item["speech_ref"] = str(row["speech_ref"])
        return item


class ManifestBuilder(_BaseBuilder):
    """Builder for prebuilt manifests: checks they exist, builds nothing.

    The CLI and recipes write the manifests themselves (``espnet snoar
    adapt`` from an audio directory, a recipe's own ``dataset/builder.py``
    from its corpus), so this builder only reports a missing manifest.
    """

    def is_source_prepared(self, manifest: Optional[str] = None, **kwargs) -> bool:
        """Return whether ``manifest`` exists."""
        from pathlib import Path

        return manifest is None or Path(manifest).is_file()

    def prepare_source(self, manifest: Optional[str] = None, **kwargs) -> None:
        """Raise: a missing manifest has to be written by the caller."""
        raise FileNotFoundError(
            f"manifest {manifest} is missing; write it with "
            "`espnet snoar manifest <audio_dir> -o <manifest.jsonl>`"
        )

    def is_built(self, **kwargs) -> bool:
        """Return True: there is nothing to build."""
        return True

    def build(self, **kwargs) -> None:
        """Do nothing."""


Dataset = NoisyCropDataset
DatasetBuilder = ManifestBuilder
