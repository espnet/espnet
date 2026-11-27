"""Supervised seed data: FastMSS meetings re-mixed with noise at a random SNR.

The FastMSS Fisher Shar shards are decoded once into a float16 cache
(:func:`build_fisher_cache`; FLAC decoding from Shar is too slow to feed a
GPU). Per shard ``k`` the cache holds

``speech.<k>.npy``
    The sum of the REVERBERANT speaker images (``spk_reverb``).
``dry.<k>.npy``
    The sum of the dry (anechoic) speaker signals (``spk_dry``),
    sample-aligned with ``speech``.
``noise.<k>.npy``
    The FastMSS noise track (optional).
``index.<k>.jsonl``
    One row per meeting: ``id``, ``offset``, ``length``, ``sup`` (speech
    intervals in seconds) and the active-speech powers ``ps`` (reverberant),
    ``ps_dry`` and the noise power ``pn``.

:class:`FisherRemixDataset` draws each crop from one of two conditions,
reverberant (input ``s_rev + n``, target ``s_rev``) or anechoic (input
``s_dry + n``, target ``s_dry``): the target is always the speech exactly as
it is in the input, so a model regresses the reverberant image when the
speech is reverberant and does not learn dereverberation. The noise is either
the cached FastMSS track or random crops of an external list of noise files
(``noise_list``), split by a stable hash so validation noise is never seen in
training.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import soundfile as sf

from espnet3.components.data.base_dataset import BaseDataset
from espnet3.components.data.dataset_builder import DatasetBuilder as _BaseBuilder
from espnet3.contract import Field
from espnet3.systems.snoar.src.data.noisy import _rng_for

logger = logging.getLogger(__name__)

#: Cache array name -> (Shar field, how its channels are combined).
CACHE_KINDS = {"speech": "spk_reverb", "dry": "spk_dry", "noise": "noise"}
#: Speech condition -> (cache array, index power key).
CONDITIONS = {"reverb": ("speech", "ps"), "dry": ("dry", "ps_dry")}
_POWER_KEY = {"speech": "ps", "dry": "ps_dry", "noise": "pn"}


# ---------------------------------------------------------------------- #
# Cache
# ---------------------------------------------------------------------- #
def list_shar_shards(shar_dir: str) -> List[Dict[str, List[str]]]:
    """Return the Lhotse Shar field files of every shard in ``shar_dir``.

    Raises:
        FileNotFoundError: If there is no ``cuts.*.jsonl.gz`` in ``shar_dir``.
    """
    cuts = sorted(glob.glob(os.path.join(shar_dir, "cuts.*.jsonl.gz")))
    if not cuts:
        raise FileNotFoundError(f"no cuts.*.jsonl.gz in {shar_dir}")
    shards = []
    for c in cuts:
        k = os.path.basename(c).split(".")[1]
        fields = {"cuts": [c]}
        for name in CACHE_KINDS.values():
            found = glob.glob(os.path.join(shar_dir, f"{name}.{k}.*"))
            if found:
                fields[name] = found
        shards.append(fields)
    return shards


def compute_active_mask(sup: Sequence[Sequence[float]], length: int, fs: int):
    """Return a boolean ``(length,)`` mask of the samples inside ``sup``."""
    mask = np.zeros(length, dtype=bool)
    for start, end in sup:
        a, b = max(0, int(start * fs)), min(length, int(np.ceil(end * fs)))
        if b > a:
            mask[a:b] = True
    return mask


def _active_power(x: np.ndarray, mask: np.ndarray) -> float:
    sel = x[mask] if mask.any() else x
    return float(np.mean(sel.astype(np.float64) ** 2))


def _cache_shard(job) -> str:
    """Write the missing arrays of one shard and (re)write its index."""
    import lhotse

    k, fields, out, kinds, fs = job
    out = Path(out)
    index = out / f"index.{k:06d}.jsonl"
    old = {}
    if index.is_file():
        with open(index, encoding="utf-8") as f:
            old = {r["id"]: r for r in map(json.loads, f)}
    todo = [n for n in kinds if not (out / f"{n}.{k:06d}.npy").is_file()]
    if old and not todo:
        return f"shard {k}: complete"
    missing = [n for n in todo if CACHE_KINDS[n] not in fields]
    if missing:
        raise FileNotFoundError(f"shard {k}: Shar has no field for {missing}")
    use = {
        "cuts": fields["cuts"],
        **{CACHE_KINDS[n]: fields[CACHE_KINDS[n]] for n in todo},
    }
    arrays = {n: [] for n in todo}
    rows, off = [], 0
    for cut in lhotse.CutSet.from_shar(fields=use):
        sig = {}
        for n in todo:
            x = cut.load_custom(CACHE_KINDS[n])
            sig[n] = (x[0] if n == "noise" else x.sum(0)).astype(np.float32)
        length = min(len(v) for v in sig.values())
        row = dict(old.get(cut.id, {}))
        if row and row["length"] != length:
            raise ValueError(
                f"shard {k}, {cut.id}: {length} samples, the cached arrays have "
                f"{row['length']}; rebuild the cache from scratch"
            )
        sup = [[round(s.start, 4), round(s.end, 4)] for s in cut.supervisions]
        mask = compute_active_mask(sup, length, fs)
        row.update(id=cut.id, offset=off, length=length, sup=sup)
        for n in todo:
            x = sig[n][:length]
            arrays[n].append(x.astype(np.float16))
            row[_POWER_KEY[n]] = (
                float(np.mean(x.astype(np.float64) ** 2))
                if n == "noise"
                else _active_power(x, mask)
            )
        rows.append(row)
        off += length
    for n, parts in arrays.items():
        tmp = out / f"{n}.{k:06d}.tmp.npy"
        np.save(tmp, np.concatenate(parts))
        os.replace(tmp, out / f"{n}.{k:06d}.npy")
    tmp = out / f"index.{k:06d}.jsonl.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, index)  # index last: marks the shard complete
    return f"shard {k}: {len(rows)} meetings, {off / fs / 3600:.1f} h, wrote {todo}"


def build_fisher_cache(
    shar_dir: str,
    cache_dir: str,
    kinds: Sequence[str] = ("speech", "dry"),
    fs: int = 16000,
    jobs: int = 8,
) -> List[str]:
    """Decode FastMSS Shar shards into the float16 cache.

    Arrays already on disk are kept, so a cache built without ``dry`` gains
    it by rerunning with ``kinds`` including ``dry`` (the index is rewritten
    with the new powers; existing fields are kept).

    Args:
        shar_dir: Directory of ``cuts.<k>.jsonl.gz`` and the field tars.
        cache_dir: Output directory (one file per shard and array).
        kinds: Arrays to write, from :data:`CACHE_KINDS`.
        fs: Sampling rate of the Shar audio.
        jobs: Shards decoded in parallel.

    Returns:
        One status line per shard.

    Raises:
        ValueError: For an unknown kind.
    """
    bad = [n for n in kinds if n not in CACHE_KINDS]
    if bad:
        raise ValueError(f"unknown cache kinds {bad}; expected {list(CACHE_KINDS)}")
    os.makedirs(cache_dir, exist_ok=True)
    shards = list_shar_shards(shar_dir)
    work = [(k, f, cache_dir, tuple(kinds), fs) for k, f in enumerate(shards)]
    with ProcessPoolExecutor(max(1, int(jobs))) as ex:
        status = list(ex.map(_cache_shard, work))
    for line in status:
        logger.info(line)
    return status


# ---------------------------------------------------------------------- #
# Noise files
# ---------------------------------------------------------------------- #
def is_valid_noise(path: str, valid_fraction: float) -> bool:
    """Return whether ``path`` falls in the held-out part (stable hash)."""
    h = int(hashlib.sha1(os.path.basename(path).encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < valid_fraction


def read_noise_list(
    path: str, split: Optional[str] = None, valid_fraction: float = 0.02
) -> List[str]:
    """Read a noise list (one audio path per line) and keep one split.

    Args:
        path: Text file of audio paths.
        split: ``train``, ``valid`` or ``None`` (all files).
        valid_fraction: Share of files held out for ``valid``.

    Raises:
        ValueError: For an unknown split or an empty result.
    """
    if split not in (None, "train", "valid"):
        raise ValueError(f"noise split {split!r}: expected train, valid or null")
    with open(path, encoding="utf-8") as f:
        files = [line.strip() for line in f if line.strip()]
    if split is not None:
        files = [
            p for p in files if is_valid_noise(p, valid_fraction) == (split == "valid")
        ]
    if not files:
        raise ValueError(f"no noise file left in {path} for split {split!r}")
    return files


# ---------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------- #
class FisherRemixDataset(BaseDataset):
    """Random crops of FastMSS meetings with speech and noise references.

    Args:
        cache_dir: Directory of the float16 cache.
        shards: Shard indices to read (e.g. ``[0, ..., 9]`` train, ``[10]``
            validation).
        segment_seconds: Crop length.
        fs: Sampling rate of the cache.
        conditions: Probability of each speech condition, e.g.
            ``{reverb: 0.5, dry: 0.5}`` (see :data:`CONDITIONS`).
        noise_list: Text file of noise audio paths. ``None`` uses the cached
            FastMSS noise track.
        noise_split: ``train`` / ``valid`` part of ``noise_list`` (stable
            hash), or ``None`` for all files.
        noise_valid_fraction: Share of ``noise_list`` held out as ``valid``.
        snr_range: ``[low, high]`` SNR in dB, relative to the meeting's
            active-speech power.
        gain_range: ``[low, high]`` crop gain in dB.
        min_active: Least fraction of the crop inside a speech interval; the
            crop is redrawn (up to 20 times) below it, so the SI-SDR target
            is not silence.
        num_items: Items per epoch.
        deterministic: Fixed crops per index (validation).

    Raises:
        FileNotFoundError: If a shard or array of the cache is missing.
        ValueError: For an unknown condition or no usable meeting.
    """

    fields = (
        Field("speech_mix", "audio"),
        Field("speech_ref", "audio"),
        Field("noise_ref", "audio"),
    )

    def __init__(
        self,
        cache_dir: str,
        shards: Sequence[int],
        segment_seconds: float = 3.0,
        fs: int = 16000,
        conditions: Optional[Mapping[str, float]] = None,
        noise_list: Optional[str] = None,
        noise_split: Optional[str] = None,
        noise_valid_fraction: float = 0.02,
        snr_range: Sequence[float] = (-5.0, 20.0),
        gain_range: Sequence[float] = (-10.0, 6.0),
        min_active: float = 0.25,
        num_items: int = 20000,
        deterministic: bool = False,
    ) -> None:
        """Index the cache (the arrays are memory-mapped lazily)."""
        self.cache = Path(cache_dir)
        self.shards = [int(k) for k in shards]
        self.fs = int(fs)
        self.seg = int(round(segment_seconds * fs))
        conditions = dict(conditions or {"reverb": 1.0})
        bad = [c for c in conditions if c not in CONDITIONS]
        if bad or not conditions or min(conditions.values()) < 0:
            raise ValueError(
                f"conditions {conditions}: expected non-negative weights for "
                f"{list(CONDITIONS)}"
            )
        total = float(sum(conditions.values()))
        self.conditions = list(conditions)
        self.cond_p = np.array([conditions[c] / total for c in self.conditions])
        self.noise_files = (
            read_noise_list(noise_list, noise_split, noise_valid_fraction)
            if noise_list
            else None
        )
        self.snr_range = tuple(float(v) for v in snr_range)
        self.gain_range = tuple(float(v) for v in gain_range)
        self.min_active = float(min_active)
        self.num_items = int(num_items)
        self.deterministic = deterministic
        needed = {CONDITIONS[c][0] for c in self.conditions}
        if self.noise_files is None:
            needed.add("noise")
        self.meetings: List[Dict[str, Any]] = []
        for k in self.shards:
            index = self.cache / f"index.{k:06d}.jsonl"
            for name in sorted(needed):
                if (
                    not (self.cache / f"{name}.{k:06d}.npy").is_file()
                    or not index.is_file()
                ):
                    raise FileNotFoundError(
                        f"{self.cache}: shard {k} has no {name!r} array; build it "
                        "with build_fisher_cache(shar_dir, cache_dir, kinds=...)"
                    )
            with open(index, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    if row["length"] >= self.seg:
                        row["shard"] = k
                        self.meetings.append(row)
        if not self.meetings:
            raise ValueError(f"no meeting in {cache_dir} is {segment_seconds} s long")
        self._arrays: Dict[tuple, np.ndarray] = {}
        self._masks: Dict[str, np.ndarray] = {}

    def _array(self, name: str, k: int) -> np.ndarray:
        if (name, k) not in self._arrays:
            self._arrays[name, k] = np.load(
                self.cache / f"{name}.{k:06d}.npy", mmap_mode="r"
            )
        return self._arrays[name, k]

    def _mask(self, m: Dict[str, Any]) -> np.ndarray:
        if m["id"] not in self._masks:
            self._masks[m["id"]] = compute_active_mask(
                m.get("sup", [[0.0, m["length"] / self.fs]]), m["length"], self.fs
            )
        return self._masks[m["id"]]

    def _draw_crop(self, rng):
        for _ in range(20):
            m = self.meetings[int(rng.integers(len(self.meetings)))]
            st = int(rng.integers(0, m["length"] - self.seg + 1))
            if self._mask(m)[st : st + self.seg].mean() >= self.min_active:
                break
        return m, st

    def _noise(self, m, st, rng) -> np.ndarray:
        if self.noise_files is None:
            arr = self._array("noise", m["shard"])
            o = m["offset"] + st
            return np.asarray(arr[o : o + self.seg], dtype=np.float32)
        n = None
        for _ in range(5):
            path = self.noise_files[int(rng.integers(len(self.noise_files)))]
            with sf.SoundFile(path) as f:
                if f.samplerate != self.fs:
                    raise ValueError(f"{path} is {f.samplerate} Hz, expected {self.fs}")
                frames = f.frames
                if frames > self.seg:
                    f.seek(int(rng.integers(0, frames - self.seg + 1)))
                x = f.read(min(frames, self.seg), dtype="float32", always_2d=True)[:, 0]
            if len(x) < self.seg:
                x = np.tile(x, int(np.ceil(self.seg / max(len(x), 1))))[: self.seg]
            n = x
            if np.mean(n.astype(np.float64) ** 2) > 1e-10:
                break
        return n

    def __len__(self) -> int:
        """Return the number of items per epoch."""
        return self.num_items

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one re-mixed crop with its references."""
        rng = _rng_for(idx, self.deterministic)
        cond = self.conditions[int(rng.choice(len(self.conditions), p=self.cond_p))]
        name, power_key = CONDITIONS[cond]
        m, st = self._draw_crop(rng)
        o = m["offset"] + st
        s = np.asarray(
            self._array(name, m["shard"])[o : o + self.seg], dtype=np.float32
        )
        n = self._noise(m, st, rng)
        pn = max(float(np.mean(n.astype(np.float64) ** 2)), 1e-12)
        snr = rng.uniform(*self.snr_range)
        n = n * np.float32(np.sqrt(m[power_key] / (pn * 10 ** (snr / 10))))
        gain = np.float32(10 ** (rng.uniform(*self.gain_range) / 20))
        s, n = s * gain, n * gain
        return {"speech_mix": s + n, "speech_ref": s, "noise_ref": n}


class FisherCacheBuilder(_BaseBuilder):
    """Build the float16 FastMSS cache when ``shar_dir`` is given."""

    def is_source_prepared(self, cache_dir: Optional[str] = None, **kwargs) -> bool:
        """Return whether the cache directory holds an index."""
        return cache_dir is not None and any(Path(cache_dir).glob("index.*.jsonl"))

    def prepare_source(
        self, cache_dir: Optional[str] = None, shar_dir: Optional[str] = None, **kwargs
    ) -> None:
        """Build the cache from ``shar_dir``.

        Raises:
            FileNotFoundError: If ``shar_dir`` is not given.
        """
        if not shar_dir:
            raise FileNotFoundError(
                f"FastMSS cache not found at {cache_dir}: set shar_dir to the "
                "FastMSS Shar directory to build it"
            )
        build_fisher_cache(
            shar_dir,
            cache_dir,
            kinds=kwargs.get("cache_kinds", ("speech", "dry")),
            jobs=int(kwargs.get("jobs", 8)),
        )

    def is_built(self, **kwargs) -> bool:
        """Return True: the cache is the built form."""
        return True

    def build(self, **kwargs) -> None:
        """Do nothing."""


Dataset = FisherRemixDataset
DatasetBuilder = FisherCacheBuilder


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Build the cache from the command line.

    ``python -m espnet3.systems.snoar.src.data.fisher <shar_dir> <cache_dir>
    [--kinds speech dry] [--jobs 8]``
    """
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("shar_dir")
    ap.add_argument("cache_dir")
    ap.add_argument("--kinds", nargs="+", default=["speech", "dry"])
    ap.add_argument("--fs", type=int, default=16000)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(argv)
    for line in build_fisher_cache(a.shar_dir, a.cache_dir, a.kinds, a.fs, a.jobs):
        print(line, flush=True)


if __name__ == "__main__":
    main()
