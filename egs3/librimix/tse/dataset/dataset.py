"""LibriMix TSE dataset implementation.

This module provides :class:`LibriMixTSEDataset`, a Torch-style dataset that
reads the simulated LibriMix directory structure produced by
:class:`egs3.librimix.tse.dataset.DatasetBuilder`.

The dataset resolves its data directory from ``<recipe_dir>/download/`` by
default, or from the ``data_dir`` argument when provided.  If the data has
not yet been simulated, :class:`~egs3.librimix.tse.dataset.builder.LibriMixTSEBuilder`
is invoked automatically to build it first.

What this dataset returns
-------------------------
Each :meth:`LibriMixTSEDataset.__getitem__` call returns a plain dict:

- ``num_spk``: ``np.ndarray`` of shape ``(1,)`` — number of speakers
- ``speech_mix``: ``np.float32`` waveform array (mixture signal)
- ``enroll_ref1``, ``enroll_ref2`` [, ``enroll_ref3``]: enrollment waveforms
- ``speech_ref1``, ``speech_ref2`` [, ``speech_ref3``]: clean source waveforms
- ``text_spk1``, ``text_spk2`` [, ``text_spk3``]: transcript strings

Keys listed in ``ignore_key_prefix`` are omitted from the returned dict.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from dataclasses import dataclass, fields
from importlib import resources
from pathlib import Path
from typing import Dict, List

import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from egs3.librimix.tse.dataset.builder import (
    LibriMixTSEBuilder,
    resolve_librimix_root,
)
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]

_KNOWN_SPLITS = {str(split) for split in _DATASET_CFG["supported_splits"]}


@dataclass(frozen=True)
class Libri2MixTSEExample:
    """Internal index entry for a 2-speaker LibriMix utterance."""

    utt_id: str
    speech_mix: str
    enroll_ref1: str
    enroll_ref2: str
    speech_ref1: str
    speech_ref2: str
    text_spk1: str
    text_spk2: str
    num_spk: int = 2


@dataclass(frozen=True)
class Libri3MixTSEExample:
    """Internal index entry for a 3-speaker LibriMix utterance."""

    utt_id: str
    speech_mix: str
    enroll_ref1: str
    enroll_ref2: str
    enroll_ref3: str
    speech_ref1: str
    speech_ref2: str
    speech_ref3: str
    text_spk1: str
    text_spk2: str
    text_spk3: str
    num_spk: int = 3


def _check_missing_files(folder: Path, split: str, num_spk: int) -> bool:
    """Check that basic data files exist; return True if enrollment map exists."""
    split_dir = folder / split
    required = ["wav.scp", "spk1.scp", "spk2.scp", "text_spk1", "text_spk2"]
    if num_spk == 3:
        required += ["spk3.scp", "text_spk3"]
    for fname in required:
        if not (split_dir / fname).exists():
            raise FileNotFoundError(
                f"Missing file: {split_dir / fname}. "
                "Please run create_dataset before loading the dataset."
            )
    if "train" in split:
        return True
    return (folder / f"{split}/mixture2enrollment").exists()


def _download_missing_files(folder: Path, split: str) -> None:
    """Download the speakerbeam enrollment map for dev/test splits."""
    if "dev" in split:
        download_url(
            "https://raw.githubusercontent.com/BUTSpeechFIT/speakerbeam/"
            "main/egs/libri2mix/data/wav8k/min/dev/map_mixture2enrollment",
            folder / f"{split}/mixture2enrollment",
        )
    elif "test" in split:
        download_url(
            "https://raw.githubusercontent.com/BUTSpeechFIT/speakerbeam/"
            "main/egs/libri2mix/data/wav8k/min/test/map_mixture2enrollment",
            folder / f"{split}/mixture2enrollment",
        )


class LibriMixTSEDataset(TorchDataset):
    """Torch dataset that reads simulated LibriMix data from disk.

    Args:
        split: LibriMix split name, e.g. ``"2mix_16k_max_train_mix-both"``.
            Format: ``{num_spk}mix_{fs}_{mode}_{dset}_{mix_type}``.
        recipe_dir: Optional recipe root directory.  When ``None``, defaults to
            ``<this_file>/../`` (i.e. the recipe root inferred from the module
            path).
        data_dir: Optional explicit path to the LibriMix dataset root (the
            directory that contains the split sub-directories and
            ``LibriMix/``).  When ``None``, ``<recipe_dir>/download/`` is used.
        ignore_key_prefix: List of key prefixes to omit from returned dicts.
            Supported keys: ``speech_mix``, ``enroll_ref{N}``,
            ``speech_ref{N}``, ``text_spk{N}``, ``utt_id``, ``num_spk``.

    Raises:
        ValueError: If ``split`` is not listed in ``config.yaml``.
        FileNotFoundError: If the data directory or required split files are
            missing.
        RuntimeError: If no utterance pairs are found for the split.

    Examples:
        >>> dataset = LibriMixTSEDataset(split="2mix_16k_max_dev_mix-clean")
        >>> sample = dataset[0]
        >>> sorted(sample.keys())
        ['enroll_ref1', 'enroll_ref2', 'num_spk', 'speech_mix', 'speech_ref1', 'speech_ref2']
    """

    def __init__(
        self,
        split: str = "2mix_16k_max_train_mix-both",
        recipe_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
        ignore_key_prefix: List[str] | None = None,
    ) -> None:
        self.split = str(split)
        if self.split not in _KNOWN_SPLITS:
            known = ", ".join(sorted(_KNOWN_SPLITS))
            raise ValueError(f"Unknown split '{self.split}'. Expected one of: {known}")

        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )

        # Resolve the LibriMix dataset root
        if data_dir is not None:
            self.librimix_root = resolve_librimix_root(
                Path(data_dir).resolve(), self.split
            )
        else:
            self.librimix_root = resolve_librimix_root(
                Path(recipe_root) / _CONFIG["builder"]["dataset_path"], self.split
            )

        # Build if not already done
        builder = LibriMixTSEBuilder()
        if not builder.is_built(recipe_dir=recipe_root):
            builder.build(recipe_dir=recipe_root)

        # Parse split parameters: {num_spk}mix_{fs}_{mode}_{dset}_{mix_type}
        # e.g. "2mix_16k_max_train_mix-both" → num_spk=2, fs=16k, mode=max,
        #       dset=train, mix_type=mix_both
        num_spk_str, fs, mode, dset, mix_type = self.split.split("_", 4)
        mix_type_us = mix_type.replace("-", "_")  # mix-both → mix_both
        self.num_spk = int(num_spk_str.split("mix")[0])
        self.partition = f"{dset}/{mix_type_us}"

        # Directory containing the raw WAV files
        self.split_dir = (
            self.librimix_root
            / f"LibriMix/libri_mix/Libri{self.num_spk}Mix/wav{fs}/{mode}"
        )
        if not self.split_dir.is_dir():
            raise FileNotFoundError(
                f"Split audio directory not found: {self.split_dir}. "
                "Please run the dataset build step first."
            )

        self.ignore_key_prefix = tuple(ignore_key_prefix) if ignore_key_prefix else ()

        # Load or download the enrollment map (dev/test only)
        enrollment_map_path = self.librimix_root / f"{self.split}/mixture2enrollment"
        has_map = _check_missing_files(self.librimix_root, self.split, self.num_spk)
        if has_map:
            self.enrollment_map = self._load_enrollment_map(
                enrollment_map_path, self.split
            )
        elif os.access(self.librimix_root, os.W_OK):
            _download_missing_files(self.librimix_root, self.split)
            self.enrollment_map = self._load_enrollment_map(
                enrollment_map_path, self.split
            )
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                _download_missing_files(Path(tmpdir), self.split)
                self.enrollment_map = self._load_enrollment_map(
                    Path(tmpdir), self.split
                )

        self._examples = self._parse_dataset()
        print(f"len(LibriMixTSEDataset[{self.split}])={len(self)}", flush=True)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Dict:
        """Load one item by index.

        Returns:
            dict with keys (filtered by ``ignore_key_prefix``):

            - ``num_spk``: ``np.ndarray`` shape ``(1,)``
            - ``speech_mix``: ``np.float32`` waveform
            - ``enroll_ref{N}``: ``np.float32`` enrollment waveform
            - ``speech_ref{N}``: ``np.float32`` clean source waveform
            - ``text_spk{N}``: transcript string
        """
        ex = self._examples[int(idx)]
        keys = [f.name for f in fields(ex)]
        ret = {}
        srs = []
        for k in keys:
            if k.startswith(self.ignore_key_prefix):
                continue
            if k.startswith(("speech_mix", "speech_ref")):
                audio, _sr = sf.read(str(getattr(ex, k)), dtype="float32")
                srs.append(_sr)
                ret[k] = audio
            elif k.startswith("enroll_ref"):
                val = getattr(ex, k)
                if val.startswith("*"):
                    # Training-time pattern: "*UID SPEAKER_ID" → random enrollment
                    cur_uid, spkid = val[1:].strip().split(maxsplit=1)
                    enroll_uid, enroll_path = random.choice(self.spk2enroll[spkid])
                    while enroll_uid == cur_uid:
                        enroll_uid, enroll_path = random.choice(self.spk2enroll[spkid])
                    audio, _sr = sf.read(str(enroll_path), dtype="float32")
                    srs.append(_sr)
                    ret[k] = audio
                else:
                    audio, _sr = sf.read(str(val), dtype="float32")
                    srs.append(_sr)
                    ret[k] = audio
            elif k.startswith("text_spk"):
                continue
                # ret[k] = getattr(ex, k)
            elif k == "utt_id":
                continue
            elif k == "num_spk":
                continue
                # ret[k] = np.array([ex.num_spk])
            else:
                raise ValueError(f"Unexpected key in example: {k}")
        assert all(sr == srs[0] for sr in srs), (srs, keys)
        return ret

    def _load_enrollment_map(self, enrollment_map_path: Path, split: str) -> dict:
        """Load the fixed enrollment map for dev/test splits."""
        if "train" in split:
            return {}
        partition_dset = self.partition.split("/")[0]
        enrollment_map: dict = {}
        with enrollment_map_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                mix_id, utt_id, enroll_id = line.split()
                sid = mix_id.split("_").index(utt_id) + 1
                enroll_path = self.split_dir / f"{partition_dset}/{enroll_id}.wav"
                enrollment_map.setdefault(mix_id, {})[f"s{sid}"] = enroll_path
        return enrollment_map

    def _parse_dataset(self) -> List:
        """Build an utterance index for the split.

        Returns:
            A list of :class:`Libri2MixTSEExample` or
            :class:`Libri3MixTSEExample` instances.
        """
        data_dir = self.librimix_root / self.split

        # 1. Load scp / transcript files
        key2path: dict = {"speech_mix": data_dir / "wav.scp"}
        for spk in range(1, self.num_spk + 1):
            key2path[f"speech_ref{spk}"] = data_dir / f"spk{spk}.scp"
            key2path[f"text_spk{spk}"] = data_dir / f"text_spk{spk}"

        files = [key2path[k].open("r") for k in key2path]
        uids: set = set()
        info: dict = {}
        for key, fh in zip(key2path.keys(), files):
            tmp_uids: set = set()
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                uid, val = line.split(maxsplit=1)
                info.setdefault(key, {})[uid] = val
                tmp_uids.add(uid)
            if not uids:
                uids = tmp_uids
            else:
                assert uids == tmp_uids, (len(uids), key, len(tmp_uids))
        for fh in files:
            fh.close()

        # 2. Resolve enrollment paths
        if "train" in data_dir.stem:
            # Training: pick enrollment samples randomly on the fly.
            # Cache the speaker→utterance map in a JSON file for speed.
            enroll_json = self.librimix_root / self.split / "spk2enroll.json"
            assert enroll_json.exists(), enroll_json
            with enroll_json.open("r", encoding="utf-8") as f:
                self.spk2enroll = json.load(f)
            assert len(self.spk2enroll) > 0, enroll_json
            # Store a placeholder; actual file is chosen in __getitem__.
            for mix_id in uids:
                utt_ids = mix_id.split("_")
                for spk in range(1, self.num_spk + 1):
                    utt_id = utt_ids[spk - 1]
                    spk_id = utt_id.split("-")[0]
                    info.setdefault(f"enroll_ref{spk}", {})[
                        mix_id
                    ] = f"*{utt_id} {spk_id}"
        else:
            # Dev/test: use the fixed enrollment map.
            for uid in uids:
                for spk in range(1, self.num_spk + 1):
                    enroll_path = self.enrollment_map[uid][f"s{spk}"]
                    info.setdefault(f"enroll_ref{spk}", {})[uid] = str(enroll_path)

        Example = Libri2MixTSEExample if self.num_spk == 2 else Libri3MixTSEExample
        ex_keys = [
            f.name for f in fields(Example) if f.name not in ("utt_id", "num_spk")
        ]
        examples = []
        for uid in uids:
            kwargs = {k: info[k][uid] for k in ex_keys}
            kwargs["utt_id"] = uid
            examples.append(Example(**kwargs))
        return examples
