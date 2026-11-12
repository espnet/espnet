"""LibriMix TSE dataset implementation.

This module provides :class:`LibriMixTSEDataset`, a Torch-style dataset that
reads the simulated LibriMix directory structure produced by
:class:`egs3.librimix.esp2_tse.dataset.DatasetBuilder`.

The dataset reads prepared path-valued manifests without downloading or building
anything. Training enrollment values are placeholders when training_placeholders
is enabled; dev/test enrollment values are fixed audio paths. Metadata can be
retained for evaluation and is omitted from training by the default key filter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from importlib import resources
from pathlib import Path
from typing import Dict, List

from egs3.librimix.esp2_tse.dataset.builder import resolve_librimix_root
from espnet3.components.data.base_dataset import BaseDataset
from espnet3.contract import Field
from espnet3.utils.config_utils import load_config_with_defaults

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


def _has_missing_files(folder: Path, split: str, num_spk: int) -> bool:
    """Check that basic data files exist; return True if enrollment map exists."""
    split_dir = folder / split
    required = ["wav.scp", "spk1.scp", "spk2.scp", "text_spk1", "text_spk2"]
    if num_spk == 3:
        required += ["spk3.scp", "text_spk3"]
    for fname in required:
        if not (split_dir / fname).exists():
            raise FileNotFoundError(
                f"Missing basic file: {split_dir / fname}. "
                "Please run the 'create_dataset' stage before loading the dataset."
            )
    if "train" in split:
        return True
    return (folder / f"{split}/mixture2enrollment").exists()


class LibriMixTSEDataset(BaseDataset):
    """Torch dataset that reads simulated LibriMix data from disk.

    Args:
        split: LibriMix split name, e.g. ``"2mix_16k_max_train_mix-both"``.
            Format: ``{num_spk}mix_{fs}_{mode}_{dset}_{mix_type}``.
        recipe_dir: Optional recipe root directory.  When ``None``, defaults to
            ``<this_file>/../`` (i.e. the recipe root inferred from the module
            path).
        data_dir: Optional explicit path to the LibriMix dataset root (the
            directory that contains the split sub-directories and
            ``LibriMix/``). When ``None``, ``<recipe_dir>/data/`` is used.
        ignore_key_prefix: List of key prefixes to omit from returned dicts.
            Supported keys: ``speech_mix``, ``enroll_ref{N}``,
            ``speech_ref{N}``, ``text_spk{N}``, ``utt_id``, ``num_spk``.
        training_placeholders: Enable ``*utterance speaker`` enrollment values
            for training splits; leave false for fixed evaluation enrollment.

    Raises:
        ValueError: If ``split`` is not listed in ``config.yaml``.
        FileNotFoundError: If the data directory or required split files are
            missing.
        RuntimeError: If no utterance pairs are found for the split.

    Examples:
        >>> dataset = LibriMixTSEDataset(split="2mix_16k_max_dev_mix-clean")
        >>> sample = dataset[0]
        >>> sorted(sample.keys())
        ['enroll_ref1', 'enroll_ref2', 'speech_mix', 'speech_ref1', 'speech_ref2']
    """

    def __init__(
        self,
        split: str = "2mix_16k_max_train_mix-both",
        recipe_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
        ignore_key_prefix: List[str] | None = None,
        training_placeholders: bool = False,
    ) -> None:
        """Index prepared manifests with the selected raw enrollment mode."""
        self.split = str(split)
        if self.split not in _KNOWN_SPLITS:
            known = ", ".join(sorted(_KNOWN_SPLITS))
            raise ValueError(f"Unknown split '{self.split}'. Expected one of: {known}")

        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )

        # Parse split parameters before any build decision:
        #   {num_spk}mix_{fs}_{mode}_{dset}_{mix_type}
        # e.g. "2mix_16k_max_train_mix-both" → num_spk=2, fs=16k, mode=max,
        #       dset=train, mix_type=mix_both
        num_spk_str, fs, mode, dset, mix_type = self.split.split("_", 4)
        mix_type_us = mix_type.replace("-", "_")  # mix-both → mix_both
        self.num_spk = int(num_spk_str.split("mix")[0])
        self.partition = f"{dset}/{mix_type_us}"

        self.training_placeholders = training_placeholders
        if "train" in dset and not training_placeholders:
            raise ValueError(
                "Training splits require training_placeholders=True; "
                "use a dev/test split with fixed enrollment for inference."
            )
        self.librimix_root = resolve_librimix_root(
            (
                Path(data_dir).resolve()
                if data_dir is not None
                else recipe_root / _CONFIG["builder"]["dataset_path"]
            ),
            self.split,
        )

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

        self.ignore_key_prefix = tuple(
            ["text_spk", "utt_id", "num_spk"]
            if ignore_key_prefix is None
            else ignore_key_prefix
        )

        # Load or download the enrollment map (dev/test only)
        enrollment_map_path = self.librimix_root / f"{self.split}/mixture2enrollment"
        has_map = _has_missing_files(self.librimix_root, self.split, self.num_spk)
        if not has_map:
            raise FileNotFoundError(
                f"Missing enrollment map: {enrollment_map_path}. "
                "Run the create_dataset stage before loading the dataset."
            )
        self.enrollment_map = self._load_enrollment_map(enrollment_map_path, self.split)
        example_type = Libri2MixTSEExample if self.num_spk == 2 else Libri3MixTSEExample
        self.fields = tuple(
            Field(
                f.name,
                (
                    "text"
                    if f.name.startswith("text_spk")
                    or f.name == "utt_id"
                    or (f.name.startswith("enroll_ref") and training_placeholders)
                    else "number" if f.name == "num_spk" else "audio"
                ),
            )
            for f in fields(example_type)
            if not f.name.startswith(self.ignore_key_prefix)
        )

        self._examples = self._parse_dataset()
        print(f"len(LibriMixTSEDataset[{self.split}])={len(self)}", flush=True)

    def __len__(self) -> int:
        """Return the number of indexed mixtures."""
        return len(self._examples)

    def __getitem__(self, idx: int) -> Dict:
        """Load one item by index.

        Returns:
            dict with keys (filtered by ``ignore_key_prefix``):

            - ``speech_mix``: mixture audio path
            - ``enroll_ref{N}``: audio path for enrollment waveform
            - ``speech_ref{N}``: audio path for clean source waveform
        """
        ex = self._examples[int(idx)]
        keys = [f.name for f in fields(ex)]
        ret = {}
        for k in keys:
            if k.startswith(self.ignore_key_prefix):
                continue
            ret[k] = getattr(ex, k)
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
                if uid in tmp_uids:
                    raise ValueError(
                        f"Duplicate utterance ID {uid!r} in {key2path[key]}"
                    )
                info.setdefault(key, {})[uid] = val
                tmp_uids.add(uid)
            if not uids:
                uids = tmp_uids
            else:
                if uids != tmp_uids:
                    raise ValueError(f"Manifest utterance IDs do not match for {key}")
        for fh in files:
            fh.close()
        uids = sorted(uids)
        if not uids:
            raise RuntimeError(f"No utterances found in {data_dir}")

        # 2. Resolve enrollment paths
        if "train" in data_dir.stem:
            # Training: pick enrollment samples randomly on the fly.
            # Cache the speaker→utterance map in a JSON file for speed.
            enroll_json = self.librimix_root / self.split / "spk2enroll.json"
            if not enroll_json.exists():
                raise FileNotFoundError(f"Missing {enroll_json}; run create_dataset")
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
                    try:
                        enroll_path = self.enrollment_map[uid][f"s{spk}"]
                    except KeyError as exc:
                        raise ValueError(
                            f"Missing enrollment for {uid}, speaker {spk}"
                        ) from exc
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
