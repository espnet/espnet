"""LibriMix dataset loader for ESPnet3 recipes.

This module provides :class:`LibriMixTSEDataset`, a small Torch-style dataset
implementation that reads the *original* LibriMix directory structure on disk.

Design goals
------------
- Minimal and explicit: it only reads audio + transcript and returns them.
- Config-friendly: it is meant to be instantiated by Hydra via
  ``_target_: src.dataset.LibriMixTSEDataset``.
- "Just works" defaults: when ``data_dir`` is not provided, it falls back to the
  ``LIBRIMIX`` environment variable; default split is ``2mix_16k_max_train_mix-both``.

What this dataset returns
-------------------------
Each ``__getitem__`` returns a plain dict:

- ``speech_mix``: ``np.float32`` waveform array (loaded from a ``.wav`` file)
- ``enroll_ref1``: ``np.float32`` waveform array (loaded from a ``.wav`` file)
- ``enroll_ref2``: ``np.float32`` waveform array (loaded from a ``.wav`` file)
- [Optional] ``enroll_ref3``: ``np.float32`` waveform array (only for Libri3Mix)
- ``speech_ref1``: ``np.float32`` waveform array (loaded from a ``.wav`` file)
- ``speech_ref2``: ``np.float32`` waveform array (loaded from a ``.wav`` file)
- [Optional] ``speech_ref3``: ``np.float32`` waveform array (only for Libri3Mix)
- ``text_spk1``: transcript string
- ``text_spk2``: transcript string
- [Optional] ``text_spk3``: transcript string (only for Libri3Mix)

There is no resampling, no feature extraction, and no optional return flags.
Those should be handled elsewhere in the pipeline (frontend / preprocessor).

Expected directory layout
-------------------------
This loader expects a LibriMix tree like:

```
<data_dir>/
|-- {split}
|   |-- wav.scp
|   |-- spk1.scp
|   |-- ...
|   |-- text_spk1
|   `-- ...
`-- LibriMix
    `-- libri_mix
        |-- Libri2Mix
        |   |-- wav16k
        |   |   |-- max
        |   |   |   |-- train-100
        |   |   |   |   |-- mix_both
        |   |   |   |   |   |-- 103-1240-0003_1235-135887-0017.wav
        |   |   |   |   |   `-- ...
        |   |   |   |   |-- mix_clean/
        |   |   |   |   |-- mix_single/
        |   |   |   |   |-- s1/
        |   |   |   |   |-- s2/
        |   |   |   |   `-- noise/
        |   |   |   |-- train-360/
        |   |   |   |-- dev/
        |   |   |   |-- test/
        |   |   |   `-- metadata/*.csv
        |   |   `-- min/
        |   `-- wav8k/
        `-- Libri3Mix/
```

For convenience, ``data_dir`` should point to: the directory that contains ``LibriMix/``
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections import defaultdict
from dataclasses import dataclass, fields
from itertools import chain
from pathlib import Path
from typing import Dict, List

import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from espnet3.utils.download_utils import download_url


@dataclass(frozen=True)
class Libri2MixTSEExample:
    """Internal index entry derived from a Libri2MixTSE transcript line."""

    uttid: str
    speech_mix: str
    enroll_ref1: str
    enroll_ref2: str
    speech_ref1: str
    speech_ref2: str
    text_spk1: str
    text_spk2: str


@dataclass(frozen=True)
class Libri3MixTSEExample:
    """Internal index entry derived from a Libri3MixTSE transcript line."""

    uttid: str
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


def get_spk2utt_librimix(paths, audio_format="wav"):
    spk2utt = defaultdict(list)
    for path in paths:
        for audio in chain(
            Path(path).rglob("s1/*.{}".format(audio_format)),
            Path(path).rglob("s2/*.{}".format(audio_format)),
            Path(path).rglob("s3/*.{}".format(audio_format)),
        ):
            spk_idx = int(audio.parent.stem[1:]) - 1
            mix_uid = audio.stem
            uid = mix_uid.split("_")[spk_idx]
            sid = uid.split("-")[0]
            spk2utt[sid].append((uid, str(audio)))

    return spk2utt


def _resolve_librimix_root(data_dir: str | Path, split: str) -> Path:
    """Resolve the LibriMix root directory.

    Args:
        data_dir: Either the directory that contains ``LibriMix/`` or the
            ``LibriMix/`` directory itself.
        split: The dataset split (e.g., ``2mix_16k_max_train_mix-both``) to check for
            existence of data files.

    Returns:
        Path to the resolved ``LibriMix`` root directory.

    Raises:
        FileNotFoundError: If the LibriMix root cannot be found.
    """
    p = Path(data_dir)
    if (p / "LibriMix").is_dir() and (p / split).is_dir():
        return p
    raise FileNotFoundError(
        "Could not find LibriMix root. Expected a directory containing both:\n"
        f"  - {p}/LibriMix/\n"
        f"  - {p}/{split}"
    )


def _check_missing_files(folder: Path, split: str, num_spk: int) -> bool:
    # Check if the basic data files exist for the given split
    split_dir = folder / split
    lst = ["wav.scp", "spk1.scp", "spk2.scp", "text_spk1", "text_spk2"]
    if num_spk == 3:
        lst.append("spk3.scp")
        lst.append("text_spk3")
    for fname in lst:
        if not (split_dir / fname).exists():
            raise FileNotFoundError(
                f"Missing file: {split_dir / fname}. "
                f"Please run `{Path(__file__).resolve()}/create_dataset.py` "
                "before loading the dataset."
            )

    # Check if the enrollment map file exists for the given split
    if "train" in split:
        return True
    return (folder / f"data/{split}/mixture2enrollment").exists()


def _download_missing_files(folder: Path, split: str) -> None:
    if "dev" in split:
        download_url(
            "https://raw.githubusercontent.com/BUTSpeechFIT/speakerbeam/"
            "main/egs/libri2mix/data/wav8k/min/dev/map_mixture2enrollment",
            folder / f"data/{split}/mixture2enrollment",
        )
    elif "test" in split:
        download_url(
            "https://raw.githubusercontent.com/BUTSpeechFIT/speakerbeam/"
            "main/egs/libri2mix/data/wav8k/min/test/map_mixture2enrollment",
            folder / f"data/{split}/mixture2enrollment",
        )


class LibriMixTSEDataset(TorchDataset):
    """Torch dataset that reads LibriMix from disk.

    Args:
        data_dir: Path to the LibriMix dataset root. If omitted, the dataset
            reads ``LIBRIMIX`` from the environment.
        split: LibriMix split directory name (default: ``2mix_16k_max_train_mix-both``).
            format: ``{num_spk}mix_{fs}_{mode}_{mix_type}``, e.g., ``2mix_16k_max_train_mix-both``.

    Raises:
        FileNotFoundError: If ``data_dir`` is missing and ``LIBRIMIX`` is not set,
            or if the resolved split directory does not exist.
        RuntimeError: If no transcript/audio pairs are found for the split.

    Notes:
        - This dataset returns *decoded* waveforms using ``soundfile``.
        - Audio is returned as float32. Channel handling is delegated to soundfile
          (multi-channel inputs will be returned as a 2D array).
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        split: str = "2mix_16k_max_train_mix-both",
    ) -> None:
        if data_dir is None:
            data_dir = os.environ.get("LIBRIMIX")
            if not data_dir:
                raise FileNotFoundError(
                    "LibriMix data_dir not provided and env var " "LIBRIMIX is not set."
                )
        self.librimix_root = _resolve_librimix_root(data_dir, split)
        self.split = split
        num_spk, fs, mode, dset, mix_type = self.split.split("_")
        self.num_spk = int(num_spk.split("mix")[0])
        self.partition = dset
        self.split_dir = (
            self.librimix_root
            / f"LibriMix/libri_mix/Libri{self.num_spk}Mix/wav{fs}/{mode}/{mix_type}"
        )
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        # Trying to download the enrollment lists if not found locally
        if _check_missing_files(self.librimix_root, split, self.num_spk):
            self.enrollment_map = self._load_enrollment_map(self.librimix_root, split)
        elif os.access(data_dir, os.W_OK):
            _download_missing_files(self.librimix_root, split)
            self.enrollment_map = self._load_enrollment_map(self.librimix_root, split)
        else:
            with tempfile.TemporaryDirectory() as tempdirname:
                _download_missing_files(Path(tempdirname), split)
                self.enrollment_map = self._load_enrollment_map(
                    Path(tempdirname), split
                )

        # Loading the dataset split
        self._examples = self._parse_dataset()

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Dict:
        """Load one item by index.

        Returns:
            dict with keys:
              - ``speech_mix``: np.float32 waveform
              - ``enroll_ref?``: np.float32 waveform
              - ``speech_ref?``: np.float32 waveform
              - ``text_spk?``: transcript string
        """
        ex = self._examples[int(idx)]
        ret = {"uttid": ex.uttid}
        keys = [field.name for field in fields(ex) if field.name != "uttid"]
        srs = []
        for k in keys:
            if k.startswith(("speech_mix", "speech_ref")):
                audio, _sr = sf.read(str(getattr(ex, k)), dtype="float32")
                srs.append(_sr)
                ret[k] = audio
            elif k.startswith("enroll_ref"):
                val = getattr(ex, k)
                if val.startswith("*"):
                    # a special format in `enroll_spk?.scp`:
                    #     MIXTURE_UID *UID SPEAKER_ID
                    cur_uid, spkid = val[1:].strip().split(maxsplit=1)
                    enroll_uid, enroll = random.choice(self.spk2enroll[spkid])
                    while enroll_uid == cur_uid:
                        enroll_uid, enroll = random.choice(self.spk2enroll[spkid])
                    audio, _sr = sf.read(str(enroll), dtype="float32")
                    srs.append(_sr)
                    ret[k] = audio
                else:
                    audio, _sr = sf.read(str(getattr(ex, k)), dtype="float32")
                    srs.append(_sr)
                    ret[k] = audio
            elif k.startswith("text_spk"):
                ret[k] = getattr(ex, k)
            else:
                raise ValueError(f"Unexpected key in example: {k}")
        assert all(sr == srs[0] for sr in srs), (srs, keys)
        return ret

    def _load_enrollment_map(self, enrollment_map_path: Path, split: str):
        if "train" in split:
            return {}
        enrollment_map = {}
        with enrollment_map_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                mix_id, utt_id, enroll_id = line.strip().split()
                sid = mix_id.split("_").index(utt_id) + 1
                # This only works for 'dev' and 'test' subsets
                #     where enrollment samples are fixed.
                # For 'train', enrollment samples are chosen on the fly,
                #     so we don't need to load a map.
                enroll_path = self.split_dir / f"{self.partition}/{enroll_id}.wav"
                enrollment_map.setdefault(mix_id, {})[f"s{sid}"] = enroll_path
        return enrollment_map

    def _parse_dataset(self) -> List:
        """Build an index for one split by reading transcript files.

        Returns:
            A list of :class:`LibriMixTSEExample` entries.

        Raises:
            RuntimeError: If no valid transcript/audio pairs are found.
        """
        examples: List = []

        # 1. Load scp files
        data_dir = self.librimix_root / self.split
        files = [(data_dir / "wav.scp").open("r")]
        key2path = {"speech_mix": data_dir / "wav.scp"}
        for spk in range(1, self.num_spk + 1):
            key2path[f"speech_ref{spk}"] = data_dir / f"spk{spk}.scp"
            key2path[f"text_spk{spk}"] = data_dir / f"text_spk{spk}"
            files.append((data_dir / f"spk{spk}.scp").open("r"))
            files.append((data_dir / f"text_spk{spk}").open("r"))

        uids = set()
        info = {}
        for key, f in zip(key2path.keys(), files):
            tmp_uids = set()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                uid, val = line.split(maxsplit=1)
                info.setdefault(key, {})[uid] = val
                tmp_uids.add(uid)

            if len(uids) == 0:
                uids = tmp_uids
            else:
                assert uids == tmp_uids, (len(uids), key, len(tmp_uids))
        for f in files:
            f.close()

        # 2. Load enrollment paths
        if "train" in data_dir.stem:
            # Mapping used for preparing random enrollment samples during training
            enroll_json = self.librimix_root / self.split / "spk2enroll.json"
            if enroll_json.exists():
                with enroll_json.open("r", encoding="utf-8") as f:
                    self.spk2enroll = json.load(f)
            else:
                self.spk2enroll = get_spk2utt_librimix(
                    [self.split_dir / sset for sset in ["train-100", "train-360"]]
                )
                with enroll_json.open("w", encoding="utf-8") as f:
                    json.dump(self.spk2enroll, f)
            # For training, we choose the auxiliary signal on the fly.
            # Thus, here we use the pattern f"*{uttID} {spkID}" to indicate it.
            # The actual enrollment sample will be chosen in the __getitem__ function.
            for mixID in uids:
                # 100-121669-0004_3180-138043-0053
                uttIDs = mixID.split("_")
                for spk in range(1, self.num_spk + 1):
                    uttID = uttIDs[spk]
                    spkID = uttID.split("-")[0]
                    info.setdefault(f"enroll_ref{spk}", {})[mixID] = f"*{uttID} {spkID}"
        else:
            for uid in uids:
                for spk in range(1, self.num_spk + 1):
                    enroll_path = self.enrollment_map[uid][f"s{spk}"]
                    info.setdefault(f"enroll_ref{spk}", {})[uid] = str(enroll_path)

        Example = Libri2MixTSEExample if self.num_spk == 2 else Libri3MixTSEExample
        keys = [field.name for field in fields(Example) if field.name != "uttid"]
        for uid in uids:
            kwargs = {k: info[k][uid] for k in keys}
            kwargs["uttid"] = uid
            examples.append(Example(**kwargs))
        return examples
