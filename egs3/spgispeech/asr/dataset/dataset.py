"""SPGISpeech dataset implementation backed by the raw corpus directly.

SPGISpeech ships ``train.csv``/``val.csv`` manifests with
``wav_filename|wav_filesize|transcript`` columns plus a matching
``spgispeech/{train,val}/<hash>/<n>.wav`` audio tree. This module reads that
raw layout directly, reproducing the utterance ids, dev/train split, and
text normalization that ``egs2/spgispeech/asr1/local/data.sh`` +
``local/data_prep.sh`` previously produced via Kaldi:

* utterance id: the wav path with its first ``/`` replaced by ``-`` and the
  ``.wav`` suffix stripped (e.g. ``07a785e...c1354bb60abca42d5/1.wav`` ->
  ``07a785e...c1354bb60abca42d5-1``).
* ``train`` / ``val``: the full csv, indexed in utterance-id sorted order
  (matching the sorted ``wav.scp``/``text`` that Kaldi's ``data_prep.sh``
  produced).
* ``dev_4k`` / ``train_nodev``: the first 4000 / remaining utterances of the
  sorted ``train`` split, matching
  ``utils/subset_data_dir.sh --first|--last`` in ``data.sh`` stage 2.
* ``*_unnorm`` splits: the raw transcript text, unmodified.
* Normalized splits (``train``, ``val``, ``dev_4k``, ``train_nodev``): the
  transcript with per-token punctuation stripped and lowercased, matching
  ``data_prep.sh``'s normalized ``text`` output.
"""

from __future__ import annotations

import csv
import functools
import string
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from egs3.spgispeech.asr.dataset.builder import (
    SPGISpeechBuilder,
    resolve_source_root,
)
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_BUILDER_CFG = _CONFIG["builder"]
_DATASET_CFG = _CONFIG["dataset"]

_KNOWN_SPLITS = {str(split) for split in _DATASET_CFG["supported_splits"]}
_DEV_SIZE = int(_DATASET_CFG["dev_size"])
_AUDIO_SUBDIR = str(_BUILDER_CFG["audio_subdir"])
_CSV_FILES = {str(k): str(v) for k, v in _BUILDER_CFG["csv_files"].items()}

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


@dataclass(frozen=True)
class SPGISpeechExample:
    """Internal index entry derived from one SPGISpeech csv row."""

    utt_id: str
    audio_path: Path
    raw_text: str


def _utt_id_from_wav_filename(wav_filename: str) -> str:
    """Reproduce data_prep.sh's utt-id derivation from a csv wav_filename."""
    stem = (
        wav_filename[: -len(".wav")] if wav_filename.endswith(".wav") else wav_filename
    )
    return stem.replace("/", "-", 1)


def normalize_text(raw_text: str) -> str:
    """Strip per-token punctuation and lowercase, matching data_prep.sh."""
    tokens = raw_text.strip().split()
    normalized_tokens = [tok.translate(_PUNCT_TABLE) for tok in tokens]
    return " ".join(tok for tok in normalized_tokens if tok).lower()


@functools.lru_cache(maxsize=8)
def _load_base_split(
    source_root: Path, base_split: str
) -> tuple[SPGISpeechExample, ...]:
    """Parse one base csv ('train' or 'val') into a utt-id sorted index."""
    csv_path = source_root / _CSV_FILES[base_split]
    audio_root = source_root / _AUDIO_SUBDIR / base_split

    examples: list[SPGISpeechExample] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="|")
        header = next(reader, None)
        if header is None:
            raise RuntimeError(f"Empty manifest: {csv_path}")
        for row in reader:
            if not row:
                continue
            wav_filename, _wav_filesize, transcript = row[0], row[1], row[2]
            examples.append(
                SPGISpeechExample(
                    utt_id=_utt_id_from_wav_filename(wav_filename),
                    audio_path=audio_root / wav_filename,
                    raw_text=transcript,
                )
            )

    if not examples:
        raise RuntimeError(
            f"No utterances found in manifest: {csv_path}. "
            "Check that the SPGISpeech source root is correct."
        )
    return tuple(sorted(examples, key=lambda example: example.utt_id))


def _resolve_split(split: str, source_root: Path) -> tuple[SPGISpeechExample, ...]:
    """Map a supported split name to its (sliced) sorted example tuple."""
    base_split = split[: -len("_unnorm")] if split.endswith("_unnorm") else split

    if base_split in _CSV_FILES:
        return _load_base_split(source_root, base_split)

    if base_split in {"dev_4k", "train_nodev"}:
        train_examples = _load_base_split(source_root, "train")
        if base_split == "dev_4k":
            return train_examples[:_DEV_SIZE]
        return train_examples[_DEV_SIZE:]

    raise ValueError(f"Unhandled split '{split}'")


class SPGISpeechDataset(TorchDataset):
    """Torch dataset that reads SPGISpeech from the raw corpus directly.

    Args:
        split: One of the supported split names (see ``config.yaml``):
            ``train``, ``val``, ``dev_4k``, ``train_nodev``, and their
            ``*_unnorm`` (unnormalized transcript) counterparts.
        recipe_dir: Optional recipe root. When omitted, defaults to the
            current recipe directory inferred from this module.
        source_dir: Optional SPGISpeech root override. When omitted, resolves
            from ``<recipe_dir>/download/spgispeech`` and then the
            ``SPGISPEECH`` environment variable (see ``dataset/config.yaml``).
        cache: Optional cache config. When enabled, the split is read from the
            HuggingFace audio index written by ``create_dataset`` instead of
            the raw csv manifests.

    Raises:
        ValueError: If ``split`` is unknown.
        FileNotFoundError: If the resolved source root is missing required
            files.
        RuntimeError: If no utterances are found for the split's manifest.

    Examples:
        >>> dataset = SPGISpeechDataset(split="val")
        >>> sample = dataset[0]
        >>> sorted(sample.keys())
        ['speech', 'text']
    """

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        source_dir: str | Path | None = None,
        cache: dict | None = None,
    ) -> None:
        """Resolve the split, preferring the cached index over the raw corpus."""
        self.split = str(split)
        if self.split not in _KNOWN_SPLITS:
            known = ", ".join(sorted(_KNOWN_SPLITS))
            raise ValueError(f"Unknown split '{self.split}'. Expected one of: {known}")

        self._hf_cache = _load_hf_cache(cache, recipe_dir, self.split)
        if self._hf_cache is not None:
            return

        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )

        builder = SPGISpeechBuilder()
        if not builder.is_source_prepared(
            recipe_dir=recipe_root,
            source_dir=source_dir,
        ):
            builder.prepare_source(recipe_dir=recipe_root, source_dir=source_dir)

        self.source_root = resolve_source_root(recipe_root, source_dir=source_dir)
        self._normalize = not self.split.endswith("_unnorm")
        self._examples = _resolve_split(self.split, self.source_root)

    def __len__(self) -> int:
        """Return the number of utterances in this split."""
        if self._hf_cache is not None:
            return len(self._hf_cache)
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return ``{speech, text}`` for one utterance."""
        if self._hf_cache is not None:
            row = self._hf_cache[int(idx)]
            array, _sr = sf.read(str(row["audio_path"]))
            # No "utt_id". espnet2's CommonPreprocessor tokenizes "text" into an
            # np.ndarray but passes keys it does not recognise through UNCHANGED,
            # so a str reaches the collate function, which assumes every value is
            # an array (espnet2/train/collate_fn.py:404, `data[0][key].dtype`):
            #   AttributeError: 'str' object has no attribute 'dtype'
            # This kills training on the FIRST batch if utt_id is added here,
            # and collect_stats goes through the same path and fails the same way.
            #
            # espnet3 identifies samples by index, not by this field: it passes
            # str(idx) to the preprocessor as the utterance id
            # (espnet3/components/data/dataset.py:208), and collect_stats keys its
            # shape files by that integer (exp/stats/train/feats_shape starts
            # "0 938,80"). The CommonVoice recipe likewise returns speech/text only.
            #
            # The HF cache DOES carry a usable "utt_id" column. To surface it at
            # inference without breaking training, add a `transform` that drops it
            # to the dataset.train/valid entries only -- transforms run before the
            # preprocessor (espnet3/components/data/dataset.py) -- and leave
            # conf/inference.yaml's entries without one.
            return {
                "speech": np.asarray(array, dtype=np.float32),
                "text": str(row["text"]),
            }
        example = self._examples[int(idx)]
        array, _sr = sf.read(str(example.audio_path))
        text = normalize_text(example.raw_text) if self._normalize else example.raw_text
        return {
            "speech": np.asarray(array, dtype=np.float32),
            "text": text,
        }


def gather_training_text(
    recipe_dir: str | Path,
    source_dir: str | Path | None = None,
    **_kwargs: Any,
) -> list[str]:
    """Collect normalized training text, preferring the verified HF cache."""
    cached = _load_hf_cache(_kwargs.get("cache"), recipe_dir, "train")
    if cached is not None:
        return [str(text) for text in cached["text"]]
    recipe_root = Path(recipe_dir).resolve()
    root = resolve_source_root(recipe_root, source_dir=source_dir)
    return [
        normalize_text(example.raw_text) for example in _resolve_split("train", root)
    ]


def _load_hf_cache(cache, recipe_dir, split):
    """Load the cached audio index for ``split``, or None when caching is off."""
    import os

    environment_root = os.environ.get("EGS3_HF_CACHE_DIR")
    if cache is None and environment_root:
        cache = {"enabled": True, "backend": "hf", "cache_dir": environment_root}
    if not cache or not cache.get("enabled", False):
        return None
    from datasets import load_from_disk

    root = Path(cache.get("cache_dir", "data/hf"))
    if not root.is_absolute():
        root = Path(recipe_dir or Path.cwd()) / root
    split_root = root / "hf_audio_index" / str(split)
    if not split_root.is_dir():
        raise FileNotFoundError(
            f"HF audio cache is missing: {split_root}. "
            "Run DatasetBuilder.build() first."
        )
    return load_from_disk(str(split_root))
