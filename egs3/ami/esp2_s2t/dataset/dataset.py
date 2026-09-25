"""AMI serialized-output-training dataset for the ESPnet3 S2T recipe.

One split is one Lhotse CutSet. Each cut is an utterance group: it names the
recording, channel, offset and duration its audio comes from, so the audio is
read through the cut rather than from a copy written out per group. The
serialized reference for every speaker in the group, separated by the
speaker-change symbol and carrying Whisper timestamp tokens, rides on the cut
as ``sot_text``.

Constructing an ``AmiSotDataset`` records the split it was built with (see
``current_utt_id``). That record has to be an environment variable rather
than a plain module-level variable: ESPnet3 loads a recipe's own
``dataset/__init__.py`` through a fresh, uniquely named module spec for
every dataset it builds
(``espnet3.components.data.dataset_module._load_local_dataset_module``), so
the copy of this file that constructs ``AmiSotDataset`` for a real run is
not necessarily the same module object ``src/inference.py`` imports by its
stable, dotted name -- a plain global set by one is invisible to the other.
``os.environ`` is the one thing every copy of this module actually shares.
``src/inference.py`` resolves an inference index to an utterance id through
``current_utt_id`` instead of loading its own, independent copy of a split's
ids, so there is exactly one place -- the split the dataset the framework
actually built for this run was constructed with -- an id can come from.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset as TorchDataset

from espnet3.utils.config_utils import load_config_with_defaults

# Written by AmiSotDataset.__init__, read by current_utt_id. See the module
# docstring for why this is an environment variable and not a plain
# module-level variable.
_CURRENT_SPLIT_ENV = "_AMI_SOT_CURRENT_SPLIT"

# Both must match dataset/builder.py.
_CUTS_FILENAME = "cuts.jsonl.gz"
_SOT_TEXT_FIELD = "sot_text"


def _load_builder_config() -> dict:
    config_path = Path(__file__).resolve().parent / "config.yaml"
    return load_config_with_defaults(str(config_path))["builder"]


_CONFIG = _load_builder_config()


def _split_dir(split: str) -> Path:
    """Return the directory holding one split's cuts.

    Args:
        split: Key of ``builder.split_dirs``.

    Returns:
        ``<data_root>/<split_dirs[split]>``.

    Raises:
        KeyError: When ``split`` is not a configured split.
    """
    split_dirs = _CONFIG["split_dirs"]
    if split not in split_dirs:
        raise KeyError(
            f"Unknown split {split!r}; expected one of {sorted(split_dirs)}."
        )
    return Path(_CONFIG["data_root"]) / split_dirs[split]


# Keyed by (path, mtime, size), not by split, so a corpus rebuilt in the same
# process is re-read rather than served stale. Parsing a split is not free --
# the train CutSet holds 53k cuts -- and load_utt_ids, load_reference_texts
# and AmiSotDataset would otherwise each pay for it separately.
_CUTS_CACHE: Dict[tuple, Any] = {}


def load_cuts(split: str):
    """Return one split's CutSet, eagerly.

    Args:
        split: Split name.

    Returns:
        A ``lhotse.CutSet`` supporting both integer and id indexing.

    Raises:
        FileNotFoundError: When the split has not been built.
    """
    from lhotse import CutSet

    path = _split_dir(split) / _CUTS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Run the create_dataset stage for split "
            f"{split!r} before reading it."
        )
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _CUTS_CACHE:
        _CUTS_CACHE.clear()
        _CUTS_CACHE[key] = CutSet.from_file(str(path)).to_eager()
    return _CUTS_CACHE[key]


def load_utt_ids(split: str) -> List[str]:
    """Return the utterance ids of one split, in cut order.

    Args:
        split: Split name.

    Returns:
        One id per cut, in the order the dataset indexes them.
    """
    return [cut.id for cut in load_cuts(split)]


def load_reference_texts(split: str) -> Dict[str, str]:
    """Return the serialized reference text of one split, keyed by id.

    Args:
        split: Split name.

    Returns:
        ``{utt_id: sot_text}`` for every cut in the split.
    """
    return {
        cut.id: (cut.custom or {}).get(_SOT_TEXT_FIELD, "") for cut in load_cuts(split)
    }


class AmiSotDataset(TorchDataset):
    """One AMI utterance group per item.

    Args:
        split: Split name, for example ``"test"``.

    Examples:
        >>> dataset = AmiSotDataset(split="test")  # doctest: +SKIP
        >>> sorted(dataset[0])  # doctest: +SKIP
        ['speech', 'text', 'text_ctc', 'text_prev']
    """

    def __init__(self, split: str) -> None:
        """Read the split's cuts once and keep them in memory.

        Records ``split`` as this process's current split (see
        ``current_utt_id``) once construction fully succeeds. A split that
        fails to read raises before that point, so a half-built dataset is
        never recorded.
        """
        self.split = split
        self._cuts = load_cuts(split)
        self._na = _CONFIG["na_symbol"]
        os.environ[_CURRENT_SPLIT_ENV] = split

    def __len__(self) -> int:
        """Return the number of utterance groups in the split."""
        return len(self._cuts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one sample.

        Args:
            idx: Position in cut order.

        Returns:
            ``speech`` as a float32 mono waveform and the three text fields
            ``text``, ``text_prev`` and ``text_ctc``. No other field: a recipe
            sample must carry only what the model and preprocessor accept, so
            the cut's own id is deliberately absent (see ``current_utt_id``).
        """
        cut = self._cuts[int(idx)]
        # The cut carries its own offset, duration and channel, so this is
        # exactly the group's audio, without a per-group file on disk.
        array = cut.load_audio()[0]
        return {
            "speech": np.asarray(array, dtype=np.float32),
            "text": (cut.custom or {}).get(_SOT_TEXT_FIELD, ""),
            # This recipe conditions on no previous text and trains with
            # ctc_weight 0.0, so both are the not-available symbol.
            "text_prev": self._na,
            "text_ctc": self._na,
        }


# Cache for current_utt_id, keyed by the split it was computed for, so a long
# test set does not re-read the cuts once per utterance. Reset whenever the
# recorded split changes.
_CACHED_SPLIT: Optional[str] = None
_CACHED_UTT_IDS: Optional[List[str]] = None


def current_utt_id(idx: int) -> str:
    """Return the utterance id for ``idx`` in the split this process built.

    ``src/inference.py``'s ``build_output`` reads an id through this
    function instead of loading its own, independent copy of a split's ids,
    so the two cannot disagree about which split is being scored: whatever
    split an ``AmiSotDataset`` was actually constructed with (recorded in
    ``os.environ``, see the module docstring) is the split ids are read
    from.

    Args:
        idx: Position in the recorded split's cut order.

    Returns:
        The utterance id at ``idx`` in the recorded split.

    Raises:
        RuntimeError: If no ``AmiSotDataset`` has been constructed yet in
            this process, so there is no split to resolve ``idx`` against.
    """
    global _CACHED_SPLIT, _CACHED_UTT_IDS
    split = os.environ.get(_CURRENT_SPLIT_ENV)
    if split is None:
        raise RuntimeError(
            "current_utt_id() was called before any AmiSotDataset was "
            f"built in this process (checked os.environ[{_CURRENT_SPLIT_ENV!r}])"
            ". build_output needs the split the dataset "
            "espnet3.systems.base.inference.infer constructs for the run "
            "(via provider.build_dataset, before the runner starts) so it "
            "can read that split's own utterance ids; call it only from "
            "there."
        )
    if split != _CACHED_SPLIT:
        _CACHED_SPLIT = split
        _CACHED_UTT_IDS = load_utt_ids(split)
    return _CACHED_UTT_IDS[int(idx)]
