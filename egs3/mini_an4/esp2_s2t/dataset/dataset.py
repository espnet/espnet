"""Mini AN4 SOT dataset for the ESPnet3 S2T recipe.

Each split is one Lhotse CutSet, in the same format as in egs3/ami/esp2_s2t.
Each cut is an utterance group over a recording that the builder of this
recipe made by joining AN4 utterances from different speakers. The
serialized reference of the group is stored in the cut as ``sot_text``. Each
item has the cut id as ``utt_id``, which inference uses to name its outputs.
The preprocessor removes ``utt_id`` before training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from espnet3.components.data.base_dataset import BaseDataset
from espnet3.contract import Field
from espnet3.utils.config_utils import load_config_with_defaults

# Both values must be the same as in dataset/builder.py.
_CUTS_FILENAME = "cuts.jsonl.gz"
_SOT_TEXT_FIELD = "sot_text"
# Whisper only accepts 16 kHz audio. The speech of an item is a plain array,
# and both the training preprocessor and the inference contract assume this
# rate.
_SAMPLE_RATE = 16000


def _load_builder_config() -> dict:
    config_path = Path(__file__).resolve().parent / "config.yaml"
    return load_config_with_defaults(str(config_path))["builder"]


_CONFIG = _load_builder_config()


def _split_dir(split: str) -> Path:
    """Return the directory of one split.

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
            f"Unknown split {split!r}. Expected one of {sorted(split_dirs)}."
        )
    return Path(_CONFIG["data_root"]) / split_dirs[split]


# The cache key is (path, mtime, size), so a corpus that is rebuilt in the
# same process is read again. Without the cache, every dataset of the same
# split would read the CutSet again.
_CUTS_CACHE: Dict[tuple, Any] = {}


def load_cuts(split: str):
    """Return the CutSet of one split, loaded into memory.

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


class MiniAn4SotDataset(BaseDataset):
    """One mini AN4 utterance group per item.

    Args:
        split: Split name, for example ``"test"``.

    Examples:
        >>> dataset = MiniAn4SotDataset(split="test")  # doctest: +SKIP
        >>> sorted(dataset[0])  # doctest: +SKIP
        ['speech', 'text', 'text_ctc', 'text_prev', 'utt_id']
    """

    fields = (
        Field("utt_id", "text"),
        Field("speech", "audio"),
        Field("text", "text"),
        Field("text_prev", "text"),
        Field("text_ctc", "text"),
    )

    def __init__(self, split: str) -> None:
        """Read the split's cuts once and keep them in memory."""
        self.split = split
        self._cuts = load_cuts(split)
        self._na = _CONFIG["na_symbol"]

    def __len__(self) -> int:
        """Return the number of utterance groups in the split."""
        return len(self._cuts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one sample.

        Args:
            idx: Position in cut order.

        Returns:
            The cut's id as ``utt_id``, ``speech`` as a float32 mono waveform
            at 16 kHz, and the three text fields ``text``, ``text_prev`` and
            ``text_ctc``.

        Raises:
            ValueError: If the audio of the cut is not 16 kHz mono. The
                returned array does not include the sampling rate, so it is
                checked here.
        """
        cut = self._cuts[int(idx)]
        # The cut stores its start time, duration and channel, so this loads
        # only the audio of the group.
        array = cut.load_audio()
        if cut.sampling_rate != _SAMPLE_RATE or array.shape[0] != 1:
            raise ValueError(
                f"cut {cut.id} has {array.shape[0]}-channel audio at "
                f"{cut.sampling_rate} Hz, but this recipe needs 16 kHz mono "
                "audio"
            )
        return {
            "utt_id": cut.id,
            "speech": np.asarray(array[0], dtype=np.float32),
            "text": (cut.custom or {}).get(_SOT_TEXT_FIELD, ""),
            # This recipe uses no previous text and trains with ctc_weight
            # 0.0, so both fields are the not-available symbol.
            "text_prev": self._na,
            "text_ctc": self._na,
        }
