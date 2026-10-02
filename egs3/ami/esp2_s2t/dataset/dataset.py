"""AMI serialized-output-training dataset for the ESPnet3 S2T recipe.

The corpus is read straight from the Kaldi-style directories the ESPnet2
recipe produced. Each utterance group is one sample, and its ``text`` is the
serialized reference for every speaker in that group, separated by ``<sc>``
and carrying Whisper timestamp tokens.

Constructing an ``AmiSotDataset`` records the split it was built with (see
``current_utt_id``). That record has to be an environment variable rather
than a plain module-level variable: ESPnet3 loads a recipe's own
``dataset/__init__.py`` through a fresh, uniquely named module spec for
every dataset it builds
(``espnet3.components.data.dataset_module._load_local_dataset_module``), so
the copy of this file that constructs ``AmiSotDataset`` for a real run is
not necessarily the same module object ``src/inference.py`` imports by its
stable, dotted name -- a plain global set by one is invisible to the other.
``os.environ`` is the one thing every copy of this module actually shares,
the same reason ``src/separator.py`` reads ``AMI_SOT_SPEAKER_CHANGE_SYMBOL``
from it instead of importing a Python value. ``src/inference.py`` resolves
an inference index to an utterance id through ``current_utt_id`` instead of
loading its own, independent copy of a split's ids, so there is exactly one
place -- the split the dataset the framework actually built for this run
was constructed with -- an id can come from.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from espnet3.utils.config_utils import load_config_with_defaults

# Written by AmiSotDataset.__init__, read by current_utt_id. See the module
# docstring for why this is an environment variable and not a plain
# module-level variable.
_CURRENT_SPLIT_ENV = "_AMI_SOT_CURRENT_SPLIT"


def _load_builder_config() -> dict:
    config_path = Path(__file__).resolve().parent / "config.yaml"
    # resolve=True so the AMI_SOT_DATA_ROOT override above is expanded.
    return load_config_with_defaults(str(config_path), resolve=True)["builder"]


_CONFIG = _load_builder_config()


def _split_dir(split: str) -> Path:
    """Return the Kaldi-style directory for one split.

    Args:
        split: One of the keys in ``config.yaml``'s ``builder.split_dirs``.

    Returns:
        Absolute path to that split's directory.

    Raises:
        ValueError: If the split is not configured.
    """
    split_dirs = _CONFIG["split_dirs"]
    if split not in split_dirs:
        raise ValueError(
            f"Unknown split '{split}'. Configured splits: {sorted(split_dirs)}"
        )
    return Path(_CONFIG["data_root"]) / split_dirs[split]


def load_utt_ids(split: str) -> List[str]:
    """Return the utterance ids of one split, in ``wav.scp`` order.

    This is the single source of utterance identity for the recipe. The
    dataset iterates in this order, and ``src/inference.py`` maps an inference
    index back to an id through the same list, so neither side re-derives it.

    Args:
        split: Split name, for example ``"test"``.

    Returns:
        The first whitespace-separated field of every non-empty ``wav.scp``
        line, in file order.
    """
    scp = _split_dir(split) / "wav.scp"
    ids = []
    with scp.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                ids.append(line.split(maxsplit=1)[0])
    return ids


def load_reference_texts(split: str, filename: str = "text") -> Dict[str, str]:
    """Return the reference text of one split, keyed by utterance id.

    Args:
        split: Split name, for example ``"test"``.
        filename: Name of the file inside the split directory. The optional
            ``text.prev`` and ``text.ctc`` use the same format.

    Returns:
        Mapping from utterance id to the rest of its ``text`` line. The value
        keeps its Whisper timestamp tokens and its ``<sc>`` separators.
    """
    text = _split_dir(split) / filename
    refs = {}
    with text.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            refs[parts[0]] = parts[1].strip() if len(parts) > 1 else ""
    return refs


def _load_wav_paths(split: str) -> List[Path]:
    """Return one absolute audio path per utterance, in ``wav.scp`` order."""
    root = Path(_CONFIG["data_root"])
    scp = _split_dir(split) / "wav.scp"
    paths = []
    with scp.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            _, rel = line.split(maxsplit=1)
            path = Path(rel)
            paths.append(path if path.is_absolute() else root / path)
    return paths


class AmiSotDataset(TorchDataset):
    """One AMI utterance group per item.

    Args:
        split: Split name, for example ``"test"``.

    Examples:
        >>> dataset = AmiSotDataset(split="test")  # doctest: +SKIP
        >>> sorted(dataset[0])  # doctest: +SKIP
        ['speech', 'text']
    """

    def __init__(self, split: str) -> None:
        """Read the split's manifests once and keep them in memory.

        Records ``split`` as this process's current split (see
        ``current_utt_id``) once construction fully succeeds. A split that
        fails to parse raises before that point, so a half-built dataset is
        never recorded.
        """
        self.split = split
        self._utt_ids = load_utt_ids(split)
        self._wav_paths = _load_wav_paths(split)
        if len(self._wav_paths) != len(self._utt_ids):
            raise ValueError(
                f"wav.scp parsing disagreed for split {split!r}: "
                f"load_utt_ids found {len(self._utt_ids)} utterance id(s) "
                f"but _load_wav_paths found {len(self._wav_paths)} audio "
                "path(s). load_utt_ids tolerates an id with no path; "
                "_load_wav_paths does not. Check wav.scp for a malformed "
                "line."
            )
        refs = load_reference_texts(split)
        self._texts = [refs[utt_id] for utt_id in self._utt_ids]
        # A corpus prepared before these files existed must still read. The
        # inference path points at exactly such a tree and cannot be rewritten.
        na = _CONFIG["na_symbol"]
        self._text_prev = self._optional_texts(split, "text.prev", na)
        self._text_ctc = self._optional_texts(split, "text.ctc", na)
        os.environ[_CURRENT_SPLIT_ENV] = split

    def _optional_texts(self, split: str, filename: str, default: str):
        """Return one text per utterance, or ``default`` where the file is absent.

        Args:
            split: Split name.
            filename: File inside the split directory.
            default: Value for every utterance when the file does not exist.

        Returns:
            A list aligned with ``self._utt_ids``.
        """
        if not (_split_dir(split) / filename).is_file():
            return [default] * len(self._utt_ids)
        texts = load_reference_texts(split, filename=filename)
        return [texts.get(utt_id, default) for utt_id in self._utt_ids]

    def __len__(self) -> int:
        """Return the number of utterance groups in the split."""
        return len(self._utt_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one sample.

        Args:
            idx: Position in ``wav.scp`` order.

        Returns:
            ``speech`` as a float32 mono waveform and the three text fields
            ``text``, ``text_prev`` and ``text_ctc``. No other field: a recipe
            sample must carry only what the model and preprocessor accept.
        """
        array, _sample_rate = sf.read(str(self._wav_paths[int(idx)]))
        if array.ndim > 1:
            array = array[:, 0]
        return {
            "speech": np.asarray(array, dtype=np.float32),
            "text": self._texts[int(idx)],
            "text_prev": self._text_prev[int(idx)],
            "text_ctc": self._text_ctc[int(idx)],
        }


# Cache for current_utt_id, keyed by the split it was computed for, so a long
# test set does not re-read and re-parse wav.scp once per utterance. Reset
# whenever the recorded split changes.
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
        idx: Position in the recorded split's ``wav.scp`` order.

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
            "inside an inference run, after that dataset exists."
        )
    if split != _CACHED_SPLIT:
        _CACHED_UTT_IDS = load_utt_ids(split)
        _CACHED_SPLIT = split
    return _CACHED_UTT_IDS[int(idx)]
