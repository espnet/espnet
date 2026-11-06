"""LibriTTS codec dataset backed by recipe TSV manifests.

Waveform-only: returns ``{"speech": np.float32 waveform}`` samples, which the
training preprocessor (``CodecPreprocessor``, declared in the template
training config) hands to the ESPnet2 codec model as ``audio``. The item
fields are declared on the class (``BaseDataset.fields``) and checked on the
first item of every instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from egs3.libritts.esp2_codec.dataset.builder import LibriTTSBuilder
from espnet3.components.data.base_dataset import BaseDataset
from espnet3.contract import Field
from espnet3.utils.config_utils import load_config_with_defaults

# ---------------------------------------------------------------------------
# Module-level config loading
# ---------------------------------------------------------------------------

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]
_BUILDER_CFG = _CONFIG["builder"]

_SPLIT_MANIFEST_PATHS: dict[str, str] = {
    str(split): str(relpath)
    for split, relpath in _DATASET_CFG["split_manifest_paths"].items()
}


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestEntry:
    """One manifest row: utterance id, wav path, transcript, and speaker id."""

    utt_id: str
    wav_path: Path
    text: str
    sid: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_manifest(path: Path) -> list[ManifestEntry]:
    """Read ``utt_id<TAB>wav_path<TAB>text<TAB>sid`` lines as manifest entries."""
    entries: list[ManifestEntry] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id, wav_path, text, sid = line.split("\t", maxsplit=3)
            entries.append(
                ManifestEntry(
                    utt_id=utt_id,
                    wav_path=Path(wav_path),
                    text=text,
                    sid=int(sid),
                )
            )
    if not entries:
        raise RuntimeError(f"Manifest is empty: {path}")
    return entries


# ---------------------------------------------------------------------------
# Public dataset class
# ---------------------------------------------------------------------------


class LibriTTSCodecDataset(BaseDataset):
    """LibriTTS dataset returning waveform-only samples for codec training.

    Reads one manifest TSV written by
    :class:`~egs3.libritts.esp2_codec.dataset.builder.LibriTTSBuilder` and returns
    ``{"speech": np.float32 waveform}`` per utterance. In ``inference`` mode the
    sample also carries ``utt_id``, which heads every inference output, and
    ``wav_path``, the ground-truth file the ``measure`` stage scores against
    (bound as ``dataset:wav_path`` in ``conf/metrics.yaml``). Both are
    declared optional because the training splits do not carry them.

    Args:
        split: Manifest split to load; one of the keys under
            ``dataset.split_manifest_paths`` in ``dataset/config.yaml``
            (``train``, ``valid``, ``test``). Ignored when ``manifest_path``
            is given.
        recipe_dir: Optional recipe root. When omitted, defaults to the
            recipe directory inferred from this module.
        manifest_path: Optional manifest override. Relative paths resolve
            against ``recipe_dir``.
        inference: Add ``utt_id`` and ``wav_path`` to each sample.

    Raises:
        RuntimeError: If the manifests have not been built yet, or the
            resolved manifest is empty.
        ValueError: If ``split`` is unknown and no ``manifest_path`` is given.
        FileNotFoundError: If the resolved manifest file does not exist.

    Examples:
        ```python
        dataset = LibriTTSCodecDataset(split="train")
        sorted(dataset[0].keys())        # -> ['speech']

        test = LibriTTSCodecDataset(split="test", inference=True)
        sorted(test[0].keys())           # -> ['speech', 'utt_id', 'wav_path']
        ```
    """

    fields = (
        Field("speech", "audio"),
        Field("utt_id", "text", optional=True),
        Field("wav_path", "text", optional=True),
    )

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        inference: bool = False,
    ) -> None:
        """Load one manifest split into memory.

        Only the manifest rows are read here; waveforms are read lazily in
        :meth:`__getitem__`.

        Args:
            split: Manifest split to load; one of the keys under
                ``dataset.split_manifest_paths`` in ``dataset/config.yaml``
                (``train``, ``valid``, ``test``). Ignored when
                ``manifest_path`` is given.
            recipe_dir: Optional recipe root. When omitted, defaults to the
                recipe directory inferred from this module.
            manifest_path: Optional manifest override. Relative paths resolve
                against ``recipe_dir``.
            inference: Add ``utt_id`` and ``wav_path`` to each sample.

        Raises:
            RuntimeError: If the manifests have not been built yet (checked
                only when no ``manifest_path`` is given), or the resolved
                manifest is empty.
            ValueError: If ``split`` is unknown and no ``manifest_path`` is
                given.
            FileNotFoundError: If the resolved manifest file does not exist.

        Examples:
            ```python
            train = LibriTTSCodecDataset(split="train")
            test = LibriTTSCodecDataset(split="test", inference=True)

            # A manifest outside the configured splits:
            other = LibriTTSCodecDataset(
                split="test",
                manifest_path="data/manifest/custom.tsv",
            )
            ```
        """
        self.split = split
        self.inference = inference
        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        self.data_dir = recipe_root / _BUILDER_CFG["data_path"]

        if manifest_path is not None:
            # An explicit manifest is an override: it may live outside the
            # configured splits, so the recipe build check does not apply.
            resolved_manifest = Path(manifest_path)
            if not resolved_manifest.is_absolute():
                resolved_manifest = (recipe_root / resolved_manifest).resolve()
        else:
            builder = LibriTTSBuilder()
            if not builder.is_built(recipe_dir=recipe_root):
                raise RuntimeError(
                    "Dataset is not built yet. Run create_dataset stage first."
                )
            if split not in _SPLIT_MANIFEST_PATHS:
                raise ValueError(
                    f"Unknown split '{split}'. Expected one of "
                    f"{sorted(_SPLIT_MANIFEST_PATHS)}"
                )
            resolved_manifest = self.data_dir / _SPLIT_MANIFEST_PATHS[split]
        if not resolved_manifest.is_file():
            raise FileNotFoundError(f"Manifest not found: {resolved_manifest}")
        self.manifest_path = resolved_manifest
        self._entries = _read_manifest(resolved_manifest)
        self._by_utt_id = {entry.utt_id: entry for entry in self._entries}

    def __len__(self) -> int:
        """Return the number of utterances in the loaded manifest.

        Returns:
            Number of manifest rows.

        Examples:
            ```python
            len(LibriTTSCodecDataset(split="test"))   # -> 4837
            ```
        """
        return len(self._entries)

    def keys(self) -> list[str]:
        """Return utterance IDs in manifest order.

        Returns:
            List of utterance IDs, in the same order as integer indexing,
            so ``dataset[dataset.keys()[i]] == dataset[i]``.

        Examples:
            ```python
            dataset = LibriTTSCodecDataset(split="test")
            dataset.keys()[:2]   # -> ['1089-134686-000002-000000', ...]
            ```
        """
        return [entry.utt_id for entry in self._entries]

    def __getitem__(self, idx: int | str) -> dict[str, Any]:
        """Return one sample, read from disk on access.

        Args:
            idx: Either a positional index, or an utterance ID as returned
                by :meth:`keys`.

        Returns:
            Dict with the waveform under ``speech`` as a 1-D float32 numpy
            array. In ``inference`` mode it also carries ``utt_id`` and
            ``wav_path``, both ``str``.

        Raises:
            KeyError: If a string *idx* is not a known utterance ID.
            IndexError: If an integer *idx* is out of range.

        Examples:
            ```python
            dataset = LibriTTSCodecDataset(split="test")
            dataset[0]["speech"].dtype            # -> dtype('float32')
            dataset[dataset.keys()[0]]["speech"]  # same sample, by utt_id
            ```
        """
        if isinstance(idx, str):
            entry = self._by_utt_id[idx]
        else:
            entry = self._entries[int(idx)]
        speech, _ = sf.read(str(entry.wav_path))
        sample: dict[str, Any] = {"speech": np.asarray(speech, dtype=np.float32)}
        if self.inference:
            sample.update(
                {
                    "utt_id": str(entry.utt_id),
                    "wav_path": str(entry.wav_path),
                }
            )
        return sample
