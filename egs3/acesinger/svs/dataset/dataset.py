"""ACE-Opencpop dataset backed by recipe TSV manifests."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from egs3.acesinger.svs.dataset.builder import ACEOpencpopBuilder
from espnet2.fileio.npy_scp import NpyScpReader
from espnet3.utils.config_utils import load_config_with_defaults

# ---------------------------------------------------------------------------
# Module-level config loading
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

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
    """One manifest row: id, wav path, phoneme text, singer, label and score."""

    utt_id: str
    wav_path: Path
    text: str
    sid: int
    label: str
    score: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_manifest(path: Path) -> list[ManifestEntry]:
    """Read ``utt_id<TAB>wav<TAB>text<TAB>singer<TAB>label<TAB>score`` lines."""
    entries: list[ManifestEntry] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            utt_id, wav_path, text, sid, label, score = line.split("\t", maxsplit=5)
            entries.append(
                ManifestEntry(
                    utt_id=utt_id,
                    wav_path=Path(wav_path),
                    text=text,
                    sid=int(sid),
                    label=label,
                    score=score,
                )
            )
    if not entries:
        raise RuntimeError(f"Manifest is empty: {path}")
    return entries


def _parse_label(label: str) -> tuple[np.ndarray, list[str]]:
    """Turn ``st et phn ...`` into what ``SVSPreprocessor`` reads as ``label``.

    This is the output format of ``espnet2.train.dataset.label_loader``: an
    ``(N, 2)`` array of phoneme start/end times and the list of phonemes.
    """
    fields = label.split()
    times = np.array(
        [[float(fields[i]), float(fields[i + 1])] for i in range(0, len(fields), 3)]
    )
    phones = [fields[i + 2] for i in range(0, len(fields), 3)]
    return times, phones


def _parse_score(score: str) -> tuple[int, list[list[Any]]]:
    """Turn the JSON score into ``(tempo, notes)`` as ``score_loader`` does."""
    obj = json.loads(score)
    return int(obj["tempo"]), obj["note"]


# ---------------------------------------------------------------------------
# Public dataset class
# ---------------------------------------------------------------------------


class ACEOpencpopDataset(TorchDataset):
    """ACE-Opencpop dataset returning the inputs of ``GANSVSTask``.

    Training samples carry the raw fields that ``SVSPreprocessor`` turns into
    model inputs:

    - ``text``: space-separated phonemes
    - ``singing``: float32 waveform
    - ``label``: ``(times, phones)`` phoneme alignment
    - ``score``: ``(tempo, notes)`` music score
    - ``sids``: singer id as an ``int64`` array of shape ``(1,)``

    With ``collect_feats_dir`` set, ``feats`` and ``pitch`` dumped by the
    ``collect_stats`` stage are added so training does not recompute them,
    the same as ``--write_collected_feats true`` in ``egs2/TEMPLATE/svs1``.
    Those dumps are keyed by dataset index, so the manifest must be the one
    the stats were collected on.

    Examples:
        Declared from ``conf/training.yaml`` through ``DataOrganizer``:
        ```yaml
        dataset:
          train:
            - data_src_args:
                split: train
                manifest_path: ${remove_long_short.save_path}/train.tsv
                collect_feats_dir: ${stats_dir}/train/collect_feats
        ```
    """

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        load_singing: bool = True,
        collect_feats_dir: str | Path | None = None,
        inference: bool = False,
    ) -> None:
        """Load one split's manifest and record what each sample should carry.

        Args:
            split: Split name (``train`` / ``valid`` / ``test``). Selects the
                default manifest from ``dataset/config.yaml`` when
                *manifest_path* is not given.
            recipe_dir: Recipe root. Defaults to this file's recipe directory.
            manifest_path: Explicit manifest to read, absolute or relative to
                *recipe_dir*. Takes precedence over *split*.
            load_singing: Read the waveform into ``singing``.
            collect_feats_dir: ``collect_feats`` directory written by the
                ``collect_stats`` stage. Adds ``feats`` and ``pitch`` to each
                sample once it exists; before that the model computes them.
            inference: Return ``text`` as the ``{"label", "score"}`` dict that
                ``SingingGenerate`` consumes, plus ``utt_id``, ``wav_path`` and
                ``raw_text`` for the infer and measure stages.

        Raises:
            ValueError: If *split* is unknown and no *manifest_path* was given.
            FileNotFoundError: If the manifest is missing.
            RuntimeError: If the dataset has not been built yet.
        """
        self.split = split
        self.load_singing = load_singing
        self.inference = inference
        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        self.data_dir = recipe_root / _BUILDER_CFG["data_path"]

        if not ACEOpencpopBuilder().is_built(recipe_dir=recipe_root):
            raise RuntimeError(
                "Dataset is not built yet. Run the create_dataset stage first."
            )

        if manifest_path is not None:
            resolved_manifest = Path(manifest_path)
            if not resolved_manifest.is_absolute():
                resolved_manifest = (recipe_root / resolved_manifest).resolve()
        else:
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

        # The same dataset config serves collect_stats and train, so the dumps
        # are optional: before collect_stats has run, the model extracts the
        # features itself.
        self._feats_reader = None
        self._pitch_reader = None
        if collect_feats_dir is not None:
            feats_dir = Path(collect_feats_dir)
            if (feats_dir / "feats.scp").is_file():
                self._feats_reader = NpyScpReader(feats_dir / "feats.scp")
                self._pitch_reader = NpyScpReader(feats_dir / "pitch.scp")
            else:
                logger.warning(
                    "%s has no collected features; fbank and pitch will be "
                    "computed on the fly.",
                    feats_dir,
                )

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self._entries[int(idx)]
        label = _parse_label(entry.label)
        score = _parse_score(entry.score)
        sids = np.array([entry.sid], dtype=np.int64)

        if self.inference:
            return {
                "text": {"label": label, "score": score},
                "sids": sids,
                "utt_id": entry.utt_id,
                "wav_path": str(entry.wav_path),
                "raw_text": entry.text,
            }

        sample: dict[str, Any] = {
            "text": entry.text,
            "label": label,
            "score": score,
            "sids": sids,
        }
        if self.load_singing:
            singing, _ = sf.read(str(entry.wav_path), dtype="float32")
            sample["singing"] = np.asarray(singing, dtype=np.float32)
        if self._feats_reader is not None:
            # collect_stats keys its dumps by the dataset index (see
            # espnet3.components.data.collect_stats.collect_stats_batch).
            key = str(int(idx))
            sample["feats"] = np.asarray(self._feats_reader[key], dtype=np.float32)
            sample["pitch"] = np.asarray(self._pitch_reader[key], dtype=np.float32)
        return sample
