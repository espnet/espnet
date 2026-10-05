"""ACE-Opencpop dataset read from the manifests written by ``create_dataset``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from espnet2.fileio.npy_scp import NpyScpReader

from .builder import manifest_path as default_manifest_path

logger = logging.getLogger(__name__)

# Relative manifest paths are taken from the recipe directory.
RECIPE_DIR = Path(__file__).resolve().parents[1]


def read_manifest(path: Path) -> list[list[str]]:
    """Read the rows of a manifest.

    Each line holds ``utt_id``, ``wav_path``, ``text``, ``singer``, ``label``
    and ``score``, separated by tabs.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(line.rstrip("\n").split("\t"))
    if len(rows) == 0:
        raise RuntimeError(f"Manifest is empty: {path}")
    return rows


def parse_label(label: str) -> tuple[np.ndarray, list[str]]:
    """Turn ``st et phn st et phn ...`` into ``(times, phones)``.

    Same as ``espnet2.train.dataset.label_loader``: an ``(N, 2)`` array of
    phoneme start and end times, rounded to float32 as there, and the list of
    phonemes.
    """
    fields = label.split()
    num_phones = len(fields) // 3
    times = np.zeros((num_phones, 2))
    phones = []
    for i in range(num_phones):
        times[i, 0] = np.float32(fields[3 * i])
        times[i, 1] = np.float32(fields[3 * i + 1])
        phones.append(fields[3 * i + 2])
    return times, phones


def parse_score(score: str) -> tuple[int, list[list[Any]]]:
    """Turn a JSON score into ``(tempo, notes)``, as ``score_loader`` does."""
    score_dict = json.loads(score)
    return score_dict["tempo"], score_dict["note"]


class ACEOpencpopDataset(TorchDataset):
    """ACE-Opencpop dataset returning the inputs of ``GANSVSTask``.

    Each sample carries the fields that ``SVSPreprocessor`` turns into model
    inputs: ``text`` (phonemes), ``singing`` (waveform), ``label`` (phoneme
    timing), ``score`` (music score) and ``sids`` (singer id). Once
    ``collect_stats`` has dumped them to ``collect_feats_dir``, ``feats`` and
    ``pitch`` are added too, as ``--write_collected_feats true`` does in egs2.

    Example:
        Declared in the training config through ``DataOrganizer``:

        .. code-block:: yaml

            dataset:
              train:
                - data_src_args:
                    split: train
                    manifest_path: ${remove_long_short.save_path}/train.tsv
                    collect_feats_dir: ${stats_dir}/train/collect_feats
    """

    def __init__(
        self,
        split: str,
        manifest_path: str | Path | None = None,
        collect_feats_dir: str | Path | None = None,
        inference: bool = False,
    ) -> None:
        """Load the manifest of one split.

        Args:
            split: ``train``, ``valid`` or ``test``. Selects the manifest
                written by ``create_dataset`` unless *manifest_path* is given.
            manifest_path: Manifest to read instead, such as a filtered one.
            collect_feats_dir: Directory of the features dumped by
                ``collect_stats``. Not used until they exist.
            inference: Return the inputs of ``SingingGenerate`` instead, plus
                ``utt_id``, ``wav_path`` and ``raw_text`` for the infer stage.

        Raises:
            ValueError: If *split* is unknown and no *manifest_path* is given.
            FileNotFoundError: If the manifest is missing.
        """
        if manifest_path is None:
            manifest_path = default_manifest_path(RECIPE_DIR, split)
        manifest_path = RECIPE_DIR / manifest_path
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. "
                "Run the create_dataset stage first."
            )
        self.rows = read_manifest(manifest_path)
        self.inference = inference

        # The same dataset config serves collect_stats, which writes the
        # features, and train, which reads them back.
        self.feats_reader = None
        self.pitch_reader = None
        if collect_feats_dir is not None:
            feats_scp = Path(collect_feats_dir) / "feats.scp"
            pitch_scp = Path(collect_feats_dir) / "pitch.scp"
            if feats_scp.is_file() and pitch_scp.is_file():
                self.feats_reader = NpyScpReader(feats_scp)
                self.pitch_reader = NpyScpReader(pitch_scp)
            else:
                logger.warning(
                    "%s has no collected features; fbank and pitch will be "
                    "computed on the fly.",
                    collect_feats_dir,
                )

    def __len__(self) -> int:
        """Return the number of utterances."""
        return len(self.rows)

    def __getitem__(self, idx: int | str) -> dict[str, Any]:
        """Return the sample of the ``idx``-th utterance."""
        idx = int(idx)
        utt_id, wav_path, text, singer, label, score = self.rows[idx]
        sids = np.array([int(singer)], dtype=np.int64)

        if self.inference:
            return {
                "text": {"label": parse_label(label), "score": parse_score(score)},
                "sids": sids,
                "utt_id": utt_id,
                "wav_path": wav_path,
                "raw_text": text,
            }

        singing, _ = sf.read(wav_path, dtype="float32")
        sample = {
            "text": text,
            "label": parse_label(label),
            "score": parse_score(score),
            "sids": sids,
            "singing": singing,
        }
        if self.feats_reader is not None:
            # collect_stats keys its dumps by the dataset index (see
            # espnet3.components.data.collect_stats.collect_stats_batch), so
            # the manifest must be the one the features were collected on.
            feats = self.feats_reader[str(idx)]
            pitch = self.pitch_reader[str(idx)]
            sample["feats"] = np.asarray(feats, dtype=np.float32)
            sample["pitch"] = np.asarray(pitch, dtype=np.float32)
        return sample
