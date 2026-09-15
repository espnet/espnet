"""AudioSet-2M dataset for BEATs pre-training."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from espnet3.systems.ssl.target_reader import BeatsTargetReader
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]
_BUILDER_CFG = _CONFIG["builder"]

_KNOWN_SPLITS = {str(split) for split in _DATASET_CFG["supported_splits"]}


def _read_manifest(manifest_path: Path) -> tuple[list[str], list[str]]:
    """Read ``utt_id<TAB>audio_path<TAB>num_samples`` rows."""
    utt_ids: list[str] = []
    audio_paths: list[str] = []
    with manifest_path.open("r", encoding="utf-8") as reader:
        for line in reader:
            if not line.strip():
                continue
            utt_id, audio_path, _ = line.rstrip("\n").split("\t", maxsplit=2)
            utt_ids.append(utt_id)
            audio_paths.append(audio_path)
    if not utt_ids:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")
    return utt_ids, audio_paths


class AudioSetDataset(TorchDataset):
    """AudioSet clips (and optional BEATs targets) in manifest order.

    Samples have the following fields:

    - ``speech``: ``float32`` waveform ``(num_samples,)`` in ``[-1, 1]`` when
      ``feats_path`` is ``None``, or a 128-bin fbank ``(num_frames, 128)`` read
      from ``feats_path`` (set ``waveform_input: false`` in the training config).
    - ``target``: space-separated token ids for this item, only when
      ``target_path`` is given (encoder training). Tokenization and tokenizer
      training use datasets without targets.

    Args:
        split: ``train`` or ``eval``.
        recipe_dir: Recipe root holding ``data/manifest/<split>.tsv``. Defaults
            to this recipe directory.
        target_path: Index-keyed ``target.scp`` written by the ``infer`` stage
            for the same split.
        feats_path: Optional Kaldi ``feats.scp`` keyed by the manifest utterance
            ids, for example an egs2 ``dump/fbank/<split>/feats.scp``. Reading
            one needs the optional ``kaldiio`` dependency
            (``pip install "espnet[kaldiio]"``).

    Raises:
        ValueError: If ``split`` is unknown.
        FileNotFoundError: If the manifest is missing (run ``create_dataset``).
        ImportError: If ``feats_path`` is given but ``kaldiio`` is not installed.
        KeyError: If ``feats_path`` lacks a manifest utterance id.
    """

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        target_path: str | Path | None = None,
        feats_path: str | Path | None = None,
    ) -> None:
        """Load the manifest, optional features index, and optional targets."""
        if split not in _KNOWN_SPLITS:
            known = ", ".join(sorted(_KNOWN_SPLITS))
            raise ValueError(f"Unknown split '{split}'. Expected one of: {known}")
        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        data_root = recipe_root / _BUILDER_CFG["data_path"]
        manifest_path = data_root / _BUILDER_CFG["manifest_paths"][split]
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. Run the create_dataset stage."
            )
        self.split = split
        self._utt_ids, self._audio_paths = _read_manifest(manifest_path)
        self._feats = None
        if feats_path is not None:
            # Optional dependency: the default waveform path does not need it.
            try:
                import kaldiio
            except ImportError as err:
                raise ImportError(
                    "Reading feats_path requires kaldiio: "
                    'pip install "espnet[kaldiio]".'
                ) from err
            self._feats = kaldiio.load_scp(str(feats_path))
            missing = next((u for u in self._utt_ids if u not in self._feats), None)
            if missing is not None:
                raise KeyError(f"{feats_path} has no features for '{missing}'.")
        self._targets = (
            BeatsTargetReader(target_path, num_items=len(self._utt_ids))
            if target_path is not None
            else None
        )

    def __len__(self) -> int:
        """Return the number of clips."""
        return len(self._utt_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return ``{"speech": ...}`` plus ``"target"`` when targets are set."""
        idx = int(idx)
        if self._feats is not None:
            speech = np.asarray(self._feats[self._utt_ids[idx]], dtype=np.float32)
        else:
            speech, _ = sf.read(self._audio_paths[idx], dtype="float32")
        sample = {"speech": speech}
        if self._targets is not None:
            sample["target"] = self._targets[idx]
        return sample
