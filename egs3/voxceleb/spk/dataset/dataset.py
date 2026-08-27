"""VoxCeleb dataset backed by the manifests written by the builder."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]

_KNOWN_SPLITS = {str(split) for split in _DATASET_CFG["supported_splits"]}
_SAMPLE_RATE = int(_DATASET_CFG["sample_rate"])


def _read_scp(path: Path) -> Dict[str, str]:
    """Read a two-column SCP file into a dict."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Run the `create_dataset` stage first."
        )
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, value = line.split(maxsplit=1)
        entries[key] = value.strip()
    return entries


def _subsample_trials(
    trials: List[Tuple[int, str, str]], num_trials: int | None
) -> List[Tuple[int, str, str]]:
    """Take an evenly strided subset of the trials, per class.

    Striding each class separately keeps the target/nontarget ratio of the full
    list, so an EER measured on the subset stays comparable across epochs and
    to the full evaluation. A single stride over the list would not: official
    trial lists alternate target and nontarget lines.

    Args:
        trials: Full trial list as `(label, utt1, utt2)` entries.
        num_trials: Number of trials to keep, or ``None`` to keep them all.

    Returns:
        The selected trials, in their original order.
    """
    if num_trials is None or num_trials >= len(trials):
        return trials

    keep: List[int] = []
    for label in (0, 1):
        indices = [i for i, trial in enumerate(trials) if trial[0] == label]
        if not indices:
            continue
        n_keep = max(1, round(num_trials * len(indices) / len(trials)))
        stride = len(indices) / n_keep
        keep.extend(indices[int(i * stride)] for i in range(n_keep))
    return [trials[i] for i in sorted(keep)]


class VoxCelebDataset(TorchDataset):
    """VoxCeleb utterances, or verification trials over them.

    The dataset serves two shapes, chosen by whether ``trials`` is set:

    - Without ``trials`` it yields single utterances with their speaker string,
      which is what speaker classification training consumes.
    - With ``trials`` it yields the two utterances of each trial together with
      a target/nontarget flag, which is what validation and scoring consume.

    Args:
        split: Split name such as ``voxceleb2_dev`` or ``voxceleb1_test``.
            To train on several splits at once, list them under
            ``dataset.train`` rather than looking for a combined split.
        trials: Name of a trial list prepared by the builder, such as
            ``vox1_o``. Leave unset for utterance mode.
        num_trials: Optional cap on the number of trials. Trials are strided
            evenly across the list, which keeps the target/nontarget balance.
            Useful to keep per-epoch validation short.
        recipe_dir: Optional recipe root. Defaults to this recipe.

    Raises:
        ValueError: If ``split`` is unknown, or ``num_trials`` is not positive.
        FileNotFoundError: If the manifests have not been built yet.

    Examples:
        >>> train = VoxCelebDataset(split="voxceleb2_dev")
        >>> sorted(train[0].keys())
        ['speech', 'spk_labels']
        >>> valid = VoxCelebDataset(split="voxceleb1_test", trials="vox1_o")
        >>> sorted(valid[0].keys())
        ['speech', 'speech2', 'spk_labels']
    """

    def __init__(
        self,
        split: str,
        trials: str | None = None,
        num_trials: int | None = None,
        recipe_dir: str | Path | None = None,
    ) -> None:
        """Load the manifests of one split and index its items."""
        self.split = str(split)
        if self.split not in _KNOWN_SPLITS:
            known = ", ".join(sorted(_KNOWN_SPLITS))
            raise ValueError(f"Unknown split '{self.split}'. Expected one of: {known}")
        if num_trials is not None and int(num_trials) <= 0:
            raise ValueError(f"num_trials must be positive, got {num_trials}")

        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        data_root = recipe_root / str(_DATASET_CFG["data_path"])
        split_dir = data_root / self.split

        self.wav_scp = _read_scp(split_dir / "wav.scp")
        self.trials = trials

        if trials is None:
            self.utt2spk = _read_scp(split_dir / "utt2spk")
            self._utt_ids = sorted(self.wav_scp)
            self._trials: List[Tuple[int, str, str]] = []
        else:
            self._utt_ids = []
            self._trials = _subsample_trials(
                self._read_trials(split_dir / f"{trials}.trials"), num_trials
            )

    @staticmethod
    def _read_trials(path: Path) -> List[Tuple[int, str, str]]:
        """Read a `<label> <utt1> <utt2>` trial list."""
        if not path.is_file():
            raise FileNotFoundError(
                f"Trial list not found: {path}. Run the `create_dataset` stage first."
            )
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            label, utt1, utt2 = line.split()
            entries.append((int(label), utt1, utt2))
        return entries

    def _read_audio(self, utt_id: str) -> np.ndarray:
        """Read one utterance as a mono float32 waveform."""
        audio, sample_rate = sf.read(self.wav_scp[utt_id], dtype="float32")
        if sample_rate != _SAMPLE_RATE:
            raise RuntimeError(
                f"{utt_id} is sampled at {sample_rate} Hz but the recipe expects "
                f"{_SAMPLE_RATE} Hz. Resample the corpus before training."
            )
        if audio.ndim > 1:
            audio = audio[:, 0]
        return audio

    def __len__(self) -> int:
        """Return the number of utterances, or of trials in trial mode."""
        return len(self._trials) if self.trials else len(self._utt_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one utterance, or one trial pair in trial mode."""
        idx = int(idx)
        if self.trials:
            label, utt1, utt2 = self._trials[idx]
            return {
                "speech": self._read_audio(utt1),
                "speech2": self._read_audio(utt2),
                "spk_labels": str(label),
            }

        utt_id = self._utt_ids[idx]
        return {
            "speech": self._read_audio(utt_id),
            "spk_labels": self.utt2spk[utt_id],
        }
