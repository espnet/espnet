"""Mini AN4 audio with optional BEATs targets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset as TorchDataset

from egs3.mini_an4.asr.dataset.dataset import MiniAn4Dataset
from espnet3.systems.ssl.target_reader import BeatsTargetReader


class MiniAn4SSLDataset(TorchDataset):
    """Mini AN4 waveforms for BEATs pre-training, ignoring transcripts.

    Samples contain ``speech`` (``float32`` waveform ``(num_samples,)``) and,
    when ``target_path`` is given, ``target`` (space-separated token ids read
    from an index-keyed ``target.scp`` written by the ``infer`` stage).

    Args:
        split: ``train``, ``valid``, or ``test``.
        recipe_dir: Recipe root where the Mini AN4 archive is extracted and
            manifests are built. Defaults to this recipe directory.
        target_path: Optional index-keyed ``target.scp`` for the same split.
    """

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        target_path: str | Path | None = None,
    ) -> None:
        """Load Mini AN4 and the optional targets."""
        if recipe_dir is None:
            recipe_dir = Path(__file__).resolve().parents[1]
        self._dataset = MiniAn4Dataset(split=split, recipe_dir=recipe_dir)
        self._targets = (
            BeatsTargetReader(target_path, num_items=len(self._dataset))
            if target_path is not None
            else None
        )

    def __len__(self) -> int:
        """Return the number of utterances."""
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return ``{"speech": ...}`` plus ``"target"`` when targets are set."""
        sample = {"speech": self._dataset[int(idx)]["speech"]}
        if self._targets is not None:
            sample["target"] = self._targets[int(idx)]
        return sample
