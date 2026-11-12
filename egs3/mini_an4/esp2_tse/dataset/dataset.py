"""Read prepared mini AN4 TSE manifests through the raw sample contract."""

import json
from pathlib import Path

from egs3.mini_an4.esp2_tse.dataset.builder import fixture_config
from espnet3.components.data.base_dataset import BaseDataset
from espnet3.contract import Field


class MiniTSEDataset(BaseDataset):
    """Return native TSE training fields, adding IDs only for evaluation."""

    def __init__(self, split, recipe_dir=None, include_ids=False):
        """Read one prepared split, optionally retaining evaluation IDs."""
        cfg = fixture_config()
        if split not in cfg.splits:
            raise ValueError(f"Unknown split {split!r}; expected {list(cfg.splits)}")
        root = (
            Path(recipe_dir)
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        path = root / cfg.data / f"{split}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run create_dataset")
        self.rows = json.loads(path.read_text())
        self.include_ids = include_ids
        self.fields = tuple(
            Field(key, "audio")
            for key in (
                "speech_mix",
                "speech_ref1",
                "speech_ref2",
                "enroll_ref1",
                "enroll_ref2",
            )
        )
        if include_ids:
            self.fields += (Field("utt_id", "text"),)

    def __len__(self):
        """Return the number of mixtures."""
        return len(self.rows)

    def __getitem__(self, idx):
        """Return only the fields consumed by the selected pipeline."""
        return {
            key: value
            for key, value in self.rows[idx].items()
            if self.include_ids or key != "utt_id"
        }
