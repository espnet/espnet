"""Read indexed VOiCES audio lazily without changing source files."""

from pathlib import Path

import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from .builder import RECIPE_ROOT, SPLITS, read_manifest


class VoicesDataset(TorchDataset):
    """Load a prepared devkit split, optionally selecting a recording condition.

    Args:
        split: ``train``, ``valid``, or ``test``.
        recipe_dir: Directory containing ``data/manifest``.
        condition: ``all`` (source + distant), ``distant``, or ``source``.
        limit: Optional prefix length for explicitly labeled smoke tests.

    Raises:
        ValueError: A split, condition, or limit is invalid.
        FileNotFoundError: Dataset preparation has not completed.
    """

    def __init__(self, split, recipe_dir=None, condition="all", limit=None):
        """Read only the small manifest; waveforms are loaded by __getitem__."""
        if split not in SPLITS or condition not in ("all", "distant", "source"):
            raise ValueError(f"Unknown VOiCES split/condition: {split}/{condition}")
        root = Path(recipe_dir or RECIPE_ROOT).resolve() / "data/manifest"
        if not (root / "build.json").is_file():
            raise FileNotFoundError("Run create_dataset before loading VOiCES")
        self.entries = [
            row
            for row in read_manifest(root / f"{split}.tsv")
            if condition == "all" or row["condition"] == condition
        ]
        if limit is not None:
            if int(limit) <= 0:
                raise ValueError("limit must be positive")
            self.entries = self.entries[: int(limit)]

    def __len__(self):
        """Return the selected number of recordings."""
        return len(self.entries)

    def __getitem__(self, index):
        """Return only the waveform and text accepted by CommonPreprocessor."""
        entry = self.entries[int(index)]
        speech, rate = sf.read(entry["path"], dtype="float32")
        if rate != 16000 or speech.ndim != 1 or len(speech) != entry["samples"]:
            raise ValueError(f"Audio changed since preparation: {entry['path']}")
        return {"speech": speech, "text": entry["text"]}
