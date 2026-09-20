"""Manifest-backed full AN4 dataset."""

from pathlib import Path

import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from egs3.an4.asr.dataset.builder import RECIPE_ROOT, SPLITS, read_manifest


class An4Dataset(TorchDataset):
    """Read prepared audio without downloading or mutating data in workers.

    Args:
        split: One of train, valid, or test.
        recipe_dir: Directory containing the prepared data/manifest directory.
    """

    def __init__(self, split: str, recipe_dir: str | Path | None = None):
        """Load the ordered manifest for one split."""
        if split not in SPLITS:
            raise ValueError(f"Unknown AN4 split: {split}")
        root = Path(recipe_dir or RECIPE_ROOT).resolve()
        if not (root / "data/manifest/build.json").is_file():
            raise FileNotFoundError("Run the create_dataset stage before loading AN4")
        self.entries = read_manifest(root / f"data/manifest/{split}.tsv")

    def __len__(self):
        """Return the number of prepared utterances, including speed variants."""
        return len(self.entries)

    def __getitem__(self, index):
        """Return only the waveform and text accepted by the ASR preprocessor."""
        _, path, text = self.entries[int(index)]
        speech, rate = sf.read(path, dtype="float32")
        if rate != 16000 or speech.ndim != 1:
            raise ValueError(f"Expected mono 16 kHz prepared audio: {path}")
        return {"speech": speech, "text": text}
