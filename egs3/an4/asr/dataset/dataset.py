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
        """Load an ordered manifest without reading its waveforms.

        Args:
            split: Prepared ``train``, ``valid`` or ``test`` split.
            recipe_dir: Root containing ``data/manifest``; None uses this recipe.

        Returns:
            None. Manifest rows are stored in ``self.entries``.

        Raises:
            ValueError: The split is unknown or its manifest is invalid.
            FileNotFoundError: Data preparation has not produced the manifest.

        Examples:
            After the create_dataset stage:

            >>> dataset = An4Dataset("train", recipe_dir=".")
        """
        if split not in SPLITS:
            raise ValueError(f"Unknown AN4 split: {split}")
        root = Path(recipe_dir or RECIPE_ROOT).resolve()
        if not (root / "data/manifest/build.json").is_file():
            raise FileNotFoundError("Run the create_dataset stage before loading AN4")
        self.entries = read_manifest(root / f"data/manifest/{split}.tsv")

    def __len__(self):
        """Return the number of selected recordings.

        Args:
            None.

        Returns:
            Integer number of indexed utterances, including recording variants.

        Examples:
            After loading a prepared split:

            >>> count = len(dataset)
        """
        return len(self.entries)

    def __getitem__(self, index):
        """Read one waveform and its transcript for ASR preprocessing.

        Args:
            index: Integer position in the prepared manifest's existing order.

        Returns:
            Dictionary with mono 16 kHz float32 ``speech`` and string ``text``.
            The utterance ID and waveform path stay in ``entries``.

        Raises:
            IndexError: The requested manifest position is out of range.
            ValueError: Audio does not match the expected prepared format.

        Examples:
            After loading a prepared split:

            >>> sample = dataset[0]
            >>> sorted(sample)
            ['speech', 'text']
        """
        _, path, text = self.entries[int(index)]
        speech, rate = sf.read(path, dtype="float32")
        if rate != 16000 or speech.ndim != 1:
            raise ValueError(f"Expected mono 16 kHz prepared audio: {path}")
        return {"speech": speech, "text": text}
