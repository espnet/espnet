"""Read indexed VOiCES audio lazily without changing source files."""

from pathlib import Path

import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from egs3.voices.esp2_asr.dataset.builder import RECIPE_ROOT, SPLITS, read_manifest


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
        """Load an ordered manifest without reading its waveforms.

        Args:
            split: Prepared ``train``, ``valid`` or ``test`` split.
            recipe_dir: Root containing ``data/manifest``; None uses this recipe.
            condition: ``all``, ``source`` or ``distant`` recording selection.
            limit: Optional positive prefix length for smoke tests; None keeps all.

        Returns:
            None. Manifest rows are stored in ``self.entries``.

        Raises:
            ValueError: The split, condition or limit is invalid.
            FileNotFoundError: Data preparation has not produced the manifest.

        Examples:
            After the create_dataset stage:

            >>> dataset = VoicesDataset("train", recipe_dir=".")

            Select at most four distant recordings, preserving manifest order:

            >>> dataset = VoicesDataset(
            ...     "test", recipe_dir=".", condition="distant", limit=4
            ... )
        """
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
            index: Integer position in the ordered, optionally filtered manifest.

        Returns:
            Dictionary with mono 16 kHz float32 ``speech`` and string ``text``.
            Metadata such as speaker ID stays in ``entries``.

        Raises:
            IndexError: The requested manifest position is out of range.
            ValueError: Audio does not match the expected prepared format.

        Examples:
            After loading a prepared split:

            >>> sample = dataset[0]
            >>> sorted(sample)
            ['speech', 'text']
        """
        entry = self.entries[int(index)]
        speech, rate = sf.read(entry["path"], dtype="float32")
        if rate != 16000 or speech.ndim != 1 or len(speech) != entry["samples"]:
            raise ValueError(f"Audio changed since preparation: {entry['path']}")
        return {"speech": speech, "text": entry["text"]}
