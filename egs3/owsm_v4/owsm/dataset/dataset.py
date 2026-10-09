"""Read path shared by every OWSM sub-dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import soundfile as sf
from datasets import load_from_disk
from torch.utils.data import Dataset as TorchDataset

from egs3.owsm_v4.owsm.dataset.utils import (
    cache_root,
    lang_token,
    sub_dataset_config,
    task_token,
)


def read_span(wav_path: str, start_time: float, end_time: float) -> np.ndarray:
    """Read ``[start_time, end_time)`` from one file in a single open."""
    with sf.SoundFile(wav_path) as handle:
        rate = handle.samplerate
        handle.seek(int(round(start_time * rate)))
        frames = int(round((end_time - start_time) * rate))
        return handle.read(frames=frames, dtype="float32", always_2d=False)


class OWSMDataset(TorchDataset):
    """OWSM samples for one split of one corpus.

    Every sub-dataset writes the same cache columns, so they differ only in
    where that cache lives and which splits exist, both of which are read
    from the subclass's own ``config.yaml``.

    Rows come back in the order they were written: no filtering, sorting or
    reindexing, so ``__len__`` cannot drift from the shape files that
    ``collect_stats`` keys by position in the combined dataset.

    Args:
        split: A name listed in ``SPLITS``.
        recipe_dir: Recipe root, used to resolve a relative cache dir.
        cache: The recipe's ``cache`` block, giving ``cache_dir``.

    Raises:
        ValueError: If ``split`` is unknown.
        FileNotFoundError: If the split has not been built.
    """

    #: Both filled from the sub-dataset's own config.yaml.
    CACHE_SUBDIR: str = ""
    SPLITS: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs) -> None:
        """Read the subclass's ``config.yaml``, as :class:`OWSMBuilder` does."""
        super().__init_subclass__(**kwargs)
        config = sub_dataset_config(cls.__module__)
        if config is None:
            return
        cls.SPLITS = tuple(str(split) for split in config["splits"])
        cls.CACHE_SUBDIR = str(config["dataset"]["cache_subdir"])

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        cache: dict | None = None,
    ) -> None:
        """Open one built split of this corpus's cache.

        Args:
            split: Split name; must be one of the corpus's configured splits.
            recipe_dir: Recipe root. Defaults to the package's own directory.
            cache: The recipe's `cache` block, naming where the cache lives.
        """
        self.split = str(split)
        if self.split not in self.SPLITS:
            raise ValueError(
                f"Unknown split '{self.split}'. Expected one of: "
                f"{', '.join(self.SPLITS)}"
            )

        recipe_root = (
            Path(recipe_dir)
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        split_dir = cache_root(recipe_root, cache, self.CACHE_SUBDIR) / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"{self.CACHE_SUBDIR} split '{self.split}' is not built: "
                f"{split_dir}. Run the create_dataset stage first."
            )
        self._rows = load_from_disk(str(split_dir))

    def __len__(self) -> int:
        """Return the number of cached utterances; the cache is the order."""
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one sample, composing the lang/task prefix and reading audio."""
        row = self._rows[int(idx)]
        # The prefix is composed here rather than stored, so the ISO 639-3
        # spelling can change without rebuilding the cache.
        prefix = lang_token(row["lang"]) + task_token(
            row["task"], row["tgt_lang"] or None
        )
        return {
            "speech": read_span(row["wav_path"], row["start_time"], row["end_time"]),
            "text": prefix + row["text"],
            "text_prev": row["text_prev"],
            "text_ctc": row["text_ctc"],
        }


def iter_text(recipe_dir, cache: dict | None, entries: list) -> Iterator[str]:
    """Yield the text stream of every row of every ``(corpus, split)`` entry.

    The tokenizer needs exactly what the model will see -- the ``<lang><task>``
    prefix and the timestamps included -- but not the audio, so this reads the
    cache columns directly rather than going through :class:`OWSMDataset`.
    It is a generator: the mixture's text does not fit in memory, and the
    caller samples from the stream.

    Args:
        recipe_dir: Recipe root, used to resolve a relative cache dir.
        cache: The recipe's ``cache`` block, giving ``cache_dir``.
        entries: ``[{"corpus": <cache_subdir>, "split": <split>}, ...]``.
    """
    for entry in entries:
        corpus, split = entry["corpus"], entry["split"]
        split_dir = cache_root(recipe_dir, cache, corpus) / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"{corpus} split '{split}' is not built: {split_dir}. "
                "Run the create_dataset stage first."
            )
        rows = load_from_disk(str(split_dir))
        for row in rows:
            target = row["tgt_lang"] or None
            prefix = lang_token(row["lang"]) + task_token(row["task"], target)
            yield prefix + row["text"]
