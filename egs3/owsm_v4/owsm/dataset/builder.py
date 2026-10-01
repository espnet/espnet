"""Build path shared by every OWSM sub-dataset.

A corpus supplies its own source layout and its own port of the egs2
preparation script; everything around that -- locating the corpus, deciding
which splits to build, writing each split atomically, recording per-file
failures -- is the same for all of them and lives here.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Iterator

from egs3.owsm_v4.owsm.dataset.utils import (
    cache_root,
    check_cache_row,
    sub_dataset_config,
)
from espnet3.components.data.dataset_builder import DatasetBuilder


class OWSMBuilder(DatasetBuilder):
    """Write one OWSM utterance index per split under the cache root.

    Subclasses set the four class attributes and implement
    :meth:`is_valid_source_root` and :meth:`iter_rows`.

    Per-corpus options come from ``create_dataset.corpora.<CORPUS>``, because
    ``create_dataset`` passes the whole ``create_dataset:`` block to every
    builder with nothing in it naming the corpus.
    """

    #: Key under ``create_dataset.corpora``. The only thing a subclass must set.
    CORPUS: str = ""

    #: Filled from the subclass's own config.yaml by __init_subclass__.
    CONFIG: dict = {}
    CACHE_SUBDIR: str = ""
    SPLITS: tuple[str, ...] = ()
    SOURCE_ENV_VAR: str = ""
    PREFIX: str = ""
    LANG: str = ""
    EXPECTED_FS: int = 0

    def __init_subclass__(cls, **kwargs) -> None:
        """Read the subclass's ``config.yaml`` and set the shared attributes.

        Every sub-dataset repeated the same dozen lines of resource lookup and
        key extraction before it could get to its port. The config sits beside
        the subclass, so the base can find it.
        """
        super().__init_subclass__(**kwargs)
        config = sub_dataset_config(cls.__module__)
        if config is None:
            return
        cls.CONFIG = config
        builder, dataset = config["builder"], config["dataset"]
        cls.SPLITS = tuple(str(split) for split in config["splits"])
        cls.SOURCE_ENV_VAR = str(builder["source_env_var"])
        cls.EXPECTED_FS = int(builder["expected_fs"])
        cls.PREFIX = str(dataset["prefix"])
        cls.LANG = str(dataset["lang"])
        cls.CACHE_SUBDIR = str(dataset["cache_subdir"])

    def is_valid_source_root(self, candidate: Path) -> bool:
        """Return whether ``candidate`` holds this corpus' expected files."""
        raise NotImplementedError

    def iter_rows(
        self,
        source_root: Path,
        split: str,
        failures: Iterable | None = None,
        **options,
    ) -> Iterator[dict]:
        """Yield one cache row per utterance of ``split``.

        A row that cannot be produced is written to ``failures`` and skipped,
        so a single bad file does not abort a build of millions.
        """
        raise NotImplementedError

    def resolve_source_root(self, source_dir: str | Path | None = None) -> Path:
        """Locate the corpus, from ``source_dir`` or the corpus' env var."""
        import os

        candidates = []
        if source_dir is not None:
            candidates.append(Path(source_dir))
        env_path = os.environ.get(self.SOURCE_ENV_VAR)
        if env_path:
            candidates.append(Path(env_path))

        for candidate in candidates:
            if candidate.is_dir() and self.is_valid_source_root(candidate):
                return candidate.resolve()

        checked = "\n".join(f"  - {path}" for path in candidates) or "  (none given)"
        raise FileNotFoundError(
            f"{self.CORPUS} source not found. Checked:\n{checked}\n"
            f"Set {self.SOURCE_ENV_VAR} or pass "
            f"create_dataset.corpora.{self.CORPUS}.source_dir."
        )

    def corpus_options(self, corpora) -> dict:
        """Return this corpus' entry from the ``corpora`` block."""
        return dict((corpora or {}).get(self.CORPUS) or {})

    def requested_splits(self, options: dict) -> list[str]:
        """Return the splits to build, defaulting to all supported ones."""
        splits = options.get("splits") or self.SPLITS
        unknown = [split for split in splits if split not in self.SPLITS]
        if unknown:
            raise ValueError(
                f"Unknown {self.CORPUS} split(s) {unknown}; "
                f"expected any of {list(self.SPLITS)}."
            )
        return [str(split) for split in splits]

    def is_source_prepared(self, corpora=None, **_kwargs) -> bool:
        """Check that the corpus is on disk and complete."""
        try:
            self.resolve_source_root(self.corpus_options(corpora).get("source_dir"))
        except FileNotFoundError:
            return False
        return True

    def prepare_source(self, corpora=None, **_kwargs) -> None:
        """Raise unless the corpus is already on disk.

        OWSM's corpora are licence-gated or distributed out of band, so there
        is nothing to download; this only reports where it looked.
        """
        self.resolve_source_root(self.corpus_options(corpora).get("source_dir"))

    def is_built(self, recipe_dir, cache=None, corpora=None, **_kwargs) -> bool:
        """Check that every requested split has a cache directory."""
        root = cache_root(recipe_dir, cache, self.CACHE_SUBDIR)
        options = self.corpus_options(corpora)
        return all((root / split).is_dir() for split in self.requested_splits(options))

    def build(self, recipe_dir, cache=None, corpora=None, **_kwargs) -> None:
        """Build the missing split caches, one directory per split.

        Each split is written to a temporary directory and renamed into place,
        so an interrupted run leaves no partial index behind.
        """
        from datasets import Dataset as HFDataset

        options = self.corpus_options(corpora)
        source_root = self.resolve_source_root(options.get("source_dir"))
        root = cache_root(recipe_dir, cache, self.CACHE_SUBDIR)
        root.mkdir(parents=True, exist_ok=True)

        row_options = {
            key: value
            for key, value in options.items()
            if key not in ("source_dir", "splits")
        }

        for split in self.requested_splits(options):
            target = root / split
            if target.is_dir():
                continue
            temporary = root / f".{split}.tmp"
            shutil.rmtree(temporary, ignore_errors=True)
            failures_path = root / f"{split}.failures.jsonl"

            def rows(split=split, failures_path=failures_path):
                with failures_path.open("w", encoding="utf-8") as failures:
                    for row in self.iter_rows(
                        source_root, split, failures=failures, **row_options
                    ):
                        yield check_cache_row(row)

            HFDataset.from_generator(rows).save_to_disk(str(temporary))
            temporary.rename(target)
