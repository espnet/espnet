"""AmericasNLP 2022 dataset builder.

The corpus of the second AmericasNLP 2022 shared task ships one tarball per
language (``<LangName>TrainDev.tar.gz``). Each archive extracts to a
``<LangName>/`` directory containing ``train/`` and ``dev/`` splits, each of
which holds one 16 kHz wav per utterance plus a ``meta.tsv`` transcript table.
The builder's only job is to make those directories available; the dataset
reads ``meta.tsv`` directly, so no task-ready artifacts are built.
"""

from __future__ import annotations

import logging
import os
from importlib import resources
from pathlib import Path
from typing import Iterable

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url, extract_targz

logger = logging.getLogger(__name__)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()


def resolve_language(lang: str) -> str:
    """Map an ISO language code to the corpus archive directory name."""
    languages = {str(k): str(v) for k, v in _CFG["languages"].items()}
    if lang not in languages:
        known = ", ".join(sorted(languages))
        raise ValueError(f"Unknown language '{lang}'. Expected one of: {known}")
    return languages[lang]


def iter_source_candidates(
    recipe_dir: str | Path,
    source_dir: str | Path | None,
) -> Iterable[Path]:
    """Yield candidate directories that may contain the corpus."""
    yield Path(recipe_dir) / str(_CFG["dataset_path"])

    if source_dir is not None:
        yield Path(source_dir)

    env_var = str(_CFG["source_env_var"])
    env_path = os.environ.get(env_var)
    if env_path:
        yield Path(env_path)


def resolve_source_root(
    recipe_dir: str | Path,
    source_dir: str | Path | None = None,
) -> Path:
    """Return the corpus root (parent of the per-language directories)."""
    for candidate in iter_source_candidates(recipe_dir, source_dir):
        if candidate.is_dir() and any(
            (candidate / Path(lang_name)).is_dir()
            for lang_name in _CFG["languages"].values()
        ):
            return candidate
    env_var = str(_CFG["source_env_var"])
    raise FileNotFoundError(
        "AmericasNLP22 corpus not found. Place it under "
        f"<recipe_dir>/{_CFG['dataset_path']}/, pass source_dir, or set "
        f"{env_var} to the corpus root."
    )


def resolve_language_dir(source_root: Path, lang: str) -> Path:
    """Return the directory of one language inside the corpus root."""
    return source_root / resolve_language(lang)


class AmericasNLP22Builder(DatasetBuilder):
    """Download and validate the AmericasNLP 2022 corpus for one language.

    This recipe reads the original ``meta.tsv`` layout directly during
    training and inference, so the builder only ensures the required split
    directories exist (downloading the language archive if necessary).
    """

    def _collect_missing_splits(
        self,
        recipe_dir: str | Path,
        lang: str,
        source_dir: str | Path | None = None,
    ) -> list[str]:
        """Collect the required splits of one language that are missing."""
        source_root = resolve_source_root(recipe_dir, source_dir=source_dir)
        lang_dir = resolve_language_dir(source_root, lang)
        return [
            str(split)
            for split in _CFG["required_splits"]
            if not (lang_dir / str(split) / "meta.tsv").is_file()
        ]

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        lang: str | None = None,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether the required splits of the language are available."""
        if lang is None:
            raise ValueError("AmericasNLP22Builder requires a `lang` argument.")
        try:
            return not self._collect_missing_splits(recipe_dir, lang, source_dir)
        except FileNotFoundError:
            return False

    def prepare_source(
        self,
        recipe_dir: str | Path,
        lang: str | None = None,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> None:
        """Download and extract the language archive if it is missing."""
        if lang is None:
            raise ValueError("AmericasNLP22Builder requires a `lang` argument.")

        if source_dir is not None:
            target_root = Path(source_dir)
        else:
            target_root = Path(recipe_dir) / str(_CFG["dataset_path"])
        target_root.mkdir(parents=True, exist_ok=True)

        lang_name = resolve_language(lang)
        lang_dir = target_root / lang_name
        missing = [
            str(split)
            for split in _CFG["required_splits"]
            if not (lang_dir / str(split) / "meta.tsv").is_file()
        ]
        if missing:
            url = f"{_CFG['url_base'].rstrip('/')}/{lang_name}{_CFG['archive_suffix']}"
            archive = target_root / f"{lang_name}{_CFG['archive_suffix']}"
            if not archive.is_file():
                logger.info("Downloading %s", url)
                download_url(url, archive)
            logger.info("Extracting %s into %s", archive, target_root)
            extract_targz(archive, target_root)
            archive.unlink(missing_ok=True)

        still_missing = self._collect_missing_splits(recipe_dir, lang, source_dir)
        if still_missing:
            raise FileNotFoundError(
                f"AmericasNLP22 source is incomplete for '{lang}'. "
                "Missing split directories: " + ", ".join(still_missing)
            )

    def is_built(
        self,
        recipe_dir: str | Path,
        lang: str | None = None,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Return source readiness because this recipe has no build artifacts."""
        return self.is_source_prepared(
            recipe_dir=recipe_dir,
            lang=lang,
            source_dir=source_dir,
        )

    def build(
        self,
        recipe_dir: str | Path,
        lang: str | None = None,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> None:
        """No-op build step for raw-directory-backed corpus access."""
        self.prepare_source(recipe_dir=recipe_dir, lang=lang, source_dir=source_dir)
