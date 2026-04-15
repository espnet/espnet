"""LJSpeech TTS dataset builder."""

from __future__ import annotations

import os
import tarfile
from importlib import resources
from pathlib import Path

from espnet2.text.build_tokenizer import build_tokenizer
from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url, setup_logger

LJSPEECH_URL = "http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


def _load_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)


_CONFIG = _load_config()
_BUILDER_CFG = _CONFIG["builder"]


def _resolve_recipe_root(recipe_dir: str | Path | None) -> Path:
    if recipe_dir is None:
        return Path(__file__).resolve().parents[1]
    return Path(recipe_dir).resolve()


def _find_existing_corpus_root(
    recipe_root: Path,
    corpus_root: str | Path | None,
) -> Path | None:
    candidates: list[Path] = []
    if corpus_root is not None:
        candidates.append(Path(corpus_root))

    env_root = os.environ.get("LJSPEECH")
    if env_root:
        candidates.append(Path(env_root))

    candidates.extend(
        [
            recipe_root / _BUILDER_CFG["source_path"],
            recipe_root / "downloads" / "LJSpeech-1.1",
            Path("downloads") / "LJSpeech-1.1",
        ]
    )

    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved.name == "LJSpeech-1.1" and (resolved / "metadata.csv").is_file():
            return resolved
        nested = resolved / "LJSpeech-1.1"
        if (nested / "metadata.csv").is_file():
            return nested

    return None


def _resolve_corpus_root(
    recipe_root: Path,
    corpus_root: str | Path | None,
) -> Path:
    resolved = _find_existing_corpus_root(recipe_root, corpus_root)
    if resolved is None:
        raise FileNotFoundError(
            "LJSpeech corpus not found. Set LJSPEECH or create_dataset.corpus_root."
        )
    return resolved


def _build_token_list(
    texts: list[str],
    token_type: str,
    g2p_type: str | None,
) -> list[str]:
    tokenizer = build_tokenizer(token_type=token_type, g2p_type=g2p_type)
    tokens = sorted(
        {
            token
            for text in texts
            for token in tokenizer.text2tokens(text)
            if token != "<unk>"
        }
    )
    return ["<unk>", *tokens]


class LJSpeechTTSBuilder(DatasetBuilder):
    """Prepare raw LJSpeech source data and the recipe token list for TTS."""

    def is_source_prepared(
        self,
        recipe_dir: str | Path | None = None,
        corpus_root: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether the raw LJSpeech corpus is already available."""
        recipe_root = _resolve_recipe_root(recipe_dir)
        return _find_existing_corpus_root(recipe_root, corpus_root) is not None

    def prepare_source(
        self,
        recipe_dir: str | Path | None = None,
        corpus_root: str | Path | None = None,
        **_kwargs,
    ) -> None:
        """Download and extract LJSpeech when no existing corpus root is found."""
        recipe_root = _resolve_recipe_root(recipe_dir)
        if self.is_source_prepared(recipe_dir=recipe_root, corpus_root=corpus_root):
            return

        data_root = recipe_root / _BUILDER_CFG["data_path"]
        data_root.mkdir(parents=True, exist_ok=True)
        logger = setup_logger("ljspeech.tts.create_dataset", log_dir=data_root)

        archive_path = recipe_root / _BUILDER_CFG["archive_path"]
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        if not archive_path.is_file():
            download_url(LJSPEECH_URL, archive_path, logger=logger)

        logger.info("Extracting: %s", archive_path.name)
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(path=archive_path.parent)

    def is_built(
        self,
        recipe_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether the token list already exists."""
        recipe_root = _resolve_recipe_root(recipe_dir)
        token_list = recipe_root / _BUILDER_CFG["token_list_path"]
        return token_list.is_file()

    def build(
        self,
        recipe_dir: str | Path | None = None,
        corpus_root: str | Path | None = None,
        token_type: str = "phn",
        g2p_type: str | None = "g2p_en_no_space",
        **_kwargs,
    ) -> None:
        """Build the LJSpeech token list from the raw corpus metadata."""
        recipe_root = _resolve_recipe_root(recipe_dir)
        ljspeech_root = _resolve_corpus_root(recipe_root, corpus_root)

        texts: list[str] = []
        metadata_path = ljspeech_root / "metadata.csv"
        with metadata_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                _utt_id, _unused, normalized_text = line.rstrip("\n").split(
                    "|",
                    maxsplit=2,
                )
                texts.append(normalized_text)

        if len(texts) <= 500:
            raise RuntimeError("LJSpeech split logic expects more than 500 utterances.")

        token_list = _build_token_list(
            texts[:-500],
            token_type=token_type,
            g2p_type=g2p_type,
        )
        token_list_path = recipe_root / _BUILDER_CFG["token_list_path"]
        token_list_path.parent.mkdir(parents=True, exist_ok=True)
        token_list_path.write_text("\n".join(token_list) + "\n", encoding="utf-8")
