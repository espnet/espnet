"""LibriSpeech 100h builder: download the raw corpus and dump it to omniio.

``create_dataset`` runs the two halves of :class:`DatasetBuilder`:

1. ``prepare_source`` makes the raw corpus available under
   ``<corpora_root>/librispeech/LibriSpeech/<split>``. A split already there
   is used as-is; only missing splits are downloaded from OpenSLR, checked
   against their md5 and extracted.
2. ``build`` dumps every split with omniio's ``Blob.append``: the FLAC files
   to ``<dump_root>/audio/librispeech/<split>`` and the transcripts to
   ``<dump_root>/text/librispeech/<split>``, both keyed by utterance ID.

Training and inference then read only the dump (see ``dataset.py``).
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tarfile
import urllib.request
from importlib import resources
from pathlib import Path

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.components.data.omniio_dataset import (
    DUMP_ROOT_ENV,
    blob_dir,
    dump_blob,
    is_blob_dumped,
)
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
CORPUS = str(_CFG["corpus"])
SPLITS: dict[str, str] = {str(k): str(v) for k, v in _CFG["splits"].items()}

_DEFAULT_RECIPE_ROOT = Path(__file__).resolve().parents[2]


def _recipe_root(recipe_dir: str | Path | None) -> Path:
    return Path(recipe_dir).resolve() if recipe_dir else _DEFAULT_RECIPE_ROOT


def resolve_corpus_dir(
    recipe_dir: str | Path | None = None,
    corpora_root: str | Path | None = None,
) -> Path:
    """Return ``<corpora_root>/librispeech``, the raw corpus directory.

    ``corpora_root`` is the argument, else ``$CORPORA_ROOT``, else
    ``<recipe_dir>/download``.
    """
    root = corpora_root or os.environ.get(str(_CFG["corpora_root_env"])) or None
    if root is None:
        root = _recipe_root(recipe_dir) / str(_CFG["local_corpora_dir"])
    return Path(root) / CORPUS


def resolve_dump_root(
    recipe_dir: str | Path | None = None,
    dump_root: str | Path | None = None,
) -> Path:
    """Return the omniio dump root.

    ``dump_root`` is the argument, else ``$OMNIIO_DUMP_ROOT``, else
    ``<recipe_dir>/dump/omniio``.
    """
    root = dump_root or os.environ.get(DUMP_ROOT_ENV) or None
    if root is None:
        root = _recipe_root(recipe_dir) / str(_CFG["local_dump_dir"])
    return Path(root)


def scan_split(split_dir: Path) -> list[tuple[str, Path, str]]:
    """Return ``(utt_id, flac_path, transcript)`` for one raw split, by ID."""
    examples = []
    for trans in sorted(split_dir.rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            utt_id, *words = fields
            if not words:
                continue
            flac = trans.parent / f"{utt_id}.flac"
            if not flac.is_file():
                raise FileNotFoundError(f"{trans} lists {utt_id} but {flac} is missing")
            examples.append((utt_id, flac, " ".join(words)))
    if not examples:
        raise RuntimeError(f"No transcript/audio pairs found under {split_dir}")
    return sorted(examples)


def _md5(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _fetch(split: str, archive: Path) -> None:
    """Download ``<split>.tar.gz`` from the first mirror that serves it intact."""
    partial = archive.with_suffix(".gz.part")
    errors = []
    for mirror in _CFG["mirrors"]:
        url = f"{mirror}/{split}.tar.gz"
        logger.info("Downloading %s -> %s", url, archive)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                with partial.open("wb") as fh:
                    shutil.copyfileobj(response, fh, length=1 << 24)
        except OSError as exc:
            errors.append(f"{url}: {exc}")
            continue
        checksum = _md5(partial)
        if checksum == SPLITS[split]:
            partial.rename(archive)
            return
        errors.append(f"{url}: md5 {checksum} != {SPLITS[split]}")
    partial.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download {split}:\n  " + "\n  ".join(errors))


def download_split(split: str, corpus_dir: Path, keep_archive: bool = False) -> None:
    """Download, verify and extract ``<split>.tar.gz`` into ``corpus_dir``.

    The archive is extracted into a scratch directory and the split is moved
    into ``corpus_dir/LibriSpeech/<split>`` only when extraction succeeded,
    so an interrupted run never leaves a half-extracted split that looks
    complete.
    """
    corpus_dir.mkdir(parents=True, exist_ok=True)
    archive = corpus_dir / f"{split}.tar.gz"
    if not archive.is_file() or _md5(archive) != SPLITS[split]:
        _fetch(split, archive)

    scratch = corpus_dir / f".extract_{split}"
    shutil.rmtree(scratch, ignore_errors=True)
    logger.info("Extracting %s", archive)
    with tarfile.open(archive) as tar:
        tar.extractall(scratch, filter="data")
    target_root = corpus_dir / "LibriSpeech"
    target_root.mkdir(exist_ok=True)
    for entry in (scratch / "LibriSpeech").iterdir():
        dest = target_root / entry.name
        if entry.name == split:
            entry.rename(dest)
        elif not dest.exists():  # README, CHAPTERS.TXT, ... shared by splits
            entry.rename(dest)
    shutil.rmtree(scratch)
    if not keep_archive:
        archive.unlink()


class LibriSpeech100OmniIOBuilder(DatasetBuilder):
    """Prepare raw LibriSpeech and dump it to omniio blobs.

    Every method accepts the ``create_dataset`` options of the training
    config: ``recipe_dir``, ``corpora_root``, ``dump_root``, ``num_workers``
    and ``keep_archive``.
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path | None = None,
        corpora_root: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Return whether every split is extracted under the corpora root."""
        root = resolve_corpus_dir(recipe_dir, corpora_root) / "LibriSpeech"
        return all((root / split).is_dir() for split in SPLITS)

    def prepare_source(
        self,
        recipe_dir: str | Path | None = None,
        corpora_root: str | Path | None = None,
        keep_archive: bool = False,
        **_kwargs,
    ) -> None:
        """Download the splits missing from the corpora root."""
        corpus_dir = resolve_corpus_dir(recipe_dir, corpora_root)
        for split in SPLITS:
            if (corpus_dir / "LibriSpeech" / split).is_dir():
                logger.info("Found %s in %s; not downloading", split, corpus_dir)
                continue
            download_split(split, corpus_dir, keep_archive=keep_archive)

    def is_built(
        self,
        recipe_dir: str | Path | None = None,
        dump_root: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Return whether every split has finished audio and text blobs."""
        root = resolve_dump_root(recipe_dir, dump_root)
        return all(
            is_blob_dumped(blob_dir(root, modality, CORPUS, split))
            for split in SPLITS
            for modality in ("audio", "text")
        )

    def build(
        self,
        recipe_dir: str | Path | None = None,
        corpora_root: str | Path | None = None,
        dump_root: str | Path | None = None,
        num_workers: int | None = None,
        **_kwargs,
    ) -> None:
        """Dump each split's audio and transcripts with ``Blob.append``."""
        source = resolve_corpus_dir(recipe_dir, corpora_root) / "LibriSpeech"
        root = resolve_dump_root(recipe_dir, dump_root)
        dump_cfg = _CFG["dump"]
        workers = int(dump_cfg["num_workers"] if num_workers is None else num_workers)
        for split in SPLITS:
            audio_dir = blob_dir(root, "audio", CORPUS, split)
            text_dir = blob_dir(root, "text", CORPUS, split)
            if is_blob_dumped(audio_dir) and is_blob_dumped(text_dir):
                logger.info("omniio dump of %s already exists; skipping", split)
                continue
            examples = scan_split(source / split)
            ids = [utt_id for utt_id, _, _ in examples]
            dump_blob(
                audio_dir,
                "audio",
                [str(flac) for _, flac, _ in examples],
                ids,
                num_workers=workers,
                **dict(dump_cfg["audio"]),
            )
            dump_blob(
                text_dir,
                "text",
                [text for _, _, text in examples],
                ids,
                num_workers=workers,
                **dict(dump_cfg["text"]),
            )
