"""LibriSpeech 100h builder: dump the raw corpus to omniio blobs.

``create_dataset`` runs the two halves of :class:`DatasetBuilder`:

1. ``prepare_source`` is inherited from the raw-LibriSpeech builder in
   ``dataset/builder.py``: it reuses an existing corpus (``source_dir`` or
   ``$LIBRISPEECH``) or downloads the missing splits into ``download/``.
2. ``build`` dumps every split with omniio's ``Blob.append``: the FLAC files
   to ``<dump_root>/audio/librispeech/<split>`` and the transcripts to
   ``<dump_root>/text/librispeech/<split>``, both keyed by utterance ID.

Training and inference then read only the dump (see ``dataset.py``). When the
dump already exists, for example one shared by another user, the raw corpus
is not needed at all.
"""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path

from egs3.librispeech_100.esp2_asr.dataset.builder import _CFG as _SOURCE_CFG
from egs3.librispeech_100.esp2_asr.dataset.builder import (
    LibriSpeech100Builder,
    resolve_source_root,
)
from egs3.librispeech_100.esp2_asr.dataset.dataset import _scan_split
from espnet3.components.data.omniio_dataset import (
    dump_blob,
    get_blob_dir,
    is_blob_dumped,
    resolve_dump_root,
)
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)


def _load_builder_config() -> dict:
    """Return the ``builder`` section of this package's ``config.yaml``."""
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
CORPUS = str(_CFG["corpus"])
# The splits the raw-LibriSpeech builder prepares (dataset/config.yaml)
SPLITS = [str(split) for split in _SOURCE_CFG["required_splits"]]


class LibriSpeech100OmniIOBuilder(LibriSpeech100Builder):
    """Locate or download raw LibriSpeech, then dump it to omniio blobs.

    Every method accepts the ``create_dataset`` options of the training
    config: ``recipe_dir``, ``source_dir``, ``dump_root`` (else
    ``$OMNIIO_DUMP_ROOT``) and ``num_workers``.

    Examples:
        >>> builder = LibriSpeech100OmniIOBuilder()
        >>> builder.is_built(recipe_dir=".", dump_root="/nonexistent")
        False
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        dump_root: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Return whether the raw corpus is available or no longer needed.

        A finished dump makes the raw corpus unnecessary, so nothing is
        downloaded for a dump made elsewhere.
        """
        if self.is_built(recipe_dir=recipe_dir, dump_root=dump_root):
            return True
        return super().is_source_prepared(recipe_dir, source_dir=source_dir)

    def is_built(
        self,
        recipe_dir: str | Path,
        dump_root: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Return whether every split has finished audio and text blobs."""
        root = resolve_dump_root(dump_root)
        return all(
            is_blob_dumped(get_blob_dir(root, modality, CORPUS, split))
            for split in SPLITS
            for modality in ("audio", "text")
        )

    def build(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        dump_root: str | Path | None = None,
        num_workers: int | None = None,
        **_kwargs,
    ) -> None:
        """Dump each split's audio and transcripts with ``Blob.append``.

        Splits already dumped are skipped, and so are utterances already in a
        partly written blob.

        Raises:
            FileNotFoundError: If the raw corpus is needed but not found.
        """
        root = resolve_dump_root(dump_root)
        dump_config = _CFG["dump"]
        workers = int(
            dump_config["num_workers"] if num_workers is None else num_workers
        )
        for split in SPLITS:
            audio_dir = get_blob_dir(root, "audio", CORPUS, split)
            text_dir = get_blob_dir(root, "text", CORPUS, split)
            if is_blob_dumped(audio_dir) and is_blob_dumped(text_dir):
                logger.info("omniio dump of %s already exists; skipping", split)
                continue
            source = resolve_source_root(Path(recipe_dir).resolve(), source_dir)
            # The raw-LibriSpeech dataset's transcript scan, sorted by ID.
            examples = _scan_split(source / split)
            ids = [example.utt_id for example in examples]
            dump_blob(
                audio_dir,
                "audio",
                [str(example.audio_path) for example in examples],
                ids,
                num_workers=workers,
                **dict(dump_config["audio"] or {}),
            )
            dump_blob(
                text_dir,
                "text",
                [example.text for example in examples],
                ids,
                num_workers=workers,
                **dict(dump_config["text"] or {}),
            )
