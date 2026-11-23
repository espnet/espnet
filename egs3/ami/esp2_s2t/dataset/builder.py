"""AMI SOT dataset builder.

The builder reads two sources and writes one CutSet for each split.

* Lhotse CutSet manifests with the speaker-attributed supervisions. They are
  downloaded from the Hugging Face repository set by ``builder.cutset_repo``.
* AMI SDM audio, downloaded with ``lhotse.recipes.ami.download_audio``. The
  manifests store absolute paths from the machine that built them. A
  recording whose stored path does not exist is pointed to the file with the
  same name under ``builder.ami_audio_root``.

``build`` serializes the supervisions of each cut with
:func:`sot_text.build_sot_text`, using the options of the split, and stores
the text in the cut. The audio is not copied. Each cut stores its recording,
channel, start time and duration, so the dataset reads the audio of an
utterance group through the CutSet.
"""

from __future__ import annotations

import logging
from pathlib import Path

from egs3.ami.esp2_s2t.dataset.dataset import _CONFIG, _split_dir
from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.systems.esp2_s2t.src.sot import sot_text
from espnet3.systems.esp2_s2t.src.sot.text_norm import get_text_norm

logger = logging.getLogger(__name__)

# CutSet file in each split directory. Each cut in it includes its serialized
# text.
_CUTS_FILENAME = "cuts.jsonl.gz"

# Key in cut.custom for the output of build_sot_text.
_SOT_TEXT_FIELD = "sot_text"

# The builder writes this marker file into data_root before it writes any
# split. If split data exists without the marker, another tool created the
# corpus, and the builder does not modify it.
_MARKER_NAME = "data/.ami_sot_builder"


def _relocate_recording(cut, audio_root: Path) -> None:
    """Point the recording of a cut to the downloaded AMI audio.

    The manifests store absolute paths from the machine that built them. A
    path that exists is kept, so a corpus that is already prepared still
    works. Any other path is replaced by the file with the same name under
    ``audio_root``, in the ``<root>/wav_db/<meeting>/audio/<file>.wav`` layout
    that ``lhotse.recipes.ami.download_audio`` writes.

    Args:
        cut: Lhotse cut, modified in place.
        audio_root: Directory that the AMI audio was downloaded into.
    """
    for source in getattr(cut.recording, "sources", []):
        if source.type != "file" or Path(source.source).exists():
            continue
        name = Path(source.source).name
        source.source = str(audio_root / "wav_db" / name.split(".")[0] / "audio" / name)


class AmiSotBuilder(DatasetBuilder):
    """Build the AMI SOT split directories from cutsets and AMI audio."""

    def _get_root(self, key: str) -> Path:
        """Return a configured directory, relative to ``data_root`` if not absolute."""
        path = Path(_CONFIG[key])
        return path if path.is_absolute() else Path(_CONFIG["data_root"]) / path

    def _get_cutset_path(self, split: str) -> Path:
        """Return the local path of ``split``'s CutSet manifest."""
        return self._get_root("cutset_dir") / _CONFIG["sot"][split]["cutset"]

    def is_source_prepared(self, **kwargs) -> bool:
        """Return True when all manifests and at least one AMI audio file exist."""
        if not all(self._get_cutset_path(s).is_file() for s in _CONFIG["sot"]):
            return False
        audio_root = self._get_root("ami_audio_root")
        return audio_root.is_dir() and any(audio_root.glob("**/audio/*.wav"))

    def _prepare_source_impl(self, **kwargs) -> None:
        """Download the manifests, and the AMI SDM audio if it is missing."""
        cutset_dir = self._get_root("cutset_dir")
        cutset_dir.mkdir(parents=True, exist_ok=True)
        for split, options in _CONFIG["sot"].items():
            if self._get_cutset_path(split).is_file():
                continue
            logger.info("Fetching %s cutset %s", split, options["cutset"])
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id=_CONFIG["cutset_repo"],
                filename=options["cutset"],
                local_dir=str(cutset_dir),
            )

        audio_root = self._get_root("ami_audio_root")
        if audio_root.is_dir() and any(audio_root.glob("**/audio/*.wav")):
            return
        audio_root.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading AMI SDM audio into %s", audio_root)
        # This uses lhotse instead of ami_download.sh from egs2, because egs2
        # is not part of the installed espnet package.
        from lhotse.recipes.ami import download_audio

        download_audio(audio_root, mic="sdm")

    def prepare_source(self, **kwargs) -> None:
        """Download the manifests and the AMI audio.

        ``BaseSystem.create_dataset`` calls this method before ``is_built``.
        Without the check below, it would create ``downloads/`` inside a
        corpus directory that this builder did not create, and start the AMI
        audio download there.
        """
        self._refuse_foreign_corpus()
        return self._prepare_source_impl(**kwargs)

    def _refuse_foreign_corpus(self) -> None:
        """Stop if ``data_root`` has a corpus that this builder did not create.

        Raises:
            RuntimeError: When ``data_root`` has split data but no marker file.
        """
        data_root = Path(_CONFIG["data_root"])
        if (data_root / _MARKER_NAME).is_file():
            return
        for split in _CONFIG["split_dirs"]:
            split_dir = _split_dir(split)
            existing = (
                [_CUTS_FILENAME] if (split_dir / _CUTS_FILENAME).is_file() else []
            )
            if existing:
                raise RuntimeError(
                    f"{split_dir} already contains {', '.join(existing)}, and "
                    f"{data_root / _MARKER_NAME} does not exist, so this "
                    "corpus was not created by this builder, and it will not "
                    "be overwritten. Set AMI_SOT_DATA_ROOT to a new directory, "
                    "or create the marker file if the corpus may be "
                    "overwritten."
                )

    def _claim_data_root(self) -> None:
        """Write the marker file that shows this builder created ``data_root``.

        The marker is written before the build starts. If it were written at
        the end and a build stopped partway, the split data would exist
        without the marker, and the builder would refuse to touch its own
        output.
        """
        marker = Path(_CONFIG["data_root"]) / _MARKER_NAME
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            "Written by AmiSotBuilder in egs3/ami/esp2_s2t. While this file "
            "exists, the builder may overwrite this directory. Delete it to "
            "protect the corpus.\n"
        )

    def is_built(self, **kwargs) -> bool:
        """Return True when the CutSet file of every split exists.

        The token list is not checked. A corpus prepared elsewhere can have the
        splits but no token list. If this method returned False for it,
        ``create_dataset`` would call ``build()``, which refuses to overwrite
        that corpus. Its owner only needs to call ``write_token_list``, which
        adds the file and changes nothing else. Without the token list,
        training stops with an error.
        """
        for split in _CONFIG["split_dirs"]:
            split_dir = _split_dir(split)
            for name in _CONFIG["required_files"]:
                if not (split_dir / name).is_file():
                    return False
        return True

    def build(self, **kwargs) -> None:
        """Write the CutSet of every configured split, then the token list."""
        self._refuse_foreign_corpus()
        self._claim_data_root()
        if not self.is_source_prepared():
            self.prepare_source()
        for split in _CONFIG["split_dirs"]:
            self._build_split(split)
        self.write_token_list()

    def write_token_list(self, **kwargs) -> None:
        """Export the Whisper vocabulary of the corpus text as the token list.

        ``build`` calls this method last. It is public and skips the corpus
        check, because it only adds a file. A corpus that is already prepared
        needs it, because the training config reads the token list and
        ``is_built`` does not check for it.

        The exporter adds the separator only when the vocabulary does not
        contain it, so the same call works for a symbol that is a Whisper token
        and for one that is not. The export is skipped when the file already
        exists, because it takes seconds and every builder test calls
        ``build``.
        """
        from espnet3.systems.esp2_s2t.src.sot.whisper_export_vocabulary import (
            export_vocabulary,
        )

        target = Path(_CONFIG["data_root"]) / _CONFIG["token_list"]
        separator = _CONFIG["separator"]
        if target.is_file():
            # A file written for another separator is exported again.
            # Otherwise the text would use one symbol and the token list
            # another.
            rows = target.read_text().splitlines()
            if rows.count(separator) == 1:
                logger.info("Token list already present at %s", target)
                return
            logger.info(
                "Rewriting the token list at %s for separator %s",
                target,
                separator,
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        export_vocabulary(
            output=str(target),
            whisper_model="whisper_multilingual",
            log_level="INFO",
            sot_asr=True,
            speaker_change_symbol=separator,
        )
        logger.info("Wrote the token list to %s", target)

    def _build_split(self, split: str) -> None:
        """Serialize the CutSet of one split and store the text in each cut.

        The audio is not copied. Each cut stores its recording, channel, start
        time and duration, so the dataset reads the audio of an utterance
        group through the CutSet.

        Args:
            split: Key in both ``builder.split_dirs`` and ``builder.sot``.
        """
        from lhotse import CutSet

        options = _CONFIG["sot"][split]
        split_dir = _split_dir(split)
        split_dir.mkdir(parents=True, exist_ok=True)
        audio_root = self._get_root("ami_audio_root")

        max_duration = _CONFIG["max_cut_duration"]
        # Created once for each split, because the CHiME-8 normalizer loads a
        # 56 KB spelling table when it is created.
        normalizer = get_text_norm(options.get("text_norm"))
        kept = []
        for cut in CutSet.from_file(str(self._get_cutset_path(split))):
            if cut.duration >= max_duration:
                # A group longer than the Whisper window was cut to exactly
                # this length, so part of the speech in its transcript is
                # missing from the audio. Such groups are dropped.
                continue
            text = sot_text.build_sot_text(
                cut.supervisions,
                max_timestamp_pause=options["max_timestamp_pause"],
                ordering=options["ordering"],
                separator=_CONFIG["separator"],
                lowercase=options["lowercase"],
                text_norm=normalizer,
                prompt=_CONFIG["prompt"],
                # _calc_att_loss appends the end token itself.
                eos=None,
            )
            _relocate_recording(cut, audio_root)
            # The text is stored in the cut, so each split is a single file,
            # and no separate text file can get out of sync with it.
            cut.custom = dict(cut.custom or {})
            cut.custom[_SOT_TEXT_FIELD] = text
            kept.append(cut)

        kept.sort(key=lambda cut: cut.id)
        CutSet.from_cuts(kept).to_file(str(split_dir / _CUTS_FILENAME))
        logger.info("Wrote %d cuts to %s", len(kept), split_dir / _CUTS_FILENAME)
