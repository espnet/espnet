"""AMI SOT dataset builder.

Two sources go in and one CutSet per split comes out:

* Lhotse CutSet manifests, which carry the speaker-attributed supervisions.
  They are fetched from the Hugging Face repo named by ``builder.cutset_repo``.
* AMI SDM audio, downloaded by ``lhotse.recipes.ami.download_audio``. The
  cutsets store absolute paths from wherever they were built, so any
  recording whose stored path is absent is relocated into
  ``builder.ami_audio_root`` by basename.

``build`` serializes each cut with :func:`sot_text.build_sot_text` under that
split's options and stores the result on the cut. The audio is left where it
is: a cut already names its recording, channel, offset and duration, so the
dataset reads the group's audio through the CutSet rather than from a copy
written out per utterance.
"""

from __future__ import annotations

import logging
from pathlib import Path

from egs3.ami.esp2_s2t.dataset import sot_text
from egs3.ami.esp2_s2t.dataset.dataset import _CONFIG, _split_dir
from egs3.ami.esp2_s2t.dataset.text_norm import get_text_norm
from espnet3.components.data.dataset_builder import DatasetBuilder

logger = logging.getLogger(__name__)

# Name of the per-split directory the extracted audio segments are written to.
# One file per split, holding the cuts with their serialized transcript.
_CUTS_FILENAME = "cuts.jsonl.gz"

# Where build_sot_text's output is stored on each cut.
_SOT_TEXT_FIELD = "sot_text"

# Written into data_root the moment the builder claims that directory. Its
# absence beside existing split data means the corpus came from somewhere
# else, and the builder will not touch it.
_MARKER_NAME = "data/.ami_sot_builder"


def _relocate_recording(cut, audio_root: Path) -> None:
    """Point a cut's recording at the locally downloaded AMI audio.

    The manifests carry absolute paths from the machine that built them. A
    path that still resolves is left alone, so a prepared corpus keeps
    working; anything else is remapped into ``audio_root`` by basename, which
    is the ``<root>/wav_db/<meeting>/audio/<meeting>.Array1-01.wav`` layout
    that ``lhotse.recipes.ami.download_audio`` writes.

    Args:
        cut: Lhotse cut, modified in place.
        audio_root: Root the AMI audio was downloaded into.
    """
    for source in getattr(cut.recording, "sources", []):
        if source.type != "file" or Path(source.source).exists():
            continue
        name = Path(source.source).name
        source.source = str(audio_root / "wav_db" / name.split(".")[0] / "audio" / name)


class AmiSotBuilder(DatasetBuilder):
    """Build the AMI SOT split directories from cutsets and AMI audio."""

    # ------------------------------------------------------------- helpers
    def _get_root(self, key: str) -> Path:
        """Resolve a configured directory against ``data_root``."""
        path = Path(_CONFIG[key])
        return path if path.is_absolute() else Path(_CONFIG["data_root"]) / path

    def _get_cutset_path(self, split: str) -> Path:
        """Return the local path of ``split``'s CutSet manifest."""
        return self._get_root("cutset_dir") / _CONFIG["sot"][split]["cutset"]

    # -------------------------------------------------------------- source
    def is_source_prepared(self, **kwargs) -> bool:
        """Return True when every cutset and some AMI audio is on disk."""
        if not all(self._get_cutset_path(s).is_file() for s in _CONFIG["sot"]):
            return False
        audio_root = self._get_root("ami_audio_root")
        return audio_root.is_dir() and any(audio_root.glob("**/audio/*.wav"))

    def _prepare_source_impl(self, **kwargs) -> None:
        """Download the cutsets and, when absent, the AMI SDM audio."""
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
        # lhotse rather than egs2's ami_download.sh: egs2 is not part of the
        # distributed package, so a recipe that shells out to it cannot run
        # from an installed espnet.
        from lhotse.recipes.ami import download_audio

        download_audio(audio_root, mic="sdm")

    # --------------------------------------------------------------- build
    def prepare_source(self, **kwargs) -> None:
        """Fetch the cutsets and the AMI audio, refusing a foreign corpus.

        ``BaseSystem.create_dataset`` calls this itself, and before
        ``is_built``, so an unguarded version would create ``downloads/``
        inside a corpus root that is not ours and start a multi-hundred-
        gigabyte AMI download there.
        """
        self._refuse_foreign_corpus()
        return self._prepare_source_impl(**kwargs)

    def _refuse_foreign_corpus(self) -> None:
        """Stop before touching a corpus root this builder did not create.

        Raises:
            RuntimeError: When ``data_root`` holds split data and no marker.
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
                    f"{split_dir} already holds {', '.join(existing)}, and "
                    f"{data_root / _MARKER_NAME} is absent, so this corpus was "
                    "not created by this builder. Refusing to overwrite it. "
                    "Point AMI_SOT_DATA_ROOT at a new directory, or, if this "
                    "corpus really is disposable, create that marker file."
                )

    def _claim_data_root(self) -> None:
        """Record that this directory belongs to the builder.

        Written before any work, not after: a build cuts tens of thousands
        of segments, and an interruption partway would otherwise leave split
        data with no marker and lock the builder out of its own output.
        """
        marker = Path(_CONFIG["data_root"]) / _MARKER_NAME
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            "Written by egs3/ami/esp2_s2t AmiSotBuilder. Its presence lets the "
            "builder rewrite this directory; delete it to protect the corpus.\n"
        )

    def is_built(self, **kwargs) -> bool:
        """Return True when every split's manifests exist.

        Deliberately does NOT require the token list. A corpus prepared before
        this recipe existed has the splits but no vocabulary, and reporting it
        unbuilt would send ``create_dataset`` into ``build()``, where the guard
        raises. Its owner needs one file, not a rebuild: ``write_token_list``
        adds it without touching anything else, and training fails loudly on a
        missing token list if they skip that step.
        """
        for split in _CONFIG["split_dirs"]:
            split_dir = _split_dir(split)
            for name in _CONFIG["required_files"]:
                if not (split_dir / name).is_file():
                    return False
        return True

    def build(self, **kwargs) -> None:
        """Write every configured split's Kaldi directory."""
        self._refuse_foreign_corpus()
        self._claim_data_root()
        if not self.is_source_prepared():
            self.prepare_source()
        for split in _CONFIG["split_dirs"]:
            self._build_split(split)
        self._write_token_list()

    def write_token_list(self, **kwargs) -> None:
        """Export the vocabulary into an existing corpus root.

        Separate from ``build`` and unguarded, because it only ever adds a
        file and skips when one is present. It is what an already-prepared
        corpus needs in order to satisfy the training config, which reads the
        token list ``is_built`` deliberately does not require.
        """
        self._write_token_list()

    def _write_token_list(self) -> None:
        """Export the Whisper vocabulary this corpus is serialized against.

        The separator is appended only when the vocabulary lacks it, which the
        exporter decides, so one call serves a symbol that is already a Whisper
        token and one that is not. Skipped when the file is already there: a
        full export takes seconds and every builder test calls ``build``.
        """
        from espnet2.bin.whisper_export_vocabulary import export_vocabulary

        target = Path(_CONFIG["data_root"]) / _CONFIG["token_list"]
        separator = _CONFIG["separator"]
        if target.is_file():
            # Skipping keeps the builder tests fast, but a bare existence
            # check would leave a stale vocabulary in place after the
            # separator changed, with the text written against one symbol and
            # the token list against another.
            rows = target.read_text().splitlines()
            if rows.count(separator) == 1:
                logger.info("Token list already present at %s", target)
                return
            logger.info(
                "Token list at %s predates separator %s; rewriting",
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
        """Serialize one split's cutset and store the text on each cut.

        The audio is not copied. A cut already names its recording, channel,
        offset and duration, so the dataset reads the group's audio straight
        through the CutSet; writing one wav per utterance group would
        duplicate the corpus for nothing.

        Args:
            split: Key shared by ``builder.split_dirs`` and ``builder.sot``.
        """
        from lhotse import CutSet

        options = _CONFIG["sot"][split]
        split_dir = _split_dir(split)
        split_dir.mkdir(parents=True, exist_ok=True)
        audio_root = self._get_root("ami_audio_root")

        max_duration = _CONFIG["max_cut_duration"]
        # Resolved once per split rather than per cut: the CHiME-8 normalizer
        # loads a 56 KB spelling table at construction.
        normalizer = get_text_norm(options.get("text_norm"))
        kept = []
        for cut in CutSet.from_file(str(self._get_cutset_path(split))):
            if cut.duration >= max_duration:
                # A group whose span overran Whisper's window was truncated to
                # exactly the limit, losing the speech the transcript still
                # names. The prepared corpus drops these rather than score a
                # reference against audio that no longer contains it.
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
            # The serialized transcript rides on the cut, so the split is one
            # artifact rather than a manifest plus a parallel text file that
            # could drift out of step with it.
            cut.custom = dict(cut.custom or {})
            cut.custom[_SOT_TEXT_FIELD] = text
            kept.append(cut)

        kept.sort(key=lambda cut: cut.id)
        CutSet.from_cuts(kept).to_file(str(split_dir / _CUTS_FILENAME))
        logger.info("Wrote %d cuts to %s", len(kept), split_dir / _CUTS_FILENAME)
