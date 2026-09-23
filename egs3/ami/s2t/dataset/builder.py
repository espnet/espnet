"""AMI SOT dataset builder.

Two sources go in and Kaldi-style split directories come out:

* Lhotse CutSet manifests, which carry the speaker-attributed supervisions.
  They are fetched from the Hugging Face repo named by ``builder.cutset_repo``.
* AMI SDM audio, downloaded by the existing
  ``egs2/ami/asr1/local/ami_download.sh``. The cutsets store absolute paths
  from wherever they were built, so any recording whose stored path is absent
  is relocated into ``builder.ami_audio_root`` by basename.

``build`` serializes each cut with :func:`sot_text.build_sot_text` under that
split's options and cuts the audio segment each utterance group refers to.
The committed options reproduce ``egs2/ami/sot_asr1/data/*/text`` byte for
byte, so this recipe needs nothing from that one.
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
from pathlib import Path

from espnet3.components.data.dataset_builder import DatasetBuilder

if __package__:
    # Normal case: this module is part of the real egs3.ami.s2t.dataset
    # package, so the relative import resolves against it directly. Gating
    # on __package__ instead of wrapping this in try/except ImportError
    # means a genuine breakage in dataset.py (for example _CONFIG or
    # _split_dir renamed) surfaces as its own ImportError here, rather than
    # being caught and silently rerouted into the fallback below.
    from . import sot_text
    from .dataset import _CONFIG, _split_dir
    from .text_norm import get_text_norm
else:
    # A relative import has no parent package to resolve against when this
    # file is loaded standalone, for example via
    # ``importlib.util.spec_from_file_location`` in tests with a flat,
    # non-dotted module name (so __package__ is ``""``). Fall back to
    # loading the sibling module by file path so ``_CONFIG`` and
    # ``_split_dir`` still come from ``dataset.py`` rather than being
    # redefined here.
    _module_key = f"{__name__}._ami_sot_dataset_impl"
    if _module_key in sys.modules:
        _dataset = sys.modules[_module_key]
    else:
        _spec = importlib.util.spec_from_file_location(
            _module_key, Path(__file__).resolve().parent / "dataset.py"
        )
        _dataset = importlib.util.module_from_spec(_spec)
        sys.modules[_module_key] = _dataset
        _spec.loader.exec_module(_dataset)
    _CONFIG = _dataset._CONFIG
    _split_dir = _dataset._split_dir

    _sot_key = f"{__name__}._ami_sot_text_impl"
    if _sot_key in sys.modules:
        sot_text = sys.modules[_sot_key]
    else:
        _sot_spec = importlib.util.spec_from_file_location(
            _sot_key, Path(__file__).resolve().parent / "sot_text.py"
        )
        sot_text = importlib.util.module_from_spec(_sot_spec)
        sys.modules[_sot_key] = sot_text
        _sot_spec.loader.exec_module(sot_text)

    _norm_key = f"{__name__}._ami_sot_text_norm_impl"
    if _norm_key in sys.modules:
        _text_norm = sys.modules[_norm_key]
    else:
        _norm_spec = importlib.util.spec_from_file_location(
            _norm_key, Path(__file__).resolve().parent / "text_norm.py"
        )
        _text_norm = importlib.util.module_from_spec(_norm_spec)
        sys.modules[_norm_key] = _text_norm
        _norm_spec.loader.exec_module(_text_norm)
    get_text_norm = _text_norm.get_text_norm


logger = logging.getLogger(__name__)

# Name of the per-split directory the extracted audio segments are written to.
_SEGMENTS_SUBDIR = "segments_wav"

# Written into data_root the moment the builder claims that directory. Its
# absence beside existing split data means the corpus came from somewhere
# else, and the builder will not touch it.
_MARKER_NAME = "data/.ami_sot_builder"


def _relocate_recording(cut, audio_root: Path) -> None:
    """Point a cut's recording at the locally downloaded AMI audio.

    The manifests carry absolute paths from the machine that built them. A
    path that still resolves is left alone, so a prepared corpus keeps
    working; anything else is remapped into ``audio_root`` by basename, which
    is the ``<root>/<meeting>/audio/<meeting>.Array1-01.wav`` layout that
    ``ami_download.sh`` writes.

    Args:
        cut: Lhotse cut, modified in place.
        audio_root: Root the AMI audio was downloaded into.
    """
    for source in getattr(cut.recording, "sources", []):
        if source.type != "file" or Path(source.source).exists():
            continue
        name = Path(source.source).name
        source.source = str(audio_root / name.split(".")[0] / "audio" / name)


class AmiSotBuilder(DatasetBuilder):
    """Build the AMI SOT split directories from cutsets and AMI audio."""

    # ------------------------------------------------------------- helpers
    def _root(self, key: str) -> Path:
        """Resolve a configured directory against ``data_root``."""
        path = Path(_CONFIG[key])
        return path if path.is_absolute() else Path(_CONFIG["data_root"]) / path

    def _cutset_path(self, split: str) -> Path:
        """Return the local path of ``split``'s CutSet manifest."""
        return self._root("cutset_dir") / _CONFIG["sot"][split]["cutset"]

    # -------------------------------------------------------------- source
    def is_source_prepared(self, **kwargs) -> bool:
        """Return True when every cutset and some AMI audio is on disk."""
        if not all(self._cutset_path(s).is_file() for s in _CONFIG["sot"]):
            return False
        audio_root = self._root("ami_audio_root")
        return audio_root.is_dir() and any(audio_root.glob("*/audio/*.wav"))

    def _prepare_source_impl(self, **kwargs) -> None:
        """Download the cutsets and, when absent, the AMI SDM audio.

        Raises:
            RuntimeError: When the AMI download script exits non-zero.
        """
        cutset_dir = self._root("cutset_dir")
        cutset_dir.mkdir(parents=True, exist_ok=True)
        for split, options in _CONFIG["sot"].items():
            target = self._cutset_path(split)
            if target.is_file():
                continue
            logger.info("Fetching %s cutset %s", split, options["cutset"])
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id=_CONFIG["cutset_repo"],
                filename=options["cutset"],
                local_dir=str(cutset_dir),
            )

        audio_root = self._root("ami_audio_root")
        if audio_root.is_dir() and any(audio_root.glob("*/audio/*.wav")):
            return
        audio_root.mkdir(parents=True, exist_ok=True)
        script = (
            Path(__file__).resolve().parents[3]
            / "egs2"
            / "ami"
            / "asr1"
            / "local"
            / "ami_download.sh"
        )
        logger.info("Downloading AMI SDM audio into %s", audio_root)
        # Run from the script's own recipe: it sources utils/parse_options.sh
        # and writes its wget list under that recipe's data/local/downloads.
        result = subprocess.run(
            ["bash", str(script), "sdm1", str(audio_root.resolve())],
            cwd=str(script.parents[1]),
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{script} failed with exit code {result.returncode}. AMI is "
                "distributed by the University of Edinburgh; check network "
                "access and re-run, or set AMI_AUDIO_ROOT to an existing copy."
            )

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
            existing = [n for n in ("text", "wav.scp") if (split_dir / n).is_file()]
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
            "Written by egs3/ami/s2t AmiSotBuilder. Its presence lets the "
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
        """Serialize one split's cutset into a Kaldi-style directory.

        Args:
            split: Key shared by ``builder.split_dirs`` and ``builder.sot``.
        """
        import soundfile as sf
        from lhotse import CutSet

        options = _CONFIG["sot"][split]
        split_dir = _split_dir(split)
        segments_dir = split_dir / _SEGMENTS_SUBDIR
        segments_dir.mkdir(parents=True, exist_ok=True)
        audio_root = self._root("ami_audio_root")
        data_root = Path(_CONFIG["data_root"])

        max_duration = _CONFIG["max_cut_duration"]
        # Resolved once per split rather than per cut: the CHiME-8 normalizer
        # loads a 56 KB spelling table at construction.
        normalizer = get_text_norm(options.get("text_norm"))
        entries = []
        for cut in CutSet.from_file(str(self._cutset_path(split))):
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
            segment_path = segments_dir / f"{cut.id}.wav"
            if not segment_path.is_file():
                # load_audio applies the cut's own offset, duration and
                # channel, so the written file is exactly the group's audio.
                audio = cut.load_audio()
                sf.write(str(segment_path), audio[0], cut.sampling_rate)
            # wav.scp stores paths relative to data_root when it can, which is
            # what makes a prepared directory relocatable.
            try:
                stored = segment_path.relative_to(data_root)
            except ValueError:
                stored = segment_path
            entries.append((cut.id, str(stored), text))

        entries.sort(key=lambda entry: entry[0])
        self._write_kaldi_dir(split_dir, entries)
        logger.info("Wrote %d entries to %s", len(entries), split_dir)

    @staticmethod
    def _write_kaldi_dir(split_dir: Path, entries) -> None:
        """Write the manifests for one split.

        Args:
            split_dir: Destination directory.
            entries: ``(utt_id, wav_path, text)`` triples, already sorted.
        """
        split_dir.mkdir(parents=True, exist_ok=True)
        na = _CONFIG["na_symbol"]
        names = ("wav.scp", "text", "utt2spk", "spk2utt", "text.prev", "text.ctc")
        files = {name: open(split_dir / name, "w") for name in names}
        try:
            for utt_id, wav_path, text in entries:
                files["wav.scp"].write(f"{utt_id} {wav_path}\n")
                files["text"].write(f"{utt_id} {text}\n")
                # Each utterance group is its own speaker under SOT: the
                # speakers live inside the serialized text, not in utt2spk.
                files["utt2spk"].write(f"{utt_id} {utt_id}\n")
                files["spk2utt"].write(f"{utt_id} {utt_id}\n")
                # The S2T model requires both. This recipe conditions on no
                # previous text and trains with ctc_weight 0.0, so both are
                # the not-available symbol.
                files["text.prev"].write(f"{utt_id} {na}\n")
                files["text.ctc"].write(f"{utt_id} {na}\n")
        finally:
            for handle in files.values():
                handle.close()
