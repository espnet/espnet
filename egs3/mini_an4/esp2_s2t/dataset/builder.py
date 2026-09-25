"""Build a tiny serialized-output-training corpus out of the AN4 fixture.

AN4 is single-speaker, so there are no multi-talker groups to read: this
builder makes them. It concatenates utterances from different speakers into
one recording and writes a Lhotse cut over it carrying one supervision per
utterance, which is the shape ``espnet3.systems.esp2_s2t.sot_text`` serializes
and the shape an AMI utterance group already has.

Everything comes from the archive committed beside this file, so the recipe
runs with no network.
"""

from __future__ import annotations

import logging
import tarfile
from importlib import resources
from pathlib import Path
from typing import Dict, List, Tuple

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.systems.esp2_s2t import sot_text
from espnet3.systems.esp2_s2t.text_norm import get_text_norm
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)

# Both must match dataset/dataset.py.
_CUTS_FILENAME = "cuts.jsonl.gz"
_SOT_TEXT_FIELD = "sot_text"

_SAMPLE_RATE = 16000


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CONFIG = _load_builder_config()


def _split_dir(split: str) -> Path:
    """Return the directory holding one split's cuts.

    Args:
        split: Key of ``builder.split_dirs``.

    Returns:
        ``<data_root>/<split_dirs[split]>``.

    Raises:
        KeyError: When ``split`` is not a configured split.
    """
    split_dirs = _CONFIG["split_dirs"]
    if split not in split_dirs:
        raise KeyError(
            f"Unknown split {split!r}; expected one of {sorted(split_dirs)}."
        )
    return Path(_CONFIG["data_root"]) / split_dirs[split]


def parse_transcript(path: Path) -> Dict[str, str]:
    """Return ``{utterance id: text}`` from an AN4 transcription file.

    Args:
        path: Transcription file, whose lines read ``<s> WORDS </s> (utt-id)``.

    Returns:
        The text of every utterance, without the sentence markers.
    """
    texts = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        words, _, utt = line.rpartition("(")
        words = words.strip().removeprefix("<s>").removesuffix("</s>").strip()
        texts[utt.rstrip(")").strip()] = words.lower()
    return texts


class MiniAn4SotBuilder(DatasetBuilder):
    """Write the three splits of the mini AN4 SOT corpus."""

    # ------------------------------------------------------------- helpers
    def _get_root(self, key: str) -> Path:
        """Resolve a configured directory against ``data_root``."""
        path = Path(_CONFIG[key])
        return path if path.is_absolute() else Path(_CONFIG["data_root"]) / path

    def _get_speaker_utterances(self, split: str) -> List[Tuple[str, str, Path]]:
        """Return ``(speaker, text, audio)`` for one split's source directory.

        AN4 ships NIST SPHERE, which libsndfile reads, so nothing has to be
        converted and the recipe needs no sph2pipe. The speaker is the
        directory the file sits in, which is how AN4 names them.

        Args:
            split: Key of ``builder.sot``.

        Returns:
            One entry per AN4 utterance, ordered by utterance id so a rebuild
            groups the same utterances together.
        """
        options = _CONFIG["sot"][split]
        an4 = self._get_root("downloads_dir") / "an4"
        texts = parse_transcript(an4 / "etc" / options["transcript"])

        return [
            (sph.parent.name, texts[sph.stem], sph)
            for sph in sorted((an4 / "wav" / options["source"]).glob("*/*.sph"))
            if sph.stem in texts
        ]

    # -------------------------------------------------------------- source
    def is_source_prepared(self, **kwargs) -> bool:
        """Report whether the archive has been expanded. Reads disk only."""
        an4 = self._get_root("downloads_dir") / "an4"
        return an4.is_dir() and any(an4.glob("wav/*/*/*.sph"))

    def prepare_source(self, **kwargs) -> None:
        """Expand the committed archive. Nothing is downloaded."""
        archive = self._get_root("archive_path")
        target = Path(_CONFIG["data_root"])
        logger.info("Extracting %s into %s", archive, target)
        with tarfile.open(archive) as tar:
            tar.extractall(target)

    # --------------------------------------------------------------- build
    def is_built(self, **kwargs) -> bool:
        """Report whether every split holds the files a split is made of."""
        for split in _CONFIG["split_dirs"]:
            split_dir = _split_dir(split)
            for name in _CONFIG["required_files"]:
                if not (split_dir / name).is_file():
                    return False
        return True

    def build(self, **kwargs) -> None:
        """Write every split, then the vocabulary they are serialized against."""
        if not self.is_source_prepared():
            self.prepare_source()
        for split in _CONFIG["split_dirs"]:
            self._build_split(split)
        self._write_token_list()

    def _write_token_list(self) -> None:
        """Export the Whisper vocabulary this corpus is serialized against.

        Offline: the exporter reads the tokenizer files that ship inside the
        openai-whisper package.
        """
        from espnet3.systems.esp2_s2t.whisper_export_vocabulary import (
            export_vocabulary,
        )

        token_list = self._get_root("token_list")
        if token_list.is_file():
            return
        token_list.parent.mkdir(parents=True, exist_ok=True)
        export_vocabulary(
            output=str(token_list),
            whisper_model="whisper_multilingual",
            log_level="INFO",
            sot_asr=True,
            speaker_change_symbol=_CONFIG["separator"],
        )

    def _build_split(self, split: str) -> None:
        """Write one split's CutSet.

        Args:
            split: Key of ``builder.split_dirs`` and of ``builder.sot``.
        """
        import numpy as np
        import soundfile as sf
        from lhotse import CutSet, MonoCut, Recording, SupervisionSegment

        options = _CONFIG["sot"][split]
        normalizer = get_text_norm(options.get("text_norm"))
        per_group = int(_CONFIG["utterances_per_group"])
        gap = float(_CONFIG["gap"])

        entries = self._get_speaker_utterances(split)
        split_dir = _split_dir(split)
        audio_dir = split_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        cuts = []
        for index in range(int(options["groups"])):
            members = [
                entries[(index * per_group + offset) % len(entries)]
                for offset in range(per_group)
            ]
            waveform, supervisions, offset = [], [], 0.0
            for position, (speaker, text, audio) in enumerate(members):
                samples, rate = sf.read(str(audio), dtype="float32")
                assert rate == _SAMPLE_RATE, f"{audio} is {rate} Hz"
                if position:
                    waveform.append(np.zeros(int(gap * rate), dtype="float32"))
                    offset += gap
                waveform.append(samples)
                supervisions.append(
                    SupervisionSegment(
                        id=f"{split}-{index}-{position}",
                        recording_id=f"{split}-{index}",
                        start=offset,
                        duration=len(samples) / rate,
                        text=text,
                        # Two AN4 speakers per group, so the group is
                        # multi-talker even though AN4 itself is not.
                        speaker=speaker,
                    )
                )
                offset += len(samples) / rate
            group_wav = audio_dir / f"{split}-{index}.wav"
            sf.write(str(group_wav), np.concatenate(waveform), _SAMPLE_RATE)

            recording = Recording.from_file(
                str(group_wav), recording_id=f"{split}-{index}"
            )
            cut = MonoCut(
                id=f"{split}-{index}",
                start=0.0,
                duration=recording.duration,
                channel=0,
                recording=recording,
                supervisions=supervisions,
            )
            cut.custom = dict(cut.custom or {})
            cut.custom[_SOT_TEXT_FIELD] = sot_text.build_sot_text(
                supervisions,
                max_timestamp_pause=options["max_timestamp_pause"],
                ordering=options["ordering"],
                separator=_CONFIG["separator"],
                lowercase=options["lowercase"],
                text_norm=normalizer,
                prompt=_CONFIG["prompt"],
                eos=None,
            )
            cuts.append(cut)

        CutSet.from_cuts(cuts).to_file(str(split_dir / _CUTS_FILENAME))
        logger.info("Wrote %d cuts to %s", len(cuts), split_dir / _CUTS_FILENAME)
