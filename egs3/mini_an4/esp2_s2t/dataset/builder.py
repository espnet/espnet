"""Build a small SOT corpus from the AN4 archive.

AN4 has one speaker per utterance, so it has no multi-talker groups. This
builder makes groups by joining AN4 utterances into one recording and writing
a Lhotse cut for that recording, with one supervision for each utterance.
``espnet3.systems.esp2_s2t.src.sot.sot_text`` serializes cuts of this form,
and AMI utterance groups have the same form.

The builder reads only the archive in the recipe directory and downloads
nothing.
"""

from __future__ import annotations

import logging
import tarfile
from importlib import resources
from pathlib import Path
from typing import Dict, List, Tuple

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.systems.esp2_s2t.src.sot import sot_text
from espnet3.systems.esp2_s2t.src.sot.text_norm import get_text_norm
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)

# Both values must be the same as in dataset/dataset.py.
_CUTS_FILENAME = "cuts.jsonl.gz"
_SOT_TEXT_FIELD = "sot_text"

_SAMPLE_RATE = 16000


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CONFIG = _load_builder_config()


def _split_dir(split: str) -> Path:
    """Return the directory of one split.

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
            f"Unknown split {split!r}. Expected one of {sorted(split_dirs)}."
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

    def _get_root(self, key: str) -> Path:
        """Return a configured directory, relative to ``data_root`` if not absolute."""
        path = Path(_CONFIG[key])
        return path if path.is_absolute() else Path(_CONFIG["data_root"]) / path

    def _get_speaker_utterances(self, split: str) -> List[Tuple[str, str, Path]]:
        """Return ``(speaker, text, audio)`` for the source files of one split.

        AN4 stores audio as NIST SPHERE files, which libsndfile reads
        directly, so nothing is converted and the recipe does not need
        sph2pipe. AN4 puts the files of each speaker in a directory named after
        the speaker.

        Args:
            split: Key of ``builder.sot``.

        Returns:
            One entry for each AN4 utterance, sorted by file path, so a rebuild
            puts the same utterances in the same groups.
        """
        options = _CONFIG["sot"][split]
        an4 = self._get_root("downloads_dir") / "an4"
        texts = parse_transcript(an4 / "etc" / options["transcript"])

        return [
            (sph.parent.name, texts[sph.stem], sph)
            for sph in sorted((an4 / "wav" / options["source"]).glob("*/*.sph"))
            if sph.stem in texts
        ]

    def is_source_prepared(self, **kwargs) -> bool:
        """Return True when the archive has been extracted."""
        an4 = self._get_root("downloads_dir") / "an4"
        return an4.is_dir() and any(an4.glob("wav/*/*/*.sph"))

    def prepare_source(self, **kwargs) -> None:
        """Extract the AN4 archive. Nothing is downloaded."""
        archive = self._get_root("archive_path")
        target = Path(_CONFIG["data_root"])
        logger.info("Extracting %s into %s", archive, target)
        with tarfile.open(archive) as tar:
            tar.extractall(target)

    def is_built(self, **kwargs) -> bool:
        """Return True when every split directory has its required files."""
        for split in _CONFIG["split_dirs"]:
            split_dir = _split_dir(split)
            for name in _CONFIG["required_files"]:
                if not (split_dir / name).is_file():
                    return False
        return True

    def build(self, **kwargs) -> None:
        """Write every split, then the token list."""
        if not self.is_source_prepared():
            self.prepare_source()
        for split in _CONFIG["split_dirs"]:
            self._build_split(split)
        self._write_token_list()

    def _write_token_list(self) -> None:
        """Export the Whisper vocabulary as the token list.

        This works offline, because the exporter reads the tokenizer files that
        come with the openai-whisper package.
        """
        from espnet3.systems.esp2_s2t.src.sot.whisper_export_vocabulary import (
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
        """Write the CutSet of one split.

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
