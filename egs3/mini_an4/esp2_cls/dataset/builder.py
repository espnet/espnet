"""Mini AN4 classification dataset builder."""

from __future__ import annotations

import re
import shutil
import subprocess
from importlib import resources
from pathlib import Path

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

TRANSCRIPT_RE = re.compile(r"^(?P<words>.+?)\s+\((?P<src>[^)]+)\)\s*$")


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()


def _parse_transcript_line(line: str) -> tuple[str, str, str]:
    """Parse one AN4 transcription line, returning ``(utt_id, source_id, speaker)``."""
    m = TRANSCRIPT_RE.match(line)
    if m is None:
        raise ValueError(f"Malformed transcript line: {line!r}")
    src = m.group("src")
    utterance, speaker, number = src.split("-")
    return f"{speaker}-{utterance}-{number}", src, speaker


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the manifests are written into."""
    return recipe_root / _CFG["data_path"]


class MiniAn4ClsBuilder(DatasetBuilder):
    """Prepare Mini AN4 as a speaker-classification corpus.

    The transcripts are ignored: the label is the speaker id read off the
    AN4 file id, which ``dataset/config.yaml`` explains.
    """

    def is_source_prepared(self, recipe_dir: str | Path, **kwargs) -> bool:
        """Check whether the raw AN4 source files are already available."""
        an4 = Path(recipe_dir).resolve() / _CFG["dataset_path"] / "an4"
        return (an4 / "etc").is_dir() and (an4 / "wav").is_dir()

    def prepare_source(self, recipe_dir: str | Path, **kwargs) -> None:
        """Unpack the AN4 archive under ``dataset_path``.

        Args:
            recipe_dir: Recipe root directory.
            **kwargs: Unused extra options for API compatibility.

        Raises:
            FileNotFoundError: If the configured archive does not exist.
            subprocess.CalledProcessError: If ``tar`` extraction fails.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir=recipe_root):
            return
        archive = (recipe_root / _CFG["archive_path"]).resolve()
        if not archive.exists():
            raise FileNotFoundError(f"Archive not found: {archive}")
        # The archive's own top-level directory is `downloads/`, so it is
        # unpacked into the recipe root rather than into `dataset_path`.
        subprocess.run(
            ["tar", "-xzf", str(archive), "-C", str(recipe_root)], check=True
        )

    def is_built(self, recipe_dir: str | Path, **kwargs) -> bool:
        """Check whether every split manifest already exists."""
        data = resolve_data_root(Path(recipe_dir).resolve())
        return all((data / p).is_file() for p in _CFG["manifest_paths"].values())

    def build(self, recipe_dir: str | Path, **kwargs) -> None:
        """Convert the SPH files to WAV and write one manifest per split.

        Args:
            recipe_dir: Recipe root directory.
            **kwargs: Unused extra options for API compatibility.

        Raises:
            RuntimeError: If the source tree is incomplete, if ``sph2pipe`` is
                not installed, or if a configured validation utterance is
                absent from the training split.
            ValueError: If a transcript line has an unexpected format.
            subprocess.CalledProcessError: If audio conversion fails.
        """
        recipe_root = Path(recipe_dir).resolve()
        data = resolve_data_root(recipe_root)
        an4 = recipe_root / _CFG["dataset_path"] / "an4"
        if not ((an4 / "etc").is_dir() and (an4 / "wav").is_dir()):
            raise RuntimeError(f"Source not prepared: {an4}")

        sph2pipe = shutil.which("sph2pipe")
        if not sph2pipe:
            raise RuntimeError("sph2pipe not found in PATH")

        test_speaker_labels = {
            str(k): str(v) for k, v in _CFG["test_speaker_labels"].items()
        }

        split_entries: dict[str, list[tuple[str, Path, str]]] = {}
        for split, spec in _CFG["splits"].items():
            entries = []
            transcript = an4 / "etc" / spec["transcript_name"]
            for raw_line in transcript.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                utt_id, src, speaker = _parse_transcript_line(line)
                sph = an4 / "wav" / spec["sph_subdir"] / speaker / f"{src}.sph"
                wav = (data / "wav" / split / f"{utt_id}.wav").resolve()
                wav.parent.mkdir(parents=True, exist_ok=True)
                if not wav.exists():
                    part = wav.with_suffix(wav.suffix + ".part")
                    try:
                        with part.open("wb") as fh:
                            subprocess.run(
                                [sph2pipe, "-f", "wav", "-p", "-c", "1", str(sph)],
                                stdout=fh,
                                check=True,
                            )
                        part.replace(wav)
                    except BaseException:
                        part.unlink(missing_ok=True)
                        raise
                label = test_speaker_labels.get(speaker, speaker)
                entries.append((utt_id, wav, label))
            split_entries[split] = sorted(entries)

        valid_utt_ids = {str(utt_id) for utt_id in _CFG["valid_utt_ids"]}
        train_pool = split_entries["train"]
        missing = valid_utt_ids - {utt_id for utt_id, _, _ in train_pool}
        if missing:
            raise RuntimeError(
                f"valid_utt_ids not found in the training split: {sorted(missing)}"
            )

        to_write = {
            "train": [e for e in train_pool if e[0] not in valid_utt_ids],
            "valid": [e for e in train_pool if e[0] in valid_utt_ids],
            "test": split_entries["test"],
        }
        for split, entries in to_write.items():
            manifest = data / _CFG["manifest_paths"][split]
            manifest.parent.mkdir(parents=True, exist_ok=True)
            # Write to a part file and rename, so an interrupted build cannot
            # leave a truncated manifest that `is_built` would accept.
            part = manifest.with_suffix(manifest.suffix + ".part")
            with part.open("w", encoding="utf-8") as fh:
                for utt_id, wav, label in entries:
                    fh.write(f"{utt_id}\t{wav}\t{label}\n")
            part.replace(manifest)
