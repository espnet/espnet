"""Mini AN4 LID dataset builder."""

from __future__ import annotations

import re
import shutil
import subprocess
from importlib import resources
from pathlib import Path

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import extract_targz

TRANSCRIPT_RE = re.compile(r"^(?P<words>.+?)\s+\((?P<src>[^)]+)\)\s*$")


def _load_builder_config() -> dict:
    """Read the builder section of ``dataset/config.yaml``."""
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()


def _resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the manifests and converted WAVs are written into."""
    return recipe_root / _CFG["data_path"]


def resolve_manifest_path(recipe_root: Path, split: str) -> Path:
    """Return the manifest of ``split`` (``train``, ``valid`` or ``test``).

    Raises:
        ValueError: If ``split`` is unknown.
    """
    manifest_paths = _CFG["manifest_paths"]
    if split not in manifest_paths:
        known = ", ".join(sorted(manifest_paths))
        raise ValueError(f"Unknown split '{split}'. Expected one of: {known}")
    return _resolve_data_root(recipe_root) / manifest_paths[split]


def _parse_transcript_line(line: str) -> tuple[str, str, str]:
    """Parse one AN4 transcription line, returning ``(utt_id, source_id, speaker)``.

    The utterance ID starts with the speaker, as in egs2/mini_an4/lid1, so
    sorting groups each language's utterances together.
    """
    m = TRANSCRIPT_RE.match(line)
    if m is None:
        raise ValueError(f"Malformed transcript line: {line!r}")
    src = m.group("src")
    utterance, speaker, number = src.split("-")
    return f"{speaker}-{utterance}-{number}", src, speaker


class MiniAn4LIDBuilder(DatasetBuilder):
    """Prepare Mini AN4 as a language-identification corpus.

    Transcripts are ignored: the language label is the speaker id read off the
    AN4 file id, which ``dataset/config.yaml`` explains.
    """

    def is_source_prepared(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check whether the raw AN4 source files are already available."""
        an4_dir = Path(recipe_dir).resolve() / _CFG["dataset_path"] / "an4"
        return (an4_dir / "etc").is_dir() and (an4_dir / "wav").is_dir()

    def prepare_source(self, recipe_dir: str | Path, **_kwargs) -> None:
        """Unpack the AN4 archive under ``dataset_path``.

        Args:
            recipe_dir: Recipe root directory.
            **_kwargs: Unused extra options for API compatibility.

        Raises:
            FileNotFoundError: If the configured archive does not exist.
            tarfile.TarError: If archive extraction fails.

        Example:
            >>> MiniAn4LIDBuilder().prepare_source("egs3/mini_an4/esp2_lid")
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir=recipe_root):
            return
        archive = (recipe_root / _CFG["archive_path"]).resolve()
        if not archive.exists():
            raise FileNotFoundError(f"Archive not found: {archive}")
        # The archive's own top-level directory is `downloads/`, so it is
        # unpacked into the recipe root rather than into `dataset_path`.
        extract_targz(archive, recipe_root)

    def is_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check whether every split manifest already exists."""
        data_dir = _resolve_data_root(Path(recipe_dir).resolve())
        return all((data_dir / p).is_file() for p in _CFG["manifest_paths"].values())

    def build(self, recipe_dir: str | Path, **_kwargs) -> None:
        """Convert the SPH files to WAV and write one manifest per split.

        Each manifest line is ``utt_id<TAB>wav_path<TAB>language``. Training
        keeps all but the first ``train_skip`` training utterances, validation
        takes the first ``valid_size``, and test speakers are relabelled with
        ``test_lang_map``.

        Args:
            recipe_dir: Recipe root directory.
            **_kwargs: Unused extra options for API compatibility.

        Raises:
            RuntimeError: If the source tree is incomplete, if ``sph2pipe`` is
                not installed, or if a test speaker has no mapped language.
            ValueError: If a transcript line has an unexpected format.
            subprocess.CalledProcessError: If audio conversion fails.

        Example:
            >>> MiniAn4LIDBuilder().build("egs3/mini_an4/esp2_lid")
        """
        recipe_root = Path(recipe_dir).resolve()
        data_dir = _resolve_data_root(recipe_root)
        an4_dir = recipe_root / _CFG["dataset_path"] / "an4"
        if not ((an4_dir / "etc").is_dir() and (an4_dir / "wav").is_dir()):
            raise RuntimeError(f"Source not prepared: {an4_dir}")

        sph2pipe = shutil.which("sph2pipe")
        if not sph2pipe:
            raise RuntimeError("sph2pipe not found in PATH")

        test_lang_map = {str(k): str(v) for k, v in _CFG["test_lang_map"].items()}

        split_entries: dict[str, list[tuple[str, Path, str]]] = {}
        for split, spec in _CFG["splits"].items():
            entries = []
            transcript = an4_dir / "etc" / spec["transcript_name"]
            for raw_line in transcript.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                utt_id, src, speaker = _parse_transcript_line(line)
                if split == "test":
                    if speaker not in test_lang_map:
                        raise RuntimeError(f"No language mapped for {speaker}")
                    language = test_lang_map[speaker]
                else:
                    language = speaker
                sph = an4_dir / "wav" / spec["sph_subdir"] / speaker / f"{src}.sph"
                wav = (data_dir / "wav" / split / f"{utt_id}.wav").resolve()
                wav.parent.mkdir(parents=True, exist_ok=True)
                if not wav.exists():
                    part = wav.with_suffix(wav.suffix + ".part")
                    with part.open("wb") as fh:
                        subprocess.run(
                            [sph2pipe, "-f", "wav", "-p", "-c", "1", str(sph)],
                            stdout=fh,
                            check=True,
                        )
                    part.replace(wav)
                entries.append((utt_id, wav, language))
            split_entries[split] = sorted(entries)

        train = split_entries["train"]
        to_write = {
            "train": train[int(_CFG["train_skip"]) :],
            "valid": train[: int(_CFG["valid_size"])],
            "test": split_entries["test"],
        }
        for split, entries in to_write.items():
            manifest = data_dir / _CFG["manifest_paths"][split]
            manifest.parent.mkdir(parents=True, exist_ok=True)
            # Write to a part file and rename, so an interrupted build cannot
            # leave a truncated manifest that `is_built` would accept.
            part = manifest.with_suffix(manifest.suffix + ".part")
            with part.open("w", encoding="utf-8") as fh:
                for utt_id, wav, language in entries:
                    fh.write(f"{utt_id}\t{wav}\t{language}\n")
            part.replace(manifest)
