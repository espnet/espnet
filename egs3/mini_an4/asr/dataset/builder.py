"""Mini AN4 dataset builder."""

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


def _normalize_downloads_layout(dataset_root: Path) -> None:
    """Normalize extracted archive layout to ``downloads/an4``.

    Some archives expand as ``downloads/downloads/*``. This function moves
    the nested entries one level up so the final layout is always:
    ``downloads/an4`` (and optionally ``downloads/noise``, ``downloads/rirs``).
    """
    nested_root = dataset_root / "downloads"
    if not nested_root.is_dir():
        return

    for child in nested_root.iterdir():
        target = dataset_root / child.name
        if target.exists():
            continue
        child.rename(target)

    if not any(nested_root.iterdir()):
        nested_root.rmdir()


def _parse_transcript_line(line: str) -> tuple[str, str, str]:
    """Parse one AN4 transcription line, returning ``(utt_id, source_id, text)``."""
    m = TRANSCRIPT_RE.match(line)
    if m is None:
        raise ValueError(f"Malformed transcript line: {line!r}")
    words = m.group("words").removeprefix("<s> ").removesuffix(" </s>")
    src = m.group("src")
    speaker, utterance, number = src.split("-")
    return f"{utterance}-{speaker}-{number}", src, words


class MiniAn4Builder(DatasetBuilder):
    """Prepare and build Mini AN4 assets for ESPnet3 recipes."""

    def is_source_prepared(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check whether raw AN4 source files are already available.

        Args:
            recipe_dir: Recipe root directory.
            **_kwargs: Unused extra options for API compatibility.

        Returns:
            ``True`` if ``<recipe_dir>/<dataset_path>/an4/etc`` and
            ``<recipe_dir>/<dataset_path>/an4/wav`` both exist; otherwise
            ``False``.

        Raises:
            None.

        Notes:
            This check is intentionally cheap and only verifies source-tree
            presence, not completeness of generated manifests.

        Examples:
            >>> builder = MiniAn4Builder()
            >>> builder.is_source_prepared("egs3/mini_an4/asr")
            True
        """
        recipe_root = Path(recipe_dir).resolve()
        an4 = recipe_root / _CFG["dataset_path"] / "an4"
        return (an4 / "etc").is_dir() and (an4 / "wav").is_dir()

    def prepare_source(self, recipe_dir: str | Path, **_kwargs) -> None:
        """Prepare raw AN4 source data under ``dataset_path``.

        Args:
            recipe_dir: Recipe root directory.
            **_kwargs: Unused extra options for API compatibility.

        Returns:
            None.

        Raises:
            FileNotFoundError: If configured archive file does not exist.
            subprocess.CalledProcessError: If ``tar`` extraction fails.

        Notes:
            This method:
            1. Resolves and validates the archive path.
            2. Ensures destination directory exists.
            3. Normalizes accidental nested layout like ``downloads/downloads``.
            4. Extracts the archive only when source is not already prepared.

        Examples:
            >>> builder = MiniAn4Builder()
            >>> builder.prepare_source("egs3/mini_an4/asr")
        """
        recipe_root = Path(recipe_dir).resolve()
        dataset_root = recipe_root / _CFG["dataset_path"]
        archive = (recipe_root / _CFG["archive_path"]).resolve()
        if not archive.exists():
            raise FileNotFoundError(f"Archive not found: {archive}")
        dataset_root.mkdir(parents=True, exist_ok=True)
        _normalize_downloads_layout(dataset_root)
        if self.is_source_prepared(recipe_dir=recipe_dir):
            return
        subprocess.run(
            ["tar", "-xzf", str(archive), "-C", str(dataset_root)],
            check=True,
        )
        _normalize_downloads_layout(dataset_root)

    def is_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check whether task-ready manifest files are already built.

        Args:
            recipe_dir: Recipe root directory.
            **_kwargs: Unused extra options for API compatibility.

        Returns:
            ``True`` when all manifest files defined in
            ``builder.manifest_paths`` exist under ``data_path``.

        Raises:
            None.

        Notes:
            This verifies only presence of expected output files, not their
            semantic correctness.

        Examples:
            >>> builder = MiniAn4Builder()
            >>> builder.is_built("egs3/mini_an4/asr")
            True
        """
        recipe_root = Path(recipe_dir).resolve()
        data = recipe_root / _CFG["data_path"]
        return all((data / p).is_file() for p in _CFG["manifest_paths"].values())

    def build(self, recipe_dir: str | Path, **_kwargs) -> None:
        """Build train/valid/test manifests and converted WAV files.

        Args:
            recipe_dir: Recipe root directory.
            **_kwargs: Unused extra options for API compatibility.

        Returns:
            None.

        Raises:
            RuntimeError: If source tree is missing required ``etc``/``wav``
                directories, or if ``sph2pipe`` is not installed.
            ValueError: If a transcript line has unexpected format.
            subprocess.CalledProcessError: If audio conversion command fails.

        Notes:
            Build flow:
            1. Read AN4 transcripts for configured splits.
            2. Convert source SPH to mono WAV (via ``sph2pipe``) under
               ``data/wav/<split>/``.
            3. Sort entries and split a small prefix of train into valid
               using ``dev_size``.
            4. Write TSV manifests for ``train``, ``valid``, ``test``.

        Examples:
            >>> builder = MiniAn4Builder()
            >>> builder.build("egs3/mini_an4/asr")
        """
        recipe_root = Path(recipe_dir).resolve()
        data = recipe_root / _CFG["data_path"]
        an4 = recipe_root / _CFG["dataset_path"] / "an4"
        if not ((an4 / "etc").is_dir() and (an4 / "wav").is_dir()):
            raise RuntimeError(f"Source not prepared: {an4}")

        sph2pipe = shutil.which("sph2pipe")
        if not sph2pipe:
            raise RuntimeError("sph2pipe not found in PATH")

        dev_size = int(_CFG["dev_size"])

        split_entries: dict[str, list[tuple[str, Path, str]]] = {}
        for split, spec in _CFG["splits"].items():
            entries = []
            transcript = an4 / "etc" / spec["transcript_name"]
            for raw_line in transcript.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                utt_id, src, text = _parse_transcript_line(line)
                sph = (
                    an4 / "wav" / spec["sph_subdir"] / src.split("-")[1] / f"{src}.sph"
                )
                wav = (data / "wav" / split / f"{utt_id}.wav").resolve()
                wav.parent.mkdir(parents=True, exist_ok=True)
                if not wav.exists():
                    with wav.open("wb") as fh:
                        subprocess.run(
                            [sph2pipe, "-f", "wav", "-p", "-c", "1", str(sph)],
                            stdout=fh,
                            check=True,
                        )
                entries.append((utt_id, wav, text))
            split_entries[split] = sorted(entries)

        train = split_entries["train"]
        if len(train) < dev_size + 1:
            raise RuntimeError(f"Training data too small (got {len(train)})")

        to_write = {
            "valid": train[:dev_size],
            "train": train[dev_size:],
            "test": split_entries["test"],
        }
        for split, entries in to_write.items():
            path = data / _CFG["manifest_paths"][split]
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                for utt_id, wav, text in entries:
                    fh.write(f"{utt_id}\t{wav}\t{text}\n")
