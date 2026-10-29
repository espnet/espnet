"""Prepare the official VOiCES devkit with the ESPnet2 speaker split."""

import csv
import hashlib
import json
from importlib import resources
from pathlib import Path

import soundfile as sf
from omegaconf import OmegaConf

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url, extract_targz

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _BUILDER_CFG = OmegaConf.to_container(
        load_config_with_defaults(str(_CONFIG_PATH), resolve=False)["builder"],
        resolve=True,
    )

RECIPE_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")
FIELDS = ("utt_id", "path", "text", "speaker", "source_id", "condition", "samples")


def write_text(path, text):
    """Atomically publish UTF-8 text to a generated artifact.

    Args:
        path: Destination; missing parent directories are created.
        text: Complete new contents.

    Raises:
        OSError: The temporary file cannot be written or published.

    Returns:
        None. Completion is recorded by the files written to disk.

    Examples:
        >>> write_text(Path("data/note.txt"), "prepared")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _manifest_digest(path):
    """Hash a manifest without loading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path):
    """Read prepared rows, preserving metadata outside model samples.

    Args:
        path: Header-bearing TSV written by VoicesBuilder.

    Returns:
        Ordered row dictionaries with integer sample counts.

    Raises:
        FileNotFoundError: Preparation has not produced this manifest.
        ValueError: A sample count is invalid.

    Examples:
        After the create_dataset stage:

        >>> rows = read_manifest(Path("data/manifest/train.tsv"))
        >>> len(rows) > 0
        True
    """
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    for row in rows:
        row["samples"] = int(row["samples"])
    return rows


def resolve_source_root(recipe_dir, source_dir=None, corpus="devkit"):
    """Resolve the extracted source without downloading anything.

    Args:
        recipe_dir: Recipe root containing the downloads directory.
        source_dir: Optional explicit extracted root.
        corpus: devkit or full; selects VOiCES_devkit or VOiCES_rebuilt.

    Returns:
        Absolute source directory.

    Raises:
        ValueError: corpus is neither devkit nor full.

    Examples:
        >>> resolve_source_root(".", source_dir="/datasets/VOiCES_devkit")
        PosixPath('/datasets/VOiCES_devkit')
    """
    if corpus not in ("devkit", "full"):
        raise ValueError(f"Unknown VOiCES corpus: {corpus}")
    if source_dir is not None:
        return Path(source_dir).expanduser().resolve()
    name = _BUILDER_CFG["source_name"] if corpus == "devkit" else "VOiCES_rebuilt"
    return Path(recipe_dir).resolve() / "downloads" / name


def _source_id(utterance):
    parts = utterance.split("-")
    if len(parts) < 7 or parts[:3] != ["Lab41", "SRI", "VOiCES"]:
        raise ValueError(f"Invalid VOiCES recording name: {utterance}")
    if parts[3] == "src":
        return utterance, parts[4].removeprefix("sp")
    if len(parts) != 12 or not parts[5].startswith("sp"):
        raise ValueError(f"Invalid distant recording name: {utterance}")
    return "-".join(parts[:3] + ["src"] + parts[5:8]), parts[5][2:]


def _load_transcripts(source_root):
    transcripts = {}
    with (source_root / "references/filename_transcripts").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = csv.reader(stream)
        next(rows)  # The official file has a header and three columns.
        for row in rows:
            if len(row) != 3:
                raise ValueError(f"Expected three transcript columns, got {row!r}")
            utterance = Path(row[1]).stem
            source_id, _ = _source_id(utterance)
            text = " ".join(row[2].split())
            if not text or transcripts.get(source_id, text) != text:
                raise ValueError(f"Missing or conflicting transcription: {source_id}")
            transcripts[source_id] = text
    return transcripts


def _scan_recordings(source_root, split, transcripts, corpus="devkit"):
    rows = []
    for condition, directory in (
        ("distant", "distant-16k/speech"),
        ("source", "source-16k"),
    ):
        paths = sorted((source_root / directory / split).rglob("*.wav"))
        if not paths:
            raise FileNotFoundError(f"No {condition} audio found for {split}")
        if condition == "distant":
            counts = _BUILDER_CFG[
                (
                    "expected_distant_counts"
                    if corpus == "devkit"
                    else "full_distant_counts"
                )
            ]
            expected = int(counts[split])
            if len(paths) != expected:
                raise ValueError(
                    f"Incomplete {corpus} {split}: expected {expected} distant files, "
                    f"found {len(paths)}"
                )
        for path in paths:
            source_id, speaker = _source_id(path.stem)
            info = sf.info(path)
            if info.samplerate != _BUILDER_CFG["sample_rate"] or info.channels != 1:
                raise ValueError(f"Expected mono 16 kHz audio: {path}")
            rows.append(
                dict(
                    utt_id=f"{speaker}_{path.stem}",
                    path=str(path.resolve()),
                    text=transcripts[source_id],
                    speaker=speaker,
                    source_id=source_id,
                    condition=condition,
                    samples=info.frames,
                )
            )
    source_ids = {row["source_id"] for row in rows if row["condition"] == "source"}
    distant_ids = {row["source_id"] for row in rows if row["condition"] == "distant"}
    if source_ids != distant_ids:
        raise ValueError(f"Source/distant utterance sets differ in {split}")
    return sorted(rows, key=lambda row: row["utt_id"])


class VoicesBuilder(DatasetBuilder):
    """Download and index the official devkit, including source and distant audio.

    ``source_dir`` names an existing extracted VOiCES_devkit directory. Without
    that override, ``prepare_source`` downloads only the official devkit archive
    into ``recipe_dir/downloads``. Manifests are written to
    ``recipe_dir/data/manifest``; audio is neither copied nor resampled.
    """

    def is_source_prepared(
        self, recipe_dir, source_dir=None, corpus="devkit", **_kwargs
    ):
        """Check source directories and references without decoding audio.

        Args:
            recipe_dir: Recipe root.
            source_dir: Optional extracted corpus root.
            corpus: devkit or full, selecting the expected directory name.
            **_kwargs: Unused stage settings.

        Returns:
            Whether both recording conditions and transcript references exist.

        Examples:
            >>> builder = VoicesBuilder()
            >>> ready = builder.is_source_prepared(recipe_dir=".")
        """
        root = resolve_source_root(recipe_dir, source_dir, corpus)
        return (root / "references/filename_transcripts").is_file() and all(
            (root / directory / split).is_dir()
            for directory in ("distant-16k/speech", "source-16k")
            for split in ("train", "test")
        )

    def prepare_source(self, recipe_dir, source_dir=None, corpus="devkit", **_kwargs):
        """Download the selected archive, or validate an existing source.

        Args:
            recipe_dir: Root owning downloads and manifests.
            source_dir: Existing extracted root, or None to download the archive.
            corpus: devkit or full; the latter downloads the full release.
            **_kwargs: Unused stage settings.

        Raises:
            FileNotFoundError: The explicit or extracted layout is incomplete.
            ValueError: The corpus selector is invalid.

        Returns:
            None. Completion is recorded by the files written to disk.

        Examples:
            Download the configured corpus when it is not already present:

            >>> builder = VoicesBuilder()
            >>> builder.prepare_source(recipe_dir=".")
        """
        if self.is_source_prepared(recipe_dir, source_dir, corpus):
            return
        if source_dir is not None:
            raise FileNotFoundError(f"Incomplete extracted VOiCES devkit: {source_dir}")
        directory = Path(recipe_dir).resolve() / "downloads"
        archive_name = (
            _BUILDER_CFG["archive_name"]
            if corpus == "devkit"
            else "VOiCES_release.tar.gz"
        )
        archive = directory / archive_name
        if not archive.is_file():
            temporary = archive.with_suffix(archive.suffix + ".part")
            download_url(
                _BUILDER_CFG["url"] if corpus == "devkit" else _BUILDER_CFG["full_url"],
                temporary,
            )
            temporary.replace(archive)
        extract_targz(archive, directory)
        if not self.is_source_prepared(recipe_dir, corpus=corpus):
            raise FileNotFoundError(f"Expected VOiCES devkit layout under {directory}")

    def is_built(self, recipe_dir, source_dir=None, corpus="devkit", **_kwargs):
        """Check the source, corpus mode and complete current-format outputs.

        Args:
            recipe_dir: Root containing data/manifest and data/lm.
            source_dir: Optional explicit extracted corpus root.
            corpus: devkit or full, matching the preparation request.
            **_kwargs: Unused stage settings.

        Returns:
            Whether settings and manifest hashes match the build marker and
            the tokenizer and LM text files exist.

        Examples:
            >>> builder = VoicesBuilder()
            >>> ready = builder.is_built(recipe_dir=".")
        """
        manifest = Path(recipe_dir).resolve() / "data/manifest"
        try:
            metadata = json.loads((manifest / "build.json").read_text())
        except (OSError, ValueError):
            return False
        if not isinstance(metadata, dict):
            return False
        hashes = metadata.get("manifest_sha256")
        if not isinstance(hashes, dict):
            return False
        matches = (
            metadata.get("format_version") == 2
            and metadata.get("corpus") == corpus
            and metadata.get("source_dir")
            == str(resolve_source_root(recipe_dir, source_dir, corpus))
            and metadata.get("builder") == _BUILDER_CFG
            and all((manifest / f"{split}.tsv").is_file() for split in SPLITS)
            and (manifest / "tokenizer_train.txt").is_file()
            and all(
                (manifest.parent / "lm" / f"{split}.txt").is_file() for split in SPLITS
            )
        )
        if not matches:
            return False
        try:
            return all(
                _manifest_digest(manifest / f"{split}.tsv") == hashes.get(split)
                for split in SPLITS
            )
        except OSError:
            return False

    def build(self, recipe_dir, source_dir=None, corpus="devkit", **_kwargs):
        """Write speaker-disjoint manifests and original pre-filtering text.

        Args:
            recipe_dir: Destination recipe root.
            source_dir: Optional existing extracted corpus root.
            corpus: devkit or full. Defaults to devkit for direct API use; the
                original training_conformer config explicitly selects full.
            **_kwargs: Unused stage settings.

        Raises:
            ValueError: Recordings, transcripts, sampling rates or splits are invalid.
            FileNotFoundError: A required recording condition is missing.

        Notes:
            The first ten lexical training speaker IDs form validation. ASR
            train/valid use strict 0.1--30 second bounds. tokenizer_train.txt
            and data/lm retain the source recipe's pre-filtering text. Audio
            remains in place. Rebuild markers include the selected corpus mode.

        Returns:
            None. Completion is recorded by the files written to disk.

        Examples:
            After prepare_source has completed:

            >>> builder = VoicesBuilder()
            >>> builder.build(recipe_dir=".")
        """
        if self.is_built(recipe_dir, source_dir, corpus):
            return
        root = resolve_source_root(recipe_dir, source_dir, corpus)
        # Resolve original transcripts before joining source/distant recordings.
        transcripts = _load_transcripts(root)
        train = _scan_recordings(root, "train", transcripts, corpus)
        test = _scan_recordings(root, "test", transcripts, corpus)
        # Reserve the first ten lexical speakers, keeping conditions together.
        speakers = sorted({row["speaker"] for row in train})
        dev_count = int(_BUILDER_CFG["dev_speakers"])
        if len(speakers) <= dev_count:
            raise ValueError("Not enough training speakers for the validation split")
        if set(speakers) & {row["speaker"] for row in test}:
            raise ValueError("Official training/test speaker sets overlap")
        valid_speakers = set(speakers[:dev_count])
        rate = int(_BUILDER_CFG["sample_rate"])
        minimum = int(_BUILDER_CFG["min_duration"] * rate)
        maximum = int(_BUILDER_CFG["max_duration"] * rate)
        # Apply the original duration bounds to train/valid only.
        selected = [row for row in train if minimum < row["samples"] < maximum]
        splits = {
            "train": [row for row in selected if row["speaker"] not in valid_speakers],
            "valid": [row for row in selected if row["speaker"] in valid_speakers],
            "test": test,
        }
        manifest = Path(recipe_dir).resolve() / "data/manifest"
        manifest.mkdir(parents=True, exist_ok=True)
        (manifest / "build.json").unlink(missing_ok=True)
        # Tokenizer and LM text retain the original pre-filter utterances.
        original_splits = {
            "train": [row for row in train if row["speaker"] not in valid_speakers],
            "valid": [row for row in train if row["speaker"] in valid_speakers],
            "test": test,
        }
        write_text(
            manifest / "tokenizer_train.txt",
            "".join(row["text"] + "\n" for row in original_splits["train"]),
        )
        for split, rows in original_splits.items():
            write_text(
                manifest.parent / "lm" / f"{split}.txt",
                "".join(
                    f"{row['utt_id']} {row['text']}\n"
                    for row in rows
                    if row["text"].strip()
                ),
            )
        # Publish each manifest atomically; fingerprints detect stale outputs.
        hashes = {}
        for split, rows in splits.items():
            if not rows or len({row["utt_id"] for row in rows}) != len(rows):
                raise ValueError(f"Empty or duplicate utterances in {split}")
            path = manifest / f"{split}.tsv"
            temporary = path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream, FIELDS, delimiter="\t", lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(path)
            hashes[split] = _manifest_digest(path)
        # Write the completion marker only after all outputs succeeded.
        write_text(
            manifest / "build.json",
            json.dumps(
                dict(
                    source_dir=str(root),
                    format_version=2,
                    corpus=corpus,
                    builder=_BUILDER_CFG,
                    validation_speakers=sorted(valid_speakers),
                    counts={split: len(rows) for split, rows in splits.items()},
                    filtered_train_valid=len(train) - len(selected),
                    manifest_sha256=hashes,
                ),
                indent=2,
            )
            + "\n",
        )
