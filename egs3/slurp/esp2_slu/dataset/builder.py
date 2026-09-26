"""SLURP corpus validation and manifest building for the ESPnet3 SLU recipe."""

from __future__ import annotations

import json
import os
import re
from importlib import resources
from pathlib import Path
from typing import Iterator, Sequence

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

_MULTI_SPACE_PATTERN = re.compile(" +")

#: Column order of the TSV manifests written by :meth:`SlurpBuilder.build`.
MANIFEST_COLUMNS = ("utt_id", "wav_path", "intent", "transcript")


def _load_builder_config() -> dict:
    """Read the ``builder`` block of the recipe-local ``dataset/config.yaml``."""
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_BUILDER_CONFIG = _load_builder_config()


def normalize_transcript(sentence: str, lowercase: bool = False) -> str:
    """Normalize one SLURP sentence into the text the model is trained on.

    The rules match ``egs2/slurp/asr1/local/prepare_slurp_data.py`` so that both
    recipes learn from identical text: ``@`` and ``#`` are spelled out, commas
    and periods are dropped, runs of spaces collapse, and ``<unk>`` becomes a
    literal word.

    Args:
        sentence: Raw ``sentence`` field of a SLURP jsonl entry.
        lowercase: Whether to lowercase the result. SLURP's synthetic training
            split is lowercased; the real splits are not.

    Returns:
        The normalized transcript, with no leading or trailing whitespace.

    Examples:
        >>> normalize_transcript("Mail  @ home, now.")
        'mail at home now'
        >>> normalize_transcript("MUTE it", lowercase=True)
        'mute it'
    """
    text = sentence.replace("@", " at ").replace("#", " hashtag ")
    text = text.replace(",", "").replace(".", "")
    text = _MULTI_SPACE_PATTERN.sub(" ", text).strip()
    if lowercase:
        text = text.lower()
    return text.replace("<unk>", "unknown")


def resolve_slurp_root(data_dir: str | Path) -> Path:
    """Resolve one candidate directory to an on-disk SLURP root.

    Args:
        data_dir: Directory that either is the SLURP root itself or holds it as
            a ``slurp`` subdirectory (the layout produced by cloning the
            upstream repository into a download directory).

    Returns:
        The directory holding ``dataset/slurp/metadata.json``.

    Raises:
        FileNotFoundError: If neither candidate layout contains the metadata
            file. Callers that want a soft check should use
            :meth:`SlurpBuilder.is_source_prepared` instead.
    """
    candidate = Path(data_dir)
    metadata_path = str(_BUILDER_CONFIG["metadata_path"])
    for root in (candidate, candidate / "slurp"):
        if (root / metadata_path).is_file():
            return root
    raise FileNotFoundError(
        "Could not find a SLURP root. Expected either:\n"
        f"  - {candidate}/{metadata_path}\n"
        f"  - {candidate}/slurp/{metadata_path}"
    )


def iter_source_candidates(
    recipe_root: Path,
    source_dir: str | Path | None,
) -> Iterator[Path]:
    """Yield the directories searched for the corpus, in priority order.

    The recipe-local download directory wins, then an explicit ``source_dir``
    from ``training_config.create_dataset``, then the ``SLURP`` environment
    variable (the same variable ``egs2/TEMPLATE/asr1/db.sh`` uses).
    """
    yield recipe_root / str(_BUILDER_CONFIG["dataset_path"])

    if source_dir is not None:
        yield Path(source_dir)

    env_path = os.environ.get(str(_BUILDER_CONFIG["source_env_var"]))
    if env_path:
        yield Path(env_path)


def resolve_source_root(
    recipe_root: Path,
    source_dir: str | Path | None = None,
) -> Path:
    """Resolve the SLURP root this recipe should read from.

    Args:
        recipe_root: Recipe directory, used to find ``download/slurp``.
        source_dir: Optional override forwarded from
            ``training_config.create_dataset.source_dir``.

    Returns:
        The resolved SLURP root directory.

    Raises:
        FileNotFoundError: If no candidate holds the corpus. The message lists
            every path that was checked and how to point the recipe at one.
    """
    checked: list[str] = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        checked.append(str(candidate))
        try:
            return resolve_slurp_root(candidate)
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        "SLURP source not found. Checked these locations:\n"
        + "\n".join(f"  - {path}" for path in checked)
        + "\n"
        + f"Place the corpus under <recipe_dir>/{_BUILDER_CONFIG['dataset_path']}/slurp"
        + f" or set {_BUILDER_CONFIG['source_env_var']} to the corpus root."
        + " See readme.md for how to obtain SLURP."
    )


def get_manifest_path(recipe_dir: str | Path, split: str) -> Path:
    """Return the manifest path of one split under the recipe data directory.

    Raises:
        ValueError: If ``split`` is not one of the splits in
            ``dataset/config.yaml``.
    """
    manifest_paths = _BUILDER_CONFIG["manifest_paths"]
    if split not in manifest_paths:
        known = ", ".join(sorted(manifest_paths))
        raise ValueError(f"Unknown split '{split}'. Expected one of: {known}")
    return (
        Path(recipe_dir)
        / str(_BUILDER_CONFIG["data_path"])
        / str(manifest_paths[split])
    )


def get_intents_path(recipe_dir: str | Path) -> Path:
    """Return the path of the intent label list written next to the manifests."""
    return (
        Path(recipe_dir)
        / str(_BUILDER_CONFIG["data_path"])
        / str(_BUILDER_CONFIG["intent_path"])
    )


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read a manifest TSV written by :meth:`SlurpBuilder.build`.

    Args:
        path: Manifest file path, normally from :func:`get_manifest_path`.

    Returns:
        One dict per row keyed by :data:`MANIFEST_COLUMNS`.

    Raises:
        FileNotFoundError: If the manifest has not been built yet.
        ValueError: If a row does not have exactly four tab-separated columns,
            which means the manifest was written by different code.
        RuntimeError: If the manifest is empty.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows: list[dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        for line in manifest_file:
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) != len(MANIFEST_COLUMNS):
                raise ValueError(f"Invalid manifest line in {manifest_path}: {line!r}")
            rows.append(dict(zip(MANIFEST_COLUMNS, fields)))

    if not rows:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")
    return rows


def ensure_built(
    recipe_dir: str | Path,
    source_dir: str | Path | None = None,
) -> Path:
    """Build the manifests if they are missing, and return the recipe root.

    This lets :class:`~egs3.slurp.esp2_slu.dataset.dataset.SlurpDataset` and the
    tokenizer text hook work even when ``create_dataset`` was never run
    explicitly. Once the manifests exist the corpus is never looked up again:
    manifests store absolute audio paths, so training does not depend on where
    the corpus was found at preparation time.
    """
    recipe_root = Path(recipe_dir).resolve()
    builder = SlurpBuilder()
    if builder.is_built(recipe_dir=recipe_root, source_dir=source_dir):
        return recipe_root
    if not builder.is_source_prepared(recipe_dir=recipe_root, source_dir=source_dir):
        builder.prepare_source(recipe_dir=recipe_root, source_dir=source_dir)
    builder.build(recipe_dir=recipe_root, source_dir=source_dir)
    return recipe_root


def _find_missing_source_files(source_root: Path) -> list[str]:
    """List corpus-relative paths that the builder needs but cannot find."""
    missing: list[str] = []
    if not (source_root / str(_BUILDER_CONFIG["metadata_path"])).is_file():
        missing.append(str(_BUILDER_CONFIG["metadata_path"]))
    for split_config in _BUILDER_CONFIG["splits"].values():
        if not (source_root / str(split_config["jsonl_path"])).is_file():
            missing.append(str(split_config["jsonl_path"]))
    for audio_dir in _BUILDER_CONFIG["audio_paths"].values():
        if not (source_root / str(audio_dir)).is_dir():
            missing.append(str(audio_dir))
    return missing


def _load_speaker_ids(source_root: Path) -> dict[str, str]:
    """Map each SLURP recording id to the user id that recorded it."""
    metadata_path = source_root / str(_BUILDER_CONFIG["metadata_path"])
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        records = json.load(metadata_file)
    speakers: dict[str, str] = {}
    for record in records.values():
        for file_name, recording in record["recordings"].items():
            speakers[file_name[6:-5]] = recording["usrid"]
    return speakers


def _iter_split_rows(
    source_root: Path,
    split_config: dict,
    speakers: dict[str, str],
    seen_recordings: set[str],
) -> Iterator[tuple[str, str, str, str]]:
    """Yield one manifest row per recording of one split.

    ``seen_recordings`` is shared across splits and mutated here: SLURP repeats
    a few recordings across prompts, and the first occurrence wins, as in the
    egs2 recipe.
    """
    audio_dir = source_root / str(
        _BUILDER_CONFIG["audio_paths"][str(split_config["audio"])]
    )
    is_synthetic = str(split_config["audio"]) == "synth"
    lowercase = bool(split_config.get("lowercase", False))
    jsonl_path = source_root / str(split_config["jsonl_path"])

    with jsonl_path.open("r", encoding="utf-8") as prompt_file:
        for line in prompt_file:
            line = line.strip()
            if not line:
                continue
            prompt = json.loads(line)
            intent = f"{prompt['scenario']}_{prompt['action']}"
            transcript = normalize_transcript(prompt["sentence"], lowercase=lowercase)
            for recording in prompt["recordings"]:
                recording_id = recording["file"][6:-5]
                if recording_id in seen_recordings:
                    continue
                seen_recordings.add(recording_id)
                speaker = "synthetic" if is_synthetic else speakers[recording_id]
                yield (
                    f"slurp_{speaker}_{recording_id}",
                    str(audio_dir / recording["file"]),
                    intent,
                    transcript,
                )


def _write_rows(path: Path, rows: Sequence[tuple[str, str, str, str]]) -> None:
    """Write manifest rows atomically.

    The rows go to a sibling temporary file that is renamed into place, so an
    interrupted ``create_dataset`` never leaves a half-written manifest that a
    later run would treat as complete.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as manifest_file:
        for row in rows:
            manifest_file.write("\t".join(row) + "\n")
    temporary_path.replace(path)


class SlurpBuilder(DatasetBuilder):
    """Validate the SLURP corpus and derive the recipe's TSV manifests.

    This is the manifest-building flavour of :class:`DatasetBuilder`: the corpus
    itself must already be on disk (``prepare_source`` only verifies it, since
    SLURP's audio is distributed separately from its metadata repository), while
    ``build`` turns the jsonl prompt files into one flat manifest per split plus
    the intent label list the tokenizer reserves.

    Stages: this class is driven by ``create_dataset``; it is resolved through
    the recipe-local ``dataset/__init__.py`` because the configs use a bare
    ``data_src_args`` entry with no ``data_src``.

    Examples:
        >>> builder = SlurpBuilder()
        >>> builder.is_source_prepared(recipe_dir=".")
        True
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **kwargs,
    ) -> bool:
        """Return whether the corpus is present and holds every needed file."""
        recipe_root = Path(recipe_dir).resolve()
        try:
            source_root = resolve_source_root(recipe_root, source_dir=source_dir)
        except FileNotFoundError:
            return False
        return not _find_missing_source_files(source_root)

    def prepare_source(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **kwargs,
    ) -> None:
        """Verify the corpus, which this recipe never downloads on its own.

        SLURP ships its metadata as a git repository and its audio as separate
        archives fetched by a script inside that repository, so there is no
        single URL to hand to ``espnet3.utils.download_utils``. ``readme.md``
        documents the two commands.

        Args:
            recipe_dir: Recipe directory.
            source_dir: Optional corpus root override.
            **kwargs: Unused; accepted because ``create_dataset`` forwards the
                whole ``training_config.create_dataset`` block.

        Raises:
            FileNotFoundError: If the corpus root is missing, or is present but
                incomplete. The message names the missing paths.
        """
        recipe_root = Path(recipe_dir).resolve()
        source_root = resolve_source_root(recipe_root, source_dir=source_dir)
        missing = _find_missing_source_files(source_root)
        if missing:
            raise FileNotFoundError(
                f"SLURP source at {source_root} is incomplete. Missing: "
                + ", ".join(missing)
                + "\nSee readme.md for how to obtain the metadata and the audio."
            )

    def is_built(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **kwargs,
    ) -> bool:
        """Return whether every manifest and the intent list already exist."""
        recipe_root = Path(recipe_dir).resolve()
        if not get_intents_path(recipe_root).is_file():
            return False
        return all(
            get_manifest_path(recipe_root, split).is_file()
            for split in _BUILDER_CONFIG["splits"]
        )

    def build(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **kwargs,
    ) -> None:
        """Write one TSV manifest per split plus the intent label list.

        Outputs land under ``<recipe_dir>/data/manifest/``. Re-running is safe:
        every file is rewritten from the corpus through a temporary file, so the
        stage is idempotent and never leaves partial output behind.

        Only the training splits contribute intent labels, mirroring
        ``cut -d' ' -f2 data/train/text`` in ``egs2/slurp/asr1/local/run_spm.sh``.

        Args:
            recipe_dir: Recipe directory.
            source_dir: Optional corpus root override.
            **kwargs: Unused; see :meth:`prepare_source`.

        Raises:
            FileNotFoundError: If the corpus is missing or incomplete.
            RuntimeError: If a split yields no utterance, which means the jsonl
                file was empty or in an unexpected format.
        """
        recipe_root = Path(recipe_dir).resolve()
        self.prepare_source(recipe_dir=recipe_root, source_dir=source_dir)
        source_root = resolve_source_root(recipe_root, source_dir=source_dir)

        speakers = _load_speaker_ids(source_root)
        seen_recordings: set[str] = set()
        intents: set[str] = set()

        for split, split_config in _BUILDER_CONFIG["splits"].items():
            rows = list(
                _iter_split_rows(
                    source_root, dict(split_config), speakers, seen_recordings
                )
            )
            if not rows:
                raise RuntimeError(f"No utterance found for split '{split}'.")
            if bool(split_config.get("intent_source", False)):
                intents.update(row[2] for row in rows)
            _write_rows(get_manifest_path(recipe_root, split), rows)

        intents_path = get_intents_path(recipe_root)
        intents_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = intents_path.with_name(f".{intents_path.name}.tmp")
        temporary_path.write_text("\n".join(sorted(intents)) + "\n", encoding="utf-8")
        temporary_path.replace(intents_path)
