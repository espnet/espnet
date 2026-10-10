"""LibriSpeech dataset builder: raw corpus -> Omni-IO archives.

LibriSpeech ships one `.flac` per utterance, so the 960 h training set is
281k files. Reading those directly from a shared cluster filesystem is what
makes a recipe IO-bound, so `create_dataset` packs each logical split into an
Omni-IO archive -- a handful of `.bin` blobs plus a Parquet offset table --
and every later stage reads items from it with a single range read.
"""

from __future__ import annotations

import logging
import os
from importlib import resources
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url, extract_targz

logger = logging.getLogger(__name__)

# One corpus entry before packing: utterance id, audio path, transcript.
RawEntry = Tuple[str, Path, str]

TEXT_TABLE_NAME = "text.tsv"
METADATA_NAME = "metadata.parquet"

_OMNIIO_HINT = (
    "Omni-IO is required by this recipe but is not installed. "
    "Install it with `pip install omniio` "
    "(it is not among ESPnet's default dependencies)."
)


def _load_builder_config() -> dict:
    """Return the ``builder`` section of ``dataset/config.yaml``."""
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_SPLITS = {str(name): [str(d) for d in dirs] for name, dirs in _CFG["splits"].items()}
_ARCHIVES = {str(name): str(archive) for name, archive in _CFG["archives"].items()}


def known_splits() -> List[str]:
    """Return every logical split this recipe can build."""
    return sorted(_SPLITS)


def resolve_split_names(splits: Sequence[str] | str | None) -> List[str]:
    """Normalize a requested split selection into known split names.

    Args:
        splits: One split name, a sequence of split names, or ``None`` to use
            ``builder.required_splits`` from ``dataset/config.yaml``.

    Returns:
        Logical split names in the requested order.

    Raises:
        ValueError: If a requested split is not defined in ``builder.splits``.
    """
    if splits is None:
        requested = [str(split) for split in _CFG["required_splits"]]
    elif isinstance(splits, str):
        requested = [splits]
    else:
        requested = [str(split) for split in splits]

    unknown = [split for split in requested if split not in _SPLITS]
    if unknown:
        raise ValueError(
            "Unknown split(s): "
            + ", ".join(unknown)
            + ". Expected one of: "
            + ", ".join(known_splits())
        )
    return requested


def raw_dirs_for(splits: Sequence[str]) -> List[str]:
    """Return the raw LibriSpeech directories required by ``splits``."""
    ordered: List[str] = []
    for split in splits:
        for raw_dir in _SPLITS[str(split)]:
            if raw_dir not in ordered:
                ordered.append(raw_dir)
    return ordered


def resolve_librispeech_root(data_dir: str | Path) -> Path:
    """Resolve a path to the on-disk ``LibriSpeech`` root.

    Args:
        data_dir: Either a directory containing ``LibriSpeech/`` or the
            ``LibriSpeech`` directory itself.

    Returns:
        The resolved ``LibriSpeech`` root.

    Raises:
        FileNotFoundError: If neither layout is present.
    """
    candidate = Path(data_dir)
    if (candidate / "LibriSpeech").is_dir():
        return candidate / "LibriSpeech"
    if candidate.name == "LibriSpeech" and candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        "Could not find LibriSpeech root. Expected either:\n"
        f"  - {candidate}/LibriSpeech/\n"
        f"  - {candidate} (when it is the LibriSpeech directory itself)"
    )


def iter_source_candidates(
    recipe_root: Path,
    source_dir: str | Path | None,
) -> Iterator[Path]:
    """Yield candidate directories that may contain LibriSpeech.

    An explicit ``source_dir`` and the ``LIBRISPEECH`` environment variable
    come before the recipe-local download directory, so a shared copy of the
    corpus wins over a per-user one.
    """
    if source_dir is not None:
        yield Path(source_dir)

    env_path = os.environ.get(str(_CFG["source_env_var"]))
    if env_path:
        yield Path(env_path)

    yield recipe_root / str(_CFG["dataset_path"])


def resolve_source_root(
    recipe_root: Path,
    source_dir: str | Path | None = None,
) -> Path:
    """Resolve the usable LibriSpeech source root for this recipe.

    Raises:
        FileNotFoundError: If no candidate directory holds a LibriSpeech root.
    """
    checked: List[str] = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        checked.append(str(candidate))
        try:
            return resolve_librispeech_root(candidate)
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        "LibriSpeech source not found. Checked these locations:\n"
        + "\n".join(f"  - {path}" for path in checked)
        + "\n"
        + f"Place the corpus under <recipe_dir>/{_CFG['dataset_path']}/LibriSpeech, "
        + f"set {_CFG['source_env_var']} to the dataset root, or run "
        + "create_dataset with `create_dataset.download: true`."
    )


def resolve_archive_root(
    recipe_root: Path,
    archive_root: str | Path | None = None,
) -> Path:
    """Return the directory that holds this recipe's Omni-IO archives.

    Args:
        recipe_root: Recipe root directory.
        archive_root: Optional override, normally ``${data_dir}`` so the
            archives follow the experiment onto fast cluster storage.
    """
    root = Path(archive_root) if archive_root is not None else recipe_root / "data"
    return root / str(_CFG["archive_subdir"])


def split_archive_dir(
    recipe_root: Path,
    split: str,
    archive_root: str | Path | None = None,
) -> Path:
    """Return the archive directory of one logical split."""
    return resolve_archive_root(recipe_root, archive_root) / str(split)


def blob_path(archive_dir: Path, bin_index: int) -> Path:
    """Return the blob file holding ``bin_index`` inside ``archive_dir``.

    The Parquet table also carries a ``path`` column, but it stores the
    absolute path from packing time. Rebuilding the name from ``bin_index``
    keeps an archive readable after it is copied to another machine.
    """
    return archive_dir / f"blob_{int(bin_index)}.bin"


def text_table_path(archive_dir: Path) -> Path:
    """Return the transcript table path of one archive directory."""
    return archive_dir / TEXT_TABLE_NAME


def read_text_table(archive_dir: Path) -> dict[str, str]:
    """Read ``text.tsv`` as an ``utt_id -> transcript`` mapping.

    Raises:
        FileNotFoundError: If the table does not exist.
        RuntimeError: If the table is empty.
    """
    path = text_table_path(archive_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Transcript table not found: {path}")

    texts: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            utt_id, text = line.split("\t", maxsplit=1)
            texts[utt_id] = text
    if not texts:
        raise RuntimeError(f"Transcript table is empty: {path}")
    return texts


def _scan_raw_dir(split_dir: Path) -> List[RawEntry]:
    """Index one raw LibriSpeech directory by reading its transcript files.

    Returns:
        ``(utt_id, audio_path, text)`` sorted by utterance id.

    Raises:
        RuntimeError: If the directory yields no transcript/audio pairs.
    """
    entries: List[RawEntry] = []
    missing_audio = 0

    for root, _dirs, files in os.walk(split_dir):
        root_path = Path(root)
        for file_name in files:
            if not file_name.endswith(".trans.txt"):
                continue
            with (root_path / file_name).open("r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line:
                        continue
                    utt_id, *words = line.split()
                    if not words:
                        continue
                    audio_path = root_path / f"{utt_id}.flac"
                    if not audio_path.is_file():
                        missing_audio += 1
                        continue
                    entries.append((utt_id, audio_path, " ".join(words)))

    if missing_audio:
        logger.warning(
            "Skipped %d transcript line(s) without audio under %s",
            missing_audio,
            split_dir,
        )
    if not entries:
        raise RuntimeError(
            f"No transcript/audio pairs found under: {split_dir}. "
            "Check that the split is extracted and the path is correct."
        )
    return sorted(entries, key=lambda entry: entry[0])


def _write_text_table(archive_dir: Path, entries: Sequence[RawEntry]) -> None:
    """Write ``text.tsv`` atomically next to the blobs."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    path = text_table_path(archive_dir)
    tmp_path = path.with_suffix(f".tsv.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as fh:
        for utt_id, _audio_path, text in entries:
            fh.write(f"{utt_id}\t{text}\n")
    os.replace(tmp_path, path)


def _open_blob(archive_dir: Path, max_bin_size: int):
    """Return an Omni-IO ``Blob`` writer for ``archive_dir``."""
    try:
        from omniio.blob.blob import Blob
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(_OMNIIO_HINT) from exc

    archive_dir.mkdir(parents=True, exist_ok=True)
    return Blob(
        archive_dir=str(archive_dir),
        modality="audio",
        max_bin_size=int(max_bin_size),
    )


class LibriSpeechBuilder(DatasetBuilder):
    """Stage LibriSpeech and pack it into one Omni-IO archive per split.

    Keyword arguments forwarded from ``training_config.create_dataset``:

    | kwarg            | meaning                                            |
    |---               |---                                                 |
    | ``recipe_dir``   | recipe root (required)                             |
    | ``source_dir``   | LibriSpeech parent/root override                   |
    | ``archive_root`` | directory receiving ``archive/<split>/``           |
    | ``splits``       | logical splits to build                            |
    | ``download``     | fetch missing archives from OpenSLR                |
    | ``num_workers``  | workers used while packing                         |
    | ``max_bin_size`` | bytes per blob file                                |

    Each step is idempotent, so a resubmitted ``create_dataset`` only does the
    work that is still missing.
    """

    def _resolve(self, recipe_dir: str | Path, splits: Sequence[str] | str | None):
        return Path(recipe_dir).resolve(), resolve_split_names(splits)

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        splits: Sequence[str] | str | None = None,
        **_kwargs,
    ) -> bool:
        """Return whether every raw directory needed by ``splits`` exists."""
        recipe_root, split_names = self._resolve(recipe_dir, splits)
        try:
            source_root = resolve_source_root(recipe_root, source_dir=source_dir)
        except FileNotFoundError:
            return False
        return all(
            (source_root / raw_dir).is_dir() for raw_dir in raw_dirs_for(split_names)
        )

    def prepare_source(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        splits: Sequence[str] | str | None = None,
        download: bool | None = None,
        **_kwargs,
    ) -> None:
        """Make sure the raw LibriSpeech directories for ``splits`` exist.

        Raises:
            FileNotFoundError: If a required directory is missing and
                downloading is disabled.
        """
        recipe_root, split_names = self._resolve(recipe_dir, splits)
        allow_download = bool(_CFG["download"]) if download is None else bool(download)
        required = raw_dirs_for(split_names)

        try:
            source_root = resolve_source_root(recipe_root, source_dir=source_dir)
        except FileNotFoundError:
            if not allow_download:
                raise
            source_root = recipe_root / str(_CFG["dataset_path"]) / "LibriSpeech"
            source_root.mkdir(parents=True, exist_ok=True)

        missing = [
            raw_dir for raw_dir in required if not (source_root / raw_dir).is_dir()
        ]
        if not missing:
            return
        if not allow_download:
            raise FileNotFoundError(
                f"LibriSpeech source is incomplete under {source_root}. "
                "Missing directories: "
                + ", ".join(missing)
                + ". Stage them there, point "
                + str(_CFG["source_env_var"])
                + " at a complete copy, or run create_dataset with "
                + "`create_dataset.download: true`."
            )

        # The OpenSLR archives expand to `<parent>/LibriSpeech/<split>`, so
        # they are extracted into the parent of the resolved root.
        extract_root = source_root.parent
        archive_dir = extract_root / "archives"
        base_url = str(_CFG["download_base_url"]).rstrip("/")
        for raw_dir in missing:
            archive_name = _ARCHIVES[raw_dir]
            archive_path = archive_dir / archive_name
            if not archive_path.is_file():
                logger.info("Downloading %s from OpenSLR", archive_name)
                download_url(f"{base_url}/{archive_name}", archive_path, logger=logger)
            logger.info("Extracting %s into %s", archive_name, extract_root)
            extract_targz(archive_path, extract_root, logger=logger)
            if not (source_root / raw_dir).is_dir():
                raise FileNotFoundError(
                    f"{archive_name} did not produce {source_root / raw_dir}"
                )

    def is_built(
        self,
        recipe_dir: str | Path,
        splits: Sequence[str] | str | None = None,
        archive_root: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Return whether every requested split is already packed."""
        recipe_root, split_names = self._resolve(recipe_dir, splits)
        for split in split_names:
            archive_dir = split_archive_dir(recipe_root, split, archive_root)
            if not (archive_dir / METADATA_NAME).is_file():
                return False
            if not text_table_path(archive_dir).is_file():
                return False
        return True

    def build(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        splits: Sequence[str] | str | None = None,
        archive_root: str | Path | None = None,
        num_workers: int | None = None,
        max_bin_size: int | None = None,
        **_kwargs,
    ) -> None:
        """Pack each requested split into ``archive/<split>/``.

        Raises:
            FileNotFoundError: If a required raw directory is missing.
            ImportError: If Omni-IO is not installed.
            RuntimeError: If a raw directory yields no utterances.
        """
        recipe_root, split_names = self._resolve(recipe_dir, splits)
        workers = int(_CFG["num_workers"] if num_workers is None else num_workers)
        bin_size = int(_CFG["max_bin_size"] if max_bin_size is None else max_bin_size)

        pending = [
            split
            for split in split_names
            if not self.is_built(
                recipe_dir=recipe_root, splits=[split], archive_root=archive_root
            )
        ]
        if not pending:
            logger.info("Every requested split is already packed; nothing to do.")
            return

        source_root = resolve_source_root(recipe_root, source_dir=source_dir)
        logger.info("Packing Omni-IO archives from %s", source_root)

        raw_entries: dict[str, List[RawEntry]] = {}
        for raw_dir in raw_dirs_for(pending):
            split_dir = source_root / raw_dir
            if not split_dir.is_dir():
                raise FileNotFoundError(f"Raw split directory not found: {split_dir}")
            logger.info("Indexing %s", split_dir)
            raw_entries[raw_dir] = _scan_raw_dir(split_dir)
            logger.info(
                "Indexed %d utterances in %s", len(raw_entries[raw_dir]), raw_dir
            )

        for split in pending:
            entries: List[RawEntry] = []
            for raw_dir in _SPLITS[split]:
                entries.extend(raw_entries[raw_dir])

            archive_dir = split_archive_dir(recipe_root, split, archive_root)
            logger.info(
                "Packing %s: %d utterances -> %s", split, len(entries), archive_dir
            )
            blob = _open_blob(archive_dir, bin_size)
            blob.append(
                [str(audio_path) for _utt_id, audio_path, _text in entries],
                ids=[utt_id for utt_id, _audio_path, _text in entries],
                num_workers=workers,
                progress=False,
            )
            # Written last: `is_built` treats the transcript table as the marker
            # that a split finished packing.
            _write_text_table(archive_dir, entries)
            logger.info("Packed %s (%d utterances)", split, len(entries))
