"""TIMIT dataset builder for phone recognition."""

from __future__ import annotations

import logging
import os
from importlib import resources
from pathlib import Path
from typing import Iterator, List

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_DROP_PHONES = {str(phone) for phone in _CFG["drop_phones"]}
_PHONE_TO_IPA = {str(k): str(v) for k, v in _CFG["phone_to_ipa"].items()}


def _iter_source_candidates(
    recipe_root: Path,
    source_dir: str | Path | None = None,
) -> Iterator[Path]:
    """Yield candidate TIMIT source roots in priority order."""
    if source_dir is not None:
        yield Path(source_dir).expanduser()

    env_path = os.environ.get(str(_CFG["source_env_var"]))
    if env_path:
        yield Path(env_path).expanduser()

    yield recipe_root / _CFG["dataset_path"]


def _missing_source_entries(source_root: Path) -> List[str]:
    """Return required source paths that are absent from ``source_root``."""
    missing: List[str] = []
    for split in _CFG["splits"]:
        split_dir = source_root / str(split)
        if not split_dir.is_dir():
            missing.append(str(split_dir))
    return missing


def _resolve_source_root(
    recipe_root: Path,
    source_dir: str | Path | None = None,
) -> Path:
    """Resolve the usable TIMIT source root for this recipe.

    Args:
        recipe_root: Recipe root directory.
        source_dir: Optional explicit source root that wins over the
            environment variable and the in-recipe download directory.

    Returns:
        Path to a directory holding the ``TRAIN`` and ``TEST`` trees.

    Raises:
        FileNotFoundError: If no candidate directory holds a complete corpus.
    """
    checked: List[str] = []
    for candidate in _iter_source_candidates(recipe_root, source_dir):
        checked.append(str(candidate))
        if not _missing_source_entries(candidate):
            return candidate

    env_var = str(_CFG["source_env_var"])
    raise FileNotFoundError(
        "TIMIT source not found. Checked these locations:\n"
        + "\n".join(f"  - {path}" for path in checked)
        + "\n"
        + "TIMIT is distributed by the LDC as catalogue LDC93S1 and cannot be "
        + "downloaded automatically; see https://catalog.ldc.upenn.edu/LDC93S1. "
        + f"Place it under <recipe_dir>/{_CFG['dataset_path']} or set {env_var} "
        + "to the directory that holds TRAIN/ and TEST/."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Resolve where the manifest is written."""
    env_var = _CFG.get("output_env_var")
    if env_var:
        env_path = os.environ.get(str(env_var))
        if env_path:
            return Path(env_path).expanduser()
    return recipe_root / _CFG["data_path"]


def _read_phones(phn_path: Path) -> List[str]:
    """Read the phone labels of one ``.PHN`` file, discarding the timings.

    Args:
        phn_path: Path to a TIMIT ``.PHN`` file, whose lines are
            ``<start_sample> <end_sample> <phone>``.

    Returns:
        The phone labels, lowercased, in order.
    """
    phones: List[str] = []
    with phn_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            fields = line.split()
            if fields:
                phones.append(fields[-1].lower())
    return phones


def convert_phones_to_ipa(phones: List[str]) -> str:
    """Convert a TIMIT phone sequence into a concatenated IPA transcript.

    Silences and stop closures are dropped, and every remaining symbol is
    mapped through the recipe's table. The result is concatenated without
    separators, which is the form the phone metrics normalize and re-segment.

    Args:
        phones: Lowercased TIMIT phone labels.

    Returns:
        The IPA transcript.

    Raises:
        KeyError: If a phone is neither droppable nor mappable, which means the
            corpus uses a symbol outside the documented TIMIT set.

    Examples:
        >>> convert_phones_to_ipa(["h#", "sh", "iy", "dcl", "d"])
        'ʃid'
    """
    ipa = []
    for phone in phones:
        if phone in _DROP_PHONES:
            continue
        if phone not in _PHONE_TO_IPA:
            raise KeyError(
                f"Unknown TIMIT phone {phone!r}. Add it to phone_to_ipa or "
                "drop_phones in dataset/config.yaml."
            )
        ipa.append(_PHONE_TO_IPA[phone])
    return "".join(ipa)


class TimitBuilder(DatasetBuilder):
    """Prepare and build TIMIT assets for ESPnet3 phone recognition recipes."""

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **kwargs,
    ) -> bool:
        """Check whether a complete TIMIT source tree is reachable."""
        recipe_root = Path(recipe_dir).resolve()
        try:
            _resolve_source_root(recipe_root, source_dir=source_dir)
        except FileNotFoundError:
            return False
        return True

    def prepare_source(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **kwargs,
    ) -> None:
        """Verify that the TIMIT corpus is reachable.

        This never downloads or writes: TIMIT is licensed by the LDC and is
        usually mounted read-only, so the corpus is validated rather than
        produced. A failure here is the message telling the user where to put
        it.

        Args:
            recipe_dir: Recipe root directory.
            source_dir: Optional explicit source root that wins over the
                environment variable and the in-recipe download directory.
            **kwargs: Unused extra options for API compatibility.

        Returns:
            None.

        Raises:
            FileNotFoundError: If no candidate location holds the corpus. The
                message lists every path that was probed.
        """
        recipe_root = Path(recipe_dir).resolve()
        source_root = _resolve_source_root(recipe_root, source_dir=source_dir)
        logger.info("TIMIT source found under %s", source_root)

    def is_built(
        self,
        recipe_dir: str | Path,
        **kwargs,
    ) -> bool:
        """Check whether the manifest already exists."""
        data_root = resolve_data_root(Path(recipe_dir).resolve())
        return (data_root / _CFG["manifest_path"]).is_file()

    def build(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **kwargs,
    ) -> None:
        """Write one TSV manifest of every utterance and its IPA reference.

        Args:
            recipe_dir: Recipe root directory.
            source_dir: Optional explicit TIMIT source root.
            **kwargs: Unused extra options for API compatibility.

        Returns:
            None.

        Raises:
            FileNotFoundError: If the TIMIT source tree cannot be resolved.
            RuntimeError: If a ``.PHN`` file has no matching ``.WAV``, or if no
                utterance is found at all.
            KeyError: If a phone is outside the documented TIMIT set.

        Notes:
            Utterance ids are ``<dialect>-<speaker>-<utterance>`` lowercased,
            matching the IPAPack/POWSM ids for the same corpus. Both the TRAIN
            and TEST trees are emitted as a single evaluation set, because that
            is what the published phone recognition results are measured on.
        """
        recipe_root = Path(recipe_dir).resolve()
        source_root = _resolve_source_root(recipe_root, source_dir=source_dir)
        data_root = resolve_data_root(recipe_root)

        entries: List[tuple] = []
        for split in _CFG["splits"]:
            # Bounded glob rather than rglob: the layout is exactly
            # <split>/<dialect>/<speaker>/<utterance>.PHN, and walking the whole
            # tree instead costs minutes on a networked filesystem.
            for phn_path in sorted((source_root / str(split)).glob("*/*/*.PHN")):
                wav_path = phn_path.with_suffix(".WAV")
                if not wav_path.is_file():
                    raise RuntimeError(
                        f"{phn_path} has no matching audio file at {wav_path}."
                    )
                dialect, speaker = phn_path.parts[-3], phn_path.parts[-2]
                utt_id = f"{dialect}-{speaker}-{phn_path.stem}".lower()
                reference = convert_phones_to_ipa(_read_phones(phn_path))
                entries.append((utt_id, wav_path.resolve(), reference))

        if not entries:
            raise RuntimeError(f"No .PHN file found under {source_root}.")

        manifest = data_root / _CFG["manifest_path"]
        manifest.parent.mkdir(parents=True, exist_ok=True)
        # Write to a part file and rename, so an interrupted build cannot leave
        # a truncated manifest that `is_built` would accept.
        part = manifest.with_suffix(manifest.suffix + ".part")
        with part.open("w", encoding="utf-8") as fh:
            for utt_id, wav_path, reference in sorted(entries, key=lambda e: e[0]):
                fh.write(f"{utt_id}\t{wav_path}\t{reference}\n")
        part.replace(manifest)
        logger.info("wrote %d entries to %s", len(entries), manifest)
