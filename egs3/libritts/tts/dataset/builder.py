"""LibriTTS dataset builder for ESPnet3 TTS recipe."""

from __future__ import annotations

import logging
import subprocess
import urllib.error
from importlib import resources, util
from pathlib import Path

from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.phoneme_tokenizer import PhonemeTokenizer
from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url

logger = logging.getLogger(__name__)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()


def _scan_subset_entries(subset_dir: Path) -> list[tuple[str, Path, str, str]]:
    """
    Scan a subset directory and return a list of
    (utt_id, wav_path, text, spk_key) tuples.

    Args:
        subset_dir: Path to the subset directory (e.g., "LibriTTS/train-clean-100")
    Returns:
        List of tuples containing:
            - utt_id: Unique utterance ID (e.g., "123-456-789")
            - wav_path: Path to the corresponding WAV file
            - text: Transcription text
            - spk_key: Speaker key (e.g., "speaker_chapter") for speaker ID mapping
    """
    entries = []
    for text_path in sorted(subset_dir.rglob("*.normalized.txt")):
        wav_path = text_path.with_suffix("").with_suffix(".wav")
        if not wav_path.is_file():
            continue
        text = text_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        utt_id = text_path.stem.replace(".normalized", "")
        speaker = text_path.parent.parent.name
        spk_key = speaker
        entries.append((utt_id, wav_path.resolve(), text, spk_key))
    return entries


def _libritts_subsets() -> list[str]:
    """Return every LibriTTS subset referenced by ``split_subsets``."""
    subsets: list[str] = []
    for split_subsets in _CFG["split_subsets"].values():
        subsets.extend(split_subsets)
    return subsets


def _librispeech_pc_paths(recipe_root: Path) -> tuple[Path, Path, Path]:
    """Resolve the three LibriSpeech-PC paths from the builder config.

    Args:
        recipe_root: Resolved recipe root directory.

    Returns:
        Tuple of ``(test_clean_root, lst_path, manifest_path)``. The first two
        live under ``builder.dataset_path``, the last under
        ``builder.data_path``.
    """
    cfg = _CFG["librispeech_pc"]
    dataset_root = recipe_root / _CFG["dataset_path"]
    data_dir = recipe_root / _CFG["data_path"]
    return (
        dataset_root / cfg["test_clean_path"],
        dataset_root / cfg["lst_path"],
        data_dir / cfg["manifest_path"],
    )


def _is_libritts_prepared(recipe_root: Path) -> bool:
    """Check whether every required LibriTTS subset is extracted."""
    libritts_root = recipe_root / _CFG["dataset_path"] / "LibriTTS"
    return all((libritts_root / subset).is_dir() for subset in _libritts_subsets())


def _is_librispeech_pc_prepared(recipe_root: Path) -> bool:
    """Check whether the LibriSpeech test-clean tree and pair list are present."""
    test_clean_root, lst_path, _ = _librispeech_pc_paths(recipe_root)
    return test_clean_root.is_dir() and lst_path.is_file()


def _load_build_manifest():
    """Return ``prepare_librispeech_pc.build_manifest``.

    Returns:
        The ``build_manifest`` function from ``local/prepare_librispeech_pc.py``.

    Raises:
        ModuleNotFoundError: If the module cannot be located either way.

    Notes:
        The package import is the documented entry point and the one the
        recipe's tests use. It resolves only when the espnet root is on
        ``sys.path``, which ``path.sh`` arranges; when it is not, load the file
        sitting next to this one instead. The fallback is not a second copy of
        the logic, it is the same file, and it is the more precise of the two:
        ``egs3`` is a namespace package, so with several espnet checkouts
        installed the package import can resolve to a different checkout's
        copy. The path is derived from ``__file__`` rather than from
        ``recipe_dir`` because ``recipe_dir`` points at the working tree the
        manifests are written into, which need not hold the recipe's source.
    """
    try:
        from egs3.libritts.tts.local.prepare_librispeech_pc import build_manifest

        return build_manifest
    except ModuleNotFoundError:
        module_path = (
            Path(__file__).resolve().parents[1] / "local" / "prepare_librispeech_pc.py"
        )
        logger.info(
            f"egs3.libritts.tts.local is not importable; loading {module_path} "
            f"directly. Source path.sh to put the espnet root on PYTHONPATH."
        )
        spec = util.spec_from_file_location(
            "_espnet3_prepare_librispeech_pc", module_path
        )
        if spec is None or spec.loader is None:
            raise ModuleNotFoundError(
                f"Could not load the LibriSpeech-PC manifest builder from "
                f"{module_path}."
            )
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.build_manifest


def _download_lst(url: str, dest: Path) -> None:
    """Download the LibriSpeech-PC pair list to ``dest``.

    Args:
        url: Pinned URL of ``librispeech_pc_test_clean_cross_sentence.lst``.
        dest: Destination path.

    Raises:
        RuntimeError: If the download fails for any reason.

    Notes:
        Downloads to a temporary sibling and renames, so an interrupted
        transfer never leaves a truncated file that later runs would treat as
        complete.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_name(dest.name + ".tmp")
    logger.info(f"Downloading LibriSpeech-PC pair list from {url}")
    try:
        download_url(url, tmp_path, logger=logger)
        tmp_path.replace(dest)
    except (urllib.error.URLError, OSError) as e:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download the LibriSpeech-PC pair list from {url}. "
            f"Check your internet connection, or download the file manually "
            f"and place it at {dest}."
        ) from e


class LibriTTSBuilder(DatasetBuilder):
    """Prepare LibriTTS manifests and token list for ESPnet3 TTS."""

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        **_kwargs,
    ) -> bool:
        """Check if the raw source corpora are prepared.

        Args:
            recipe_dir: Recipe root directory.
            **_kwargs: Unused extra options for API compatibility.
        Returns:
            True if the required LibriTTS subsets, the LibriSpeech test-clean
            tree, and the LibriSpeech-PC pair list are all present; False
            otherwise.
        """

        recipe_root = Path(recipe_dir).resolve()
        return _is_libritts_prepared(recipe_root) and _is_librispeech_pc_prepared(
            recipe_root
        )

    def prepare_source(
        self,
        recipe_dir: str | Path,
        **_kwargs,
    ) -> None:
        """Prepare the raw source corpora by downloading whatever is missing.

        Args:
            recipe_dir: Recipe root directory.
            **_kwargs: Unused extra options for API compatibility.

        Raises:
            RuntimeError: If any download fails.

        Notes:
            Two independent corpora are handled, each gated on its own
            readiness check so an already-extracted LibriTTS tree is never
            re-downloaded just because the LibriSpeech side is missing:

            1. LibriTTS (OpenSLR 60), one download per subset in
               ``split_subsets``.
            2. LibriSpeech test-clean (OpenSLR 12) plus the LibriSpeech-PC
               pair list, the two external inputs of the default eval config.

            All downloads are idempotent: extracted subsets carry a
            ``.complete`` marker and the pair list is skipped when present, so
            re-running ``create_dataset`` transfers nothing.
        """
        recipe_root = Path(recipe_dir).resolve()
        dataset_root = recipe_root / _CFG["dataset_path"]

        if _is_libritts_prepared(recipe_root):
            logger.info("LibriTTS source data is already prepared, skipping download.")
        else:
            dataset_root.mkdir(parents=True, exist_ok=True)
            script_path = recipe_root / "local" / "download_libritts.sh"

            for subset in _libritts_subsets():
                subset_dir = dataset_root / "LibriTTS" / subset
                if (subset_dir / ".complete").is_file() or subset_dir.is_dir():
                    logger.info(f"Subset {subset} already downloaded, skipping.")
                    continue
                logger.info(f"Downloading LibriTTS subset: {subset}")
                try:
                    subprocess.run(
                        ["bash", str(script_path), str(dataset_root), subset],
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(
                        f"Failed to download LibriTTS subset {subset}. "
                        f"Check internet connection and disk space."
                    ) from e

        if _is_librispeech_pc_prepared(recipe_root):
            logger.info(
                "LibriSpeech-PC source data is already prepared, skipping download."
            )
            return

        dataset_root.mkdir(parents=True, exist_ok=True)
        lspc_cfg = _CFG["librispeech_pc"]
        test_clean_root, lst_path, _ = _librispeech_pc_paths(recipe_root)

        if not test_clean_root.is_dir():
            subset = lspc_cfg["subset"]
            script_path = recipe_root / "local" / "download_librispeech.sh"
            logger.info(f"Downloading LibriSpeech subset: {subset}")
            try:
                subprocess.run(
                    ["bash", str(script_path), str(dataset_root), subset],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Failed to download LibriSpeech subset {subset}. "
                    f"Check internet connection and disk space."
                ) from e

        if not lst_path.is_file():
            _download_lst(lspc_cfg["lst_url"], lst_path)

    def is_libritts_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check only the LibriTTS split manifests, which training reads.

        Args:
            recipe_dir: Recipe root directory.
            **_kwargs: Unused extra options for API compatibility.
        Returns:
            True if the LibriTTS split manifests exist; False otherwise.

        Notes:
            Deliberately narrower than :meth:`is_built`. ``LibriTTSDataset``
            guards on this one, so training is not blocked by a missing
            LibriSpeech-PC eval manifest it never reads. Widening this to the
            eval manifest would mean an existing checkout whose LibriTTS
            manifests are already built could no longer start training without
            first downloading LibriSpeech.
        """

        data_dir = Path(recipe_dir).resolve() / _CFG["data_path"]
        return all(
            (data_dir / relpath).is_file()
            for relpath in _CFG["manifest_paths"].values()
        )

    def is_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check if the dataset artifacts (manifests) are built.

        Args:
            recipe_dir: Recipe root directory.
            **_kwargs: Unused extra options for API compatibility.
        Returns:
            True if the LibriTTS split manifests and the LibriSpeech-PC eval
            manifest all exist; False otherwise.

        Notes:
            The LibriSpeech-PC manifest is part of this check because
            ``conf/inference_f5.yaml``, the default eval config, reads it. Were
            it left out, ``create_dataset`` would report success while the
            default eval had nothing to read. Use
            :meth:`is_libritts_built` for the training-only subset.
        """

        recipe_root = Path(recipe_dir).resolve()
        _, _, lspc_manifest = _librispeech_pc_paths(recipe_root)
        return self.is_libritts_built(recipe_dir=recipe_root) and (
            lspc_manifest.is_file()
        )

    def build(
        self,
        recipe_dir: str | Path,
        **_kwargs,
    ) -> None:
        """Build the dataset artifacts (manifests).

        Args:
            recipe_dir: Recipe root directory.
            **_kwargs: Optional keyword arguments for build customization:

        Returns:
            None.

        Raises:
            FileNotFoundError: If a required source tree or the LibriSpeech-PC
                pair list is missing.

        Notes:
            Build flow:

            1. Scan the LibriTTS subsets and write the train/valid/test
               manifests.
            2. Write the LibriSpeech-PC cross-sentence eval manifest read by
               ``conf/inference_f5.yaml``.

            This method performs no network I/O. Everything it reads is
            fetched by ``prepare_source()``.
        """

        recipe_root = Path(recipe_dir).resolve()
        libritts_root = recipe_root / _CFG["dataset_path"] / "LibriTTS"
        data_dir = recipe_root / _CFG["data_path"]
        data_dir.mkdir(parents=True, exist_ok=True)

        split_entries = {}
        speaker_to_id = {}

        for split, subsets in _CFG["split_subsets"].items():
            entries = []
            for subset in subsets:
                subset_dir = libritts_root / subset
                if not subset_dir.is_dir():
                    raise FileNotFoundError(f"Subset directory not found: {subset_dir}")
                entries.extend(_scan_subset_entries(subset_dir))
            entries = sorted(entries, key=lambda x: x[0])
            split_entries[split] = entries
            for _, _, _, spk_key in entries:
                if spk_key not in speaker_to_id:
                    speaker_to_id[spk_key] = len(speaker_to_id)

        for split, entries in split_entries.items():
            manifest_path = data_dir / _CFG["manifest_paths"][split]
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("w", encoding="utf-8") as f:
                for utt_id, wav_path, text, spk_key in entries:
                    sid = speaker_to_id[spk_key]
                    f.write(f"{utt_id}\t{wav_path}\t{text}\t{sid}\n")

        self._build_librispeech_pc_manifest(recipe_root)

    @staticmethod
    def _build_librispeech_pc_manifest(recipe_root: Path) -> None:
        """Write the LibriSpeech-PC cross-sentence eval manifest.

        Args:
            recipe_root: Resolved recipe root directory.

        Raises:
            FileNotFoundError: If the pair list or the LibriSpeech test-clean
                tree is missing.

        Notes:
            Delegates to ``local/prepare_librispeech_pc.py``'s
            ``build_manifest`` so the standalone CLI and this stage cannot
            drift apart. The import is deferred to call time: the dataset
            module is normally loaded from its file path rather than as
            ``egs3.libritts.tts.dataset``, so a module-level import would make
            even the LibriTTS-only path fail whenever the espnet root is not
            on ``sys.path``. See ``_load_build_manifest``.
        """
        build_manifest = _load_build_manifest()

        test_clean_root, lst_path, manifest_path = _librispeech_pc_paths(recipe_root)
        for path, what in (
            (lst_path, "LibriSpeech-PC pair list"),
            (test_clean_root, "LibriSpeech test-clean tree"),
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {what}: {path}. Run the create_dataset stage so "
                    f"prepare_source() downloads it."
                )
        n_rows = build_manifest(lst_path, test_clean_root, manifest_path)
        logger.info(f"Wrote {n_rows} LibriSpeech-PC rows to {manifest_path}")
