"""Manifests for WHAM! single-speaker enhancement (16 kHz, min)."""

from __future__ import annotations

import hashlib
import logging
import os
from importlib import resources
from pathlib import Path
from typing import Optional

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.systems.snoar.src.data.manifest import describe_recordings, write_manifest
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_CFG = _CONFIG["builder"]
MANIFESTS = ("train", "valid", "test", "noise_wham")


def resolve_source_root() -> Optional[Path]:
    """Return the WHAM! 16 kHz ``min`` directory named by ``WHAM_ROOT``."""
    root = os.environ.get(_CFG["source_env_var"])
    return Path(root) if root else None


def resolve_output_root(recipe_dir: Optional[str | Path] = None) -> Path:
    """Return where the manifests are written."""
    out = os.environ.get(_CFG["output_env_var"])
    if out:
        return Path(out)
    base = Path(recipe_dir) if recipe_dir else Path(__file__).resolve().parents[1]
    return base / _CFG["output_path"]


def is_x1_utterance(utt: str, fraction: float, salt: str) -> bool:
    """Return whether ``utt`` belongs to the x1 half (stable across runs).

    Examples:
        >>> is_x1_utterance("u1", 1.0, "s")
        True
    """
    h = hashlib.sha1(f"{salt}:{utt}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF < fraction


class WHAMBuilder(DatasetBuilder):
    """Write ``train``/``valid``/``test`` recording manifests and the noise pool.

    ``train`` holds the x1 half of ``tr`` (no references: no mode reads them
    for training); ``valid`` and ``test`` hold ``cv``/``tt`` with
    ``speech_ref``; ``noise_wham`` lists the noise files of the other half.
    """

    def is_source_prepared(self, **kwargs) -> bool:
        """Return whether ``WHAM_ROOT`` holds the noisy ``tr`` directory."""
        root = resolve_source_root()
        return bool(root) and (root / "tr" / _CFG["noisy_subdir"]).is_dir()

    def prepare_source(self, **kwargs) -> None:
        """Explain how to obtain WHAM! (it needs the LDC WSJ0 corpus)."""
        raise FileNotFoundError(
            "WHAM! is not found. Generate the 16 kHz `min` WHAM!/WHAMR! files "
            "(https://wham.whisper.ai, needs LDC WSJ0) and set WHAM_ROOT to the "
            "directory holding tr/ cv/ tt/, e.g. <WHAMR>/wav16k/min."
        )

    def is_built(self, recipe_dir=None, **kwargs) -> bool:
        """Return whether every manifest exists."""
        out = resolve_output_root(recipe_dir) / "manifest"
        return all((out / f"{name}.jsonl").is_file() for name in MANIFESTS)

    def build(self, recipe_dir=None, **kwargs) -> None:
        """Write the manifests."""
        root = resolve_source_root()
        out = resolve_output_root(recipe_dir) / "manifest"
        noisy, speech, noise = (
            _CFG["noisy_subdir"],
            _CFG["speech_subdir"],
            _CFG["noise_subdir"],
        )
        frac, salt = float(_CFG["x1_fraction"]), str(_CFG["salt"])
        for name, split in _CFG["splits"].items():
            files = sorted((root / split / noisy).glob("*.wav"))
            if not files:
                raise FileNotFoundError(f"no wav files in {root / split / noisy}")
            if name == "train":
                x1 = [f for f in files if is_x1_utterance(f.stem, frac, salt)]
                donors = [f for f in files if not is_x1_utterance(f.stem, frac, salt)]
                rows = describe_recordings(x1)
                pool = [
                    {
                        "audio": str((root / split / noise / f.name).resolve()),
                        "start": 0,
                        "end": None,
                        "group": f.stem,
                    }
                    for f in donors
                ]
                write_manifest(pool, out / "noise_wham.jsonl")
                logger.info("noise pool: %d WHAM! noise files", len(pool))
            else:
                rows = describe_recordings(files)
                for r in rows:
                    r["speech_ref"] = str(
                        (root / split / speech / f"{r['id']}.wav").resolve()
                    )
            n = write_manifest(rows, out / f"{name}.jsonl")
            logger.info("%s: %d recordings -> %s", name, n, out / f"{name}.jsonl")
