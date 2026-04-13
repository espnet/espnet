"""Mini AN4 TTS dataset builder."""

from __future__ import annotations

import subprocess
from importlib import resources
from pathlib import Path

from egs3.mini_an4.asr.dataset.builder import MiniAn4Builder as MiniAn4ASRBuilder
from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()


def _normalize_downloads_layout(dataset_root: Path) -> None:
    """Normalize extracted archive layout to ``downloads/an4``."""
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


class MiniAn4TTSBuilder(DatasetBuilder):
    """Prepare Mini AN4 TTS artifacts from the shared Mini AN4 source corpus."""

    def __init__(self) -> None:
        self._asr_builder = MiniAn4ASRBuilder()

    def _resolve_archive(self, recipe_root: Path) -> Path:
        """Resolve the bundled Mini AN4 archive for the TTS recipe."""
        candidates = [
            recipe_root / "downloads.tar.gz",
            recipe_root.parent / "asr" / "downloads.tar.gz",
            (recipe_root / "../../../egs2/mini_an4/asr1/downloads.tar.gz").resolve(),
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        formatted = "\n".join(f"- {path}" for path in candidates)
        raise FileNotFoundError(
            "Mini AN4 archive not found. Checked:\n"
            f"{formatted}"
        )

    def is_source_prepared(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check whether the shared Mini AN4 source tree is already available."""
        return self._asr_builder.is_source_prepared(recipe_dir=recipe_dir)

    def prepare_source(self, recipe_dir: str | Path, **_kwargs) -> None:
        """Extract the shared Mini AN4 source tree if it is not available yet."""
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir=recipe_root):
            return

        dataset_root = recipe_root / "downloads"
        archive = self._resolve_archive(recipe_root)
        dataset_root.mkdir(parents=True, exist_ok=True)
        _normalize_downloads_layout(dataset_root)
        subprocess.run(
            ["tar", "-xzf", str(archive), "-C", str(dataset_root)],
            check=True,
        )
        _normalize_downloads_layout(dataset_root)

    def is_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check whether TTS manifests and the token list already exist."""
        recipe_root = Path(recipe_dir).resolve()
        data_root = recipe_root / _CFG["data_path"]
        token_list = recipe_root / _CFG["token_list_path"]
        manifest_paths = _CFG["manifest_paths"].values()
        return token_list.is_file() and all(
            (data_root / relpath).is_file() for relpath in manifest_paths
        )

    def build(self, recipe_dir: str | Path, **_kwargs) -> None:
        """Build shared manifests and generate a character token list for TTS."""
        recipe_root = Path(recipe_dir).resolve()
        self._asr_builder.build(recipe_dir=recipe_root)

        manifest_path = recipe_root / _CFG["data_path"] / _CFG["manifest_paths"]["train"]
        chars: set[str] = set()
        with manifest_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", maxsplit=2)
                if len(parts) != 3:
                    raise ValueError(f"Invalid manifest line: {line}")
                chars.update(char for char in parts[2] if char != " ")

        token_list = recipe_root / _CFG["token_list_path"]
        token_list.parent.mkdir(parents=True, exist_ok=True)
        tokens = ["<unk>", "<space>", *sorted(chars)]
        token_list.write_text("\n".join(tokens) + "\n", encoding="utf-8")
