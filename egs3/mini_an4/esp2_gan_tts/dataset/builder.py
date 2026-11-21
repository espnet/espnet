"""Mini AN4 GAN-TTS dataset builder: the ASR manifests plus a speaker column."""

from __future__ import annotations

from pathlib import Path

import egs3.mini_an4.esp2_asr.dataset.builder as asr_builder
from egs3.mini_an4.esp2_asr.dataset.builder import MiniAn4Builder
from espnet3.utils.config_utils import load_config_with_defaults

# The ASR recipe's builder settings: where it writes the data and manifests.
_ASR_BUILDER_CONFIG = load_config_with_defaults(
    str(Path(asr_builder.__file__).with_name("config.yaml")), resolve=False
)["builder"]


def resolve_manifest_path(recipe_dir: str | Path, split: str) -> Path:
    """Return the manifest the builder writes for one split.

    Args:
        recipe_dir: Recipe root directory.
        split: ``train``, ``valid`` or ``test``.

    Returns:
        Path of the split's manifest under the recipe's data directory.

    Raises:
        ValueError: If ``split`` is not one of the manifest splits.

    Example:
        .. code-block:: python

            >>> resolve_manifest_path("egs3/mini_an4/esp2_gan_tts", "train").name
            'train.tsv'
    """
    manifest_paths = _ASR_BUILDER_CONFIG["manifest_paths"]
    if split not in manifest_paths:
        known = ", ".join(sorted(manifest_paths))
        raise ValueError(f"Unknown split '{split}'. Expected one of: {known}")
    data_dir = Path(recipe_dir).resolve() / _ASR_BUILDER_CONFIG["data_path"]
    return data_dir / manifest_paths[split]


class MiniAn4GANTTSBuilder(MiniAn4Builder):
    """Build the Mini AN4 manifests with the speaker id as a fourth column.

    The ASR recipe's manifests hold ``utt_id``, ``wav_path`` and ``text``. The
    ``compute_xvectors`` stage also needs a speaker id per row, so this builder
    appends the AN4 speaker, the part of the utterance id before its first
    ``-``, to every row.

    Example:
        .. code-block:: python

            >>> builder = MiniAn4GANTTSBuilder()
            >>> if not builder.is_built(recipe_dir="."):
            ...     builder.build(recipe_dir=".")
    """

    @staticmethod
    def _resolve_manifest_paths(recipe_dir: str | Path) -> list[Path]:
        """Return the manifest of every split, in the config's order."""
        return [
            resolve_manifest_path(recipe_dir, split)
            for split in _ASR_BUILDER_CONFIG["manifest_paths"]
        ]

    @staticmethod
    def _has_speaker_column(manifest_path: Path) -> bool:
        """Check that every non-blank row of a manifest has four columns."""
        rows = [
            line
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return bool(rows) and all(len(row.split("\t")) >= 4 for row in rows)

    def is_built(self, recipe_dir: str | Path, **kwargs) -> bool:
        """Check that the manifests exist and carry the speaker column.

        Args:
            recipe_dir: Recipe root directory.
            **kwargs: Unused extra options for API compatibility.

        Returns:
            ``True`` when the ASR builder's outputs exist and every manifest
            row has a speaker id.

        Example:
            .. code-block:: python

                >>> MiniAn4GANTTSBuilder().is_built(recipe_dir=".")
                True
        """
        if not super().is_built(recipe_dir=recipe_dir, **kwargs):
            return False
        return all(
            self._has_speaker_column(path)
            for path in self._resolve_manifest_paths(recipe_dir)
        )

    def build(self, recipe_dir: str | Path, **kwargs) -> None:
        """Build the ASR manifests, then append the speaker id to every row.

        Args:
            recipe_dir: Recipe root directory.
            **kwargs: Unused extra options for API compatibility.

        Returns:
            None. The manifests are rewritten in place.

        Raises:
            RuntimeError: If the AN4 source is not prepared, or ``sph2pipe``
                is not installed.

        Example:
            .. code-block:: python

                >>> MiniAn4GANTTSBuilder().build(recipe_dir=".")
        """
        if not super().is_built(recipe_dir=recipe_dir, **kwargs):
            super().build(recipe_dir=recipe_dir, **kwargs)
        for manifest_path in self._resolve_manifest_paths(recipe_dir):
            rows = []
            for line in manifest_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                utt_id, wav_path, text = line.split("\t")[:3]
                speaker_id = utt_id.split("-", maxsplit=1)[0]
                rows.append(f"{utt_id}\t{wav_path}\t{text}\t{speaker_id}\n")
            manifest_path.write_text("".join(rows), encoding="utf-8")
