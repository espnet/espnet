"""Mini AN4 ST dataset: AN4 audio with a synthetic target-language side."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from egs3.mini_an4.esp2_st.dataset.builder import MiniAn4STBuilder
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]
_BUILDER_CFG = _CONFIG["builder"]

_SPLIT_MANIFEST_PATHS: dict[str, str] = {
    str(split): str(relpath)
    for split, relpath in _DATASET_CFG["split_manifest_paths"].items()
}
TRANSLATION: dict[str, str] = {
    str(k): str(v) for k, v in _DATASET_CFG["translation"].items()
}


@dataclass(frozen=True)
class ManifestEntry:
    """One manifest row: utterance id, wav path, and transcript text."""

    utt_id: str
    wav_path: Path
    text: str


def translate(text: str) -> str:
    """Translate word by word through the table, passing unknown words through."""
    return " ".join(TRANSLATION.get(word, word) for word in text.split())


def _read_manifest(manifest_path: Path) -> list[ManifestEntry]:
    """Read ``utt_id<TAB>wav_path<TAB>text`` lines as manifest entries."""
    entries: list[ManifestEntry] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            utt_id, wav_path, text = line.split("\t", maxsplit=2)
            entries.append(ManifestEntry(utt_id, Path(wav_path), text))
    if not entries:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")
    return entries


class MiniAn4STDataset(TorchDataset):
    """AN4 as a speech-translation set, for the integration test only.

    ``src_text`` is AN4's English transcript and ``text`` is its synthetic
    translation, so the two streams differ and a crossed pair fails the test.

    Args:
        split: One of ``train``, ``valid``, ``test``.
        recipe_dir: Recipe root; inferred from this file when omitted.
        return_utt_id: Add ``utt_id`` to each sample. Inference needs it;
            training cannot take it, because ``CommonCollateFn`` stacks every
            value and a str has no dtype.

    Examples:
        >>> sample = MiniAn4STDataset(split="test", return_utt_id=True)[0]
        >>> sorted(sample)
        ['speech', 'src_text', 'text', 'utt_id']
    """

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        return_utt_id: bool = False,
    ) -> None:
        self.split = split
        self.return_utt_id = bool(return_utt_id)
        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        self.dataset_dir = recipe_root / _BUILDER_CFG["data_path"]

        builder = MiniAn4STBuilder()
        if not builder.is_source_prepared(recipe_dir=recipe_root):
            builder.prepare_source(recipe_dir=recipe_root)
        if not builder.is_built(recipe_dir=recipe_root):
            builder.build(recipe_dir=recipe_root)

        if split not in _SPLIT_MANIFEST_PATHS:
            known = ", ".join(sorted(_SPLIT_MANIFEST_PATHS))
            raise ValueError(f"Unknown split '{split}'. Expected one of: {known}")

        manifest_path = self.dataset_dir / _SPLIT_MANIFEST_PATHS[split]
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        self._entries = _read_manifest(manifest_path)

    def __len__(self) -> int:
        """Number of utterances in the split."""
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one sample: speech, target text, source text, optional id."""
        entry = self._entries[int(idx)]
        array, _sr = sf.read(str(entry.wav_path))
        sample = {
            "speech": np.asarray(array, dtype=np.float32),
            "text": translate(entry.text),
            "src_text": entry.text,
        }
        if self.return_utt_id:
            sample["utt_id"] = entry.utt_id
        return sample


def gather_training_text(
    recipe_dir: str | Path | None = None, side: str = "tgt", **_kwargs
) -> list[str]:
    """Collect one side's training text for a SentencePiece model.

    Args:
        recipe_dir: Recipe root; inferred when omitted.
        side: ``tgt`` for the translated stream, ``src`` for the transcript.

    Returns:
        The requested side's lines, in manifest order.

    Raises:
        ValueError: If ``side`` is neither ``tgt`` nor ``src``.
    """
    if side not in {"tgt", "src"}:
        raise ValueError(f"Unknown side {side!r}; expected tgt or src")
    dataset = MiniAn4STDataset(split="train", recipe_dir=recipe_dir)
    return [
        translate(entry.text) if side == "tgt" else entry.text
        for entry in dataset._entries
    ]
