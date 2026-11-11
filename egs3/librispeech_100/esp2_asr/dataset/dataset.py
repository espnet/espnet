"""LibriSpeech 100h dataset backed by saved Hugging Face indexes."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from egs3.librispeech_100.esp2_asr.dataset.builder import _resolve_cache_dir
from espnet3.components.data.base_dataset import BaseDataset
from espnet3.contract import Field
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_SUPPORTED_SPLITS = tuple(
    str(split) for split in _CONFIG["dataset"]["supported_splits"]
)


class LibriSpeech100Dataset(BaseDataset):
    """Read LibriSpeech audio and transcripts from recipe-local HF Datasets.

    Args:
        split: LibriSpeech split name, for example ``train-clean-100``.
        recipe_dir: Optional recipe root that owns the saved dataset indexes.

    Raises:
        ValueError: If ``split`` is not supported by this recipe.
        ImportError: If the ESPnet ``train`` extra is not installed.
        FileNotFoundError: If the split has not been built or audio is missing.
        RuntimeError: If an audio sample is not 16 kHz or lacks a transcript.

    Examples:
        >>> dataset = LibriSpeech100Dataset("dev-clean")
        >>> sample = dataset[0]
        >>> sorted(sample)
        ['speech', 'text']
    """

    fields = (Field("speech", "audio"), Field("text", "text"))

    def __init__(self, split: str, recipe_dir: str | Path | None = None) -> None:
        self.split = str(split)
        if self.split not in _SUPPORTED_SPLITS:
            known = ", ".join(_SUPPORTED_SPLITS)
            raise ValueError(f"Unknown split '{self.split}'. Expected one of: {known}")
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise ImportError(
                "LibriSpeech loading requires Hugging Face Datasets. "
                "Install the ESPnet training extra with `pip install 'espnet[train]'`."
            ) from exc

        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        dataset_path = _resolve_cache_dir(recipe_root) / self.split
        if not (dataset_path / "state.json").is_file():
            raise FileNotFoundError(
                f"Built LibriSpeech split not found at {dataset_path}. "
                "Run the recipe's create_dataset stage first."
            )
        self._dataset = load_from_disk(str(dataset_path))

    def __len__(self) -> int:
        """Return the number of utterances in this split."""
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load one waveform and transcript from the saved path/text row.

        Args:
            idx: Zero-based utterance index.

        Returns:
            dict[str, Any]: Float32 waveform in ``speech`` and transcript in ``text``.

        Example:
            >>> sample = dataset[0]
            >>> sample["speech"].dtype
            dtype('float32')
        """
        row = self._dataset[int(idx)]
        audio_path = row.get("audio")
        text = row.get("text")
        if not isinstance(audio_path, str) or not Path(audio_path).is_file():
            raise FileNotFoundError(f"LibriSpeech audio file is missing: {audio_path}")
        if not isinstance(text, str):
            raise RuntimeError("Saved LibriSpeech row has no text transcript")

        waveform, sample_rate = sf.read(audio_path, dtype="float32")
        if sample_rate != 16000:
            raise RuntimeError(
                f"Expected LibriSpeech audio at 16000 Hz, got {sample_rate} Hz"
            )
        return {"speech": np.asarray(waveform, dtype=np.float32), "text": text}
