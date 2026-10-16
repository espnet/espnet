"""Dataset for the mini_an4 kNN-VC recipe.

The same three ``kind``s as ``egs3/librispeech_100/knnvc``, over the an4 audio
that ships with the repository: ``audio`` for ``prepare_features``, ``vocoder``
for ``train`` and ``conversion`` for ``infer``. Speaker directories are the
prematching pools, standing in for LibriSpeech's chapter directories.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from egs3.mini_an4.knnvc.dataset.builder import speaker_root

FEATURE_SUFFIX = ".npy"
_KINDS = ("audio", "vocoder", "conversion")


def _read_audio(path: Path) -> np.ndarray:
    """Read one utterance as float32 mono."""
    array, sample_rate = sf.read(str(path), dtype="float32")
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz audio, got {sample_rate} for {path}")
    return np.ascontiguousarray(array)


class MiniAn4KNNVCDataset(TorchDataset):
    """an4 utterances served in the three kNN-VC stage views.

    Args:
        kind: ``"audio"``, ``"vocoder"`` or ``"conversion"``.
        recipe_dir: Recipe root; defaults to this recipe's directory.
        features_dir: Root of the ``prepare_features`` output (``vocoder``).
        hop_length: Encoder frame hop in samples.
        segment_frames: Feature frames per training item, or ``None`` for whole
            utterances (``vocoder`` only).
        num_pairs: Number of conversion pairs to build (``conversion`` only).
        seed: Seed for target-speaker sampling (``conversion`` only).

    Raises:
        ValueError: On an unknown ``kind``, or a missing ``features_dir`` for
            ``kind="vocoder"``.
        FileNotFoundError: If the corpus has not been extracted yet.
    """

    def __init__(
        self,
        kind: str = "audio",
        recipe_dir: Optional[str] = None,
        features_dir: Optional[str] = None,
        hop_length: int = 80,
        segment_frames: Optional[int] = 8,
        num_pairs: int = 2,
        seed: int = 0,
        **_kwargs: Any,
    ) -> None:
        """Index the corpus and, for ``conversion``, build the pair list."""
        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
        if kind == "vocoder" and features_dir is None:
            raise ValueError("features_dir is required for kind='vocoder'")

        self.kind = kind
        self.hop_length = int(hop_length)
        self.segment_frames = segment_frames
        self.features_dir = Path(features_dir) if features_dir else None

        root = speaker_root(recipe_dir)
        self.paths: List[Path] = sorted(root.glob("*/*.sph"))
        if not self.paths:
            raise FileNotFoundError(
                f"No .sph utterances under {root}. Run the create_dataset stage."
            )

        self.pairs: List[tuple] = []
        if kind == "conversion":
            rng = random.Random(seed)
            speakers = sorted({path.parent.name for path in self.paths})
            pairs = []
            for path in self.paths:
                targets = [s for s in speakers if s != path.parent.name]
                if targets:
                    pairs.append((path, rng.choice(targets)))
            # Ordered by target speaker so references are encoded once each.
            self.pairs = sorted(pairs, key=lambda p: (p[1], p[0].name))[:num_pairs]

    def get_pool_key(self, idx: int) -> str:
        """Return the prematching pool of item ``idx``: its speaker."""
        return self.paths[int(idx)].parent.name

    def get_feature_name(self, idx: int) -> str:
        """Return item ``idx``'s feature path (no suffix) under features_dir."""
        path = self.paths[int(idx)]
        return f"{path.parent.name}/{path.stem}"

    def __len__(self) -> int:
        """Return the number of items in this view."""
        return len(self.pairs) if self.kind == "conversion" else len(self.paths)

    def _conversion_item(self, idx: int) -> Dict[str, Any]:
        """Build one source/target-speaker conversion item."""
        path, target = self.pairs[int(idx)]
        references = [p for p in self.paths if p.parent.name == target]
        return {
            "speech": _read_audio(path),
            "reference_speech": [_read_audio(p) for p in references],
            "target_speaker": target,
            "pair_id": f"{path.stem}_to_{target}",
        }

    def _vocoder_item(self, idx: int) -> Dict[str, np.ndarray]:
        """Build one aligned feature/waveform training segment."""
        path = self.paths[int(idx)]
        speech = _read_audio(path)
        feats = np.load(
            self.features_dir / f"{self.get_feature_name(idx)}{FEATURE_SUFFIX}"
        ).astype(np.float32)
        n_frames = min(feats.shape[0], speech.shape[0] // self.hop_length)
        feats = feats[:n_frames]
        speech = speech[: n_frames * self.hop_length]
        if self.segment_frames and n_frames > self.segment_frames:
            start = random.randint(0, n_frames - self.segment_frames)
            feats = feats[start : start + self.segment_frames]
            speech = speech[
                start
                * self.hop_length : (start + self.segment_frames)
                * self.hop_length
            ]
        return {"feats": feats, "speech": speech}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one item according to ``kind``."""
        if self.kind == "conversion":
            return self._conversion_item(idx)
        if self.kind == "vocoder":
            return self._vocoder_item(idx)
        return {"speech": _read_audio(self.paths[int(idx)])}
