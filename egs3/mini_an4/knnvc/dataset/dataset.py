"""Dataset for the mini_an4 kNN-VC recipe.

The same three ``kind``s as ``egs3/librispeech_100/knnvc``, over the an4 audio
that ships with the repository: ``audio`` for ``prepare_features``, ``vocoder``
for ``train`` and ``conversion`` for ``infer``. Speaker directories are the
prematching pools, standing in for LibriSpeech's chapter directories. The
``conversion`` view carries the source transcript from an4's
``etc/an4_train.transcription``, so ``measure`` can score it as the
LibriSpeech recipe does; the one bundled utterance without a transcript
(``cen7-fash-b``) serves as a reference only, never as a source.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

import egs3.TEMPLATE.knnvc.src.contracts  # noqa: F401  registers the kinds below
from egs3.mini_an4.knnvc.dataset.builder import speaker_root
from espnet3.components.data.base_dataset import BaseDataset
from espnet3.contract import Field

FEATURE_SUFFIX = ".npy"
_KINDS = ("audio", "vocoder", "conversion")
_FIELDS = {
    "audio": (Field("speech", "audio"),),
    "vocoder": (Field("feats", "features"), Field("speech", "audio")),
    "conversion": (
        Field("speech", "audio"),
        Field("reference_speech", "reference_audio"),
        Field("target_speaker", "text"),
        Field("pair_id", "text"),
        Field("text", "text"),
    ),
}
# `<s> YES </s> (an251-fash-b)`: the words, then the utterance id in parentheses.
_TRANSCRIPTION_LINE = re.compile(
    r"^(?:<s>\s*)?(?P<words>.*?)(?:\s*</s>)?\s*\((?P<utt>\S+)\)$"
)


def _read_transcripts(root: Path) -> Dict[str, str]:
    """Map utterance id to transcript from ``an4/etc/an4_train.transcription``.

    Args:
        root: The speaker directory root, ``.../an4/wav/an4_clstk``.

    Returns:
        ``{"an251-fash-b": "YES", ...}`` for the training speakers.

    Raises:
        FileNotFoundError: If the transcription file is not extracted.
    """
    path = root.parents[1] / "etc" / "an4_train.transcription"
    if not path.is_file():
        raise FileNotFoundError(f"an4 transcripts not found: {path}")
    transcripts: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _TRANSCRIPTION_LINE.match(line.strip())
        if match:
            transcripts[match.group("utt")] = match.group("words").strip()
    return transcripts


def _read_audio(path: Path) -> np.ndarray:
    """Read one utterance as float32 mono."""
    array, sample_rate = sf.read(str(path), dtype="float32")
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz audio, got {sample_rate} for {path}")
    return np.ascontiguousarray(array)


class MiniAn4KNNVCDataset(BaseDataset):
    """an4 utterances served in the three kNN-VC stage views.

    Args:
        kind: ``"audio"``, ``"vocoder"`` or ``"conversion"``; the declared
            ``fields`` follow it, as in ``egs3/librispeech_100/knnvc``.
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
        self.fields = _FIELDS[kind]
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
        self.transcripts: Dict[str, str] = {}
        if kind == "conversion":
            self.transcripts = _read_transcripts(root)
            rng = random.Random(seed)
            speakers = sorted({path.parent.name for path in self.paths})
            pairs = []
            for path in self.paths:
                if path.stem not in self.transcripts:
                    continue  # no transcript to score against: reference only
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
            "text": self.transcripts[path.stem],
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
