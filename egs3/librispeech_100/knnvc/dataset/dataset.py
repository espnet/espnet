"""LibriSpeech 100h dataset for the kNN-VC recipe (three item kinds).

The same corpus index serves three stages, selected by ``kind``:

- ``audio`` (``prepare_features``): one waveform per utterance plus the
  ``get_pool_key`` / ``get_feature_name`` lookups the stage needs;
- ``vocoder`` (``train``): aligned ``(feats, speech)`` segments read from the
  precomputed feature files;
- ``conversion`` (``infer``): source utterance + target speaker reference set
  pairs for any-to-any voice conversion.
"""

from __future__ import annotations

import os
import random
import zlib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from egs3.librispeech_100.knnvc.dataset.builder import resolve_source_root
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]

_KNOWN_SPLITS = {str(split) for split in _DATASET_CFG["supported_splits"]}
_HOP_LENGTH = int(_DATASET_CFG["hop_length"])
_SAMPLE_RATE = int(_DATASET_CFG["sample_rate"])
_KINDS = ("audio", "vocoder", "conversion")
_PREMATCH_POOLS = ("chapter", "speaker")
_SUBSETS = (None, "train", "valid")
FEATURE_SUFFIX = ".npy"


@dataclass(frozen=True)
class LibriSpeechExample:
    """Internal index entry derived from a LibriSpeech transcript line."""

    utt_id: str
    speaker: str
    chapter: str
    audio_path: Path
    feature_name: str
    text: str


def _scan_split(librispeech_root: Path, split: str) -> List[LibriSpeechExample]:
    """Build the utterance index of one split from its ``*.trans.txt`` files."""
    split_dir = librispeech_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    examples: List[LibriSpeechExample] = []
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
                    audio_path = root_path / f"{utt_id}.flac"
                    if not audio_path.is_file():
                        continue
                    feature_name = (
                        audio_path.relative_to(librispeech_root)
                        .with_suffix("")
                        .as_posix()
                    )
                    examples.append(
                        LibriSpeechExample(
                            utt_id=utt_id,
                            speaker=utt_id.split("-")[0],
                            chapter=utt_id.split("-")[1],
                            audio_path=audio_path.resolve(),
                            feature_name=feature_name,
                            text=" ".join(words),
                        )
                    )
    if not examples:
        raise RuntimeError(
            f"No transcript/audio pairs found under: {split_dir}. "
            "Check that the split is extracted and the path is correct."
        )
    return sorted(examples, key=lambda example: example.utt_id)


def _is_valid_utterance(utt_id: str, valid_ratio: float) -> bool:
    """Deterministic utterance-level train/valid split by CRC32 hash."""
    return (zlib.crc32(utt_id.encode("utf-8")) % 10000) < int(valid_ratio * 10000)


def _read_audio(path: Path) -> np.ndarray:
    """Read one 16 kHz utterance as a contiguous float32 array."""
    array, sample_rate = sf.read(str(path), dtype="float32")
    if sample_rate != _SAMPLE_RATE:
        raise ValueError(f"{path}: expected {_SAMPLE_RATE} Hz audio, got {sample_rate}")
    if array.ndim > 1:
        array = array.mean(axis=-1)
    return np.ascontiguousarray(array, dtype=np.float32)


class LibriSpeech100Dataset(TorchDataset):
    """LibriSpeech dataset for the kNN-VC recipe.

    Args:
        split: LibriSpeech split directory name such as ``train-clean-100``.
        kind: ``"audio"``, ``"vocoder"`` or ``"conversion"`` (see the module
            docstring for what each item contains).
        prematch_pool: Which utterances form an utterance's prematching pool in
            ``prepare_features`` (``audio`` only). ``"chapter"`` (default) pools
            the other utterances of the same speaker *chapter directory*, which
            is what the official ``prematch_dataset.py`` does
            (``path.parent.rglob``) and what the released vocoders were trained
            on; ``"speaker"`` pools all other utterances of the speaker, as
            the paper describes.
        recipe_dir: Optional recipe root; defaults to this recipe's directory.
        source_dir: Optional LibriSpeech parent/root override.
        features_dir: Root of the ``prepare_features`` output (``vocoder`` only).
        subset: ``"train"`` / ``"valid"`` utterance-level holdout selection, or
            ``None`` for the whole split (``vocoder`` only).
        valid_ratio: Fraction of utterances assigned to ``subset="valid"``.
        segment_frames: Number of feature frames per training item; the
            waveform segment is ``segment_frames * hop_length`` samples.
            ``None`` returns whole utterances (``vocoder`` only).
        num_pairs: Number of conversion pairs to build, ``None`` for one per
            utterance (``conversion`` only). Pairs are ordered by target
            speaker so that reference audio and features are decoded and
            encoded once per speaker.
        seed: Seed for pair sampling (``conversion`` only).
        max_reference_seconds: Cap on the total duration of the target
            speaker's reference utterances (``conversion`` only).

    Item contract:

    - ``kind="audio"``: ``{"speech": float32 (samples,)}``;
    - ``kind="vocoder"``: ``{"feats": float32 (frames, dim), "speech": float32
      (frames * hop_length,)}``;
    - ``kind="conversion"``: ``{"speech", "reference_speech": list[array],
      "target_speaker": str, "pair_id": str, "text": str}``.

    Raises:
        ValueError: On unknown ``split`` / ``kind`` / ``subset`` or when
            ``features_dir`` is missing for ``kind="vocoder"``.
        FileNotFoundError: If the LibriSpeech root or the requested split
            directory cannot be found. Other splits are not required here; the
            ``create_dataset`` stage checks the full recipe requirements.

    Examples:
        >>> audio = LibriSpeech100Dataset("dev-clean", kind="audio")
        >>> audio.get_pool_key(0), audio.get_feature_name(0)
        ('1272/128104', 'dev-clean/1272/128104/1272-128104-0000')
        >>> vocoder = LibriSpeech100Dataset(
        ...     "train-clean-100", kind="vocoder", features_dir="data/features",
        ...     subset="train", segment_frames=22,
        ... )
        >>> vocoder[0]["feats"].shape, vocoder[0]["speech"].shape
        ((22, 1024), (7040,))
    """

    def __init__(
        self,
        split: str,
        kind: str = "audio",
        prematch_pool: str = "chapter",
        recipe_dir: str | Path | None = None,
        source_dir: str | Path | None = None,
        features_dir: str | Path | None = None,
        subset: Optional[str] = None,
        valid_ratio: float = 0.05,
        segment_frames: Optional[int] = 22,
        num_pairs: Optional[int] = 200,
        seed: int = 1234,
        max_reference_seconds: Optional[float] = 300.0,
    ) -> None:
        """Index the split and prepare the item kind."""
        self.split = str(split)
        if self.split not in _KNOWN_SPLITS:
            known = ", ".join(sorted(_KNOWN_SPLITS))
            raise ValueError(f"Unknown split '{self.split}'. Expected one of: {known}")
        if kind not in _KINDS:
            raise ValueError(f"Unknown kind '{kind}'. Expected one of: {_KINDS}")
        if subset not in _SUBSETS:
            raise ValueError(f"Unknown subset '{subset}'. Expected one of: {_SUBSETS}")
        if prematch_pool not in _PREMATCH_POOLS:
            raise ValueError(
                f"Unknown prematch_pool '{prematch_pool}'. "
                f"Expected one of: {_PREMATCH_POOLS}"
            )
        self.kind = kind
        self.prematch_pool = prematch_pool
        self.hop_length = _HOP_LENGTH
        self.sample_rate = _SAMPLE_RATE

        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        # Only the requested split has to exist here; `create_dataset` (the
        # builder) is what enforces the full set of required splits. This keeps
        # `infer` with the released checkpoints runnable from `dev-clean` alone.
        self.librispeech_root = resolve_source_root(recipe_root, source_dir=source_dir)

        self._examples = _scan_split(self.librispeech_root, self.split)

        if kind == "vocoder":
            if features_dir is None:
                raise ValueError("features_dir is required for kind='vocoder'.")
            self.features_dir = Path(features_dir)
            self.segment_frames = (
                None if segment_frames is None else int(segment_frames)
            )
            if subset is not None:
                want_valid = subset == "valid"
                self._examples = [
                    example
                    for example in self._examples
                    if _is_valid_utterance(example.utt_id, valid_ratio) == want_valid
                ]
        elif kind == "conversion":
            self.max_reference_seconds = max_reference_seconds
            self._reference_cache: Dict[str, List[LibriSpeechExample]] = {}
            # Pairs are ordered by target speaker, so caching the most recent
            # speaker's decoded references is enough to avoid re-reading up to
            # `max_reference_seconds` of audio for every pair.
            self._reference_audio_cache: Dict[str, List[np.ndarray]] = {}
            self._pairs = self._build_pairs(num_pairs, seed)

    # ------------------------------------------------------------------
    # prepare_features contract (kind="audio")
    # ------------------------------------------------------------------
    def get_pool_key(self, idx: int) -> str:
        """Return the prematching-pool key of item ``idx`` without loading audio.

        ``"<speaker>/<chapter>"`` for ``prematch_pool="chapter"`` (official
        behaviour), ``"<speaker>"`` for ``prematch_pool="speaker"``.
        """
        example = self._examples[int(idx)]
        if self.prematch_pool == "speaker":
            return example.speaker
        return f"{example.speaker}/{example.chapter}"

    def get_feature_name(self, idx: int) -> str:
        """Return item ``idx``'s feature path (no suffix) relative to features_dir."""
        return self._examples[int(idx)].feature_name

    # ------------------------------------------------------------------
    # conversion pairs (kind="conversion")
    # ------------------------------------------------------------------
    def _build_pairs(self, num_pairs: Optional[int], seed: int) -> List[Dict]:
        """Deterministically pair source utterances with other speakers."""
        rng = random.Random(seed)
        speakers = sorted({example.speaker for example in self._examples})
        if len(speakers) < 2:
            raise RuntimeError(
                f"Split '{self.split}' has {len(speakers)} speaker(s); conversion "
                "pairs need at least two."
            )
        sources = list(self._examples)
        rng.shuffle(sources)
        if num_pairs is not None:
            sources = sources[: int(num_pairs)]
        pairs = []
        for example in sources:
            candidates = [spk for spk in speakers if spk != example.speaker]
            target = rng.choice(candidates)
            pairs.append({"source": example, "target_speaker": target})
        # Group by target speaker so consecutive items reuse one reference set
        # (both the decoded audio here and the encoded features in the model).
        return sorted(
            pairs, key=lambda pair: (pair["target_speaker"], pair["source"].utt_id)
        )

    def _get_reference_examples(self, speaker: str) -> List[LibriSpeechExample]:
        """Return the reference utterances of ``speaker`` (cached, duration-capped)."""
        if speaker in self._reference_cache:
            return self._reference_cache[speaker]
        selected = []
        total = 0.0
        for example in self._examples:
            if example.speaker != speaker:
                continue
            if self.max_reference_seconds is not None:
                duration = sf.info(str(example.audio_path)).duration
                if selected and total + duration > self.max_reference_seconds:
                    break
                total += duration
            selected.append(example)
        self._reference_cache[speaker] = selected
        return selected

    # ------------------------------------------------------------------
    # torch Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Return the number of items for the configured ``kind``."""
        if self.kind == "conversion":
            return len(self._pairs)
        return len(self._examples)

    def _getitem_vocoder(self, example: LibriSpeechExample) -> Dict[str, Any]:
        """Return one aligned feature/waveform segment for vocoder training."""
        feats_path = self.features_dir / (example.feature_name + FEATURE_SUFFIX)
        if not feats_path.is_file():
            raise FileNotFoundError(
                f"Feature file not found: {feats_path}. Run the prepare_features "
                "stage with features_dir pointing at the same directory."
            )
        feats = np.load(feats_path).astype(np.float32)
        speech = _read_audio(example.audio_path)

        num_frames = min(feats.shape[0], speech.shape[0] // self.hop_length)
        feats = feats[:num_frames]
        speech = speech[: num_frames * self.hop_length]

        if self.segment_frames is not None:
            seg = self.segment_frames
            if num_frames > seg:
                start = random.randint(0, num_frames - seg)
                feats = feats[start : start + seg]
                hop = self.hop_length
                speech = speech[start * hop : (start + seg) * hop]
            elif num_frames < seg:
                feats = np.pad(feats, ((0, seg - num_frames), (0, 0)))
                speech = np.pad(speech, (0, seg * self.hop_length - speech.shape[0]))
        return {"feats": feats, "speech": speech}

    def _read_reference_audio(self, target_speaker: str) -> List[np.ndarray]:
        """Return the target speaker's reference waveforms, decoding once."""
        cached = self._reference_audio_cache.get(target_speaker)
        if cached is not None:
            return cached
        references = self._get_reference_examples(target_speaker)
        waveforms = [_read_audio(ref.audio_path) for ref in references]
        # Keep only the speaker in flight; the full set would be gigabytes.
        self._reference_audio_cache = {target_speaker: waveforms}
        return waveforms

    def _getitem_conversion(self, pair: Dict) -> Dict[str, Any]:
        """Return one source utterance with its target speaker's references."""
        source: LibriSpeechExample = pair["source"]
        target_speaker: str = pair["target_speaker"]
        return {
            "speech": _read_audio(source.audio_path),
            "reference_speech": self._read_reference_audio(target_speaker),
            "target_speaker": target_speaker,
            "pair_id": f"{source.utt_id}_to_{target_speaker}",
            "text": source.text,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one item, shaped by ``kind``.

        Args:
            idx: Item index.

        Returns:
            For ``kind="audio"``, the utterance for ``prepare_features``::

                {"speech": float32 (93680,)}

            For ``kind="vocoder"``, an aligned training segment of
            ``segment_frames`` WavLM frames and ``segment_frames * 320``
            samples::

                {"feats": float32 (22, 1024), "speech": float32 (7040,)}

            For ``kind="conversion"``, a source utterance and up to
            ``max_reference_seconds`` of the target speaker's audio::

                {
                    "speech": float32 (45120,),
                    "reference_speech": [float32 (318240,), ...],  # 33 arrays
                    "target_speaker": "3536",
                    "pair_id": "5536-43363-0000_to_3536",
                    "text": "REINCARNATION AND THE CONVERSE OF SPIRITS",
                }

            The shapes are from dev-clean with the recipe defaults.
        """
        if self.kind == "conversion":
            return self._getitem_conversion(self._pairs[int(idx)])
        example = self._examples[int(idx)]
        if self.kind == "vocoder":
            return self._getitem_vocoder(example)
        return {"speech": _read_audio(example.audio_path)}
