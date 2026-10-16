"""Lhotse-backed diarization dataset for Sortformer.

Each item is a fixed-length window of (mono) audio plus a frame-level speaker
activity matrix derived from the cut's supervisions:

    {
      "speech":      float32 (num_samples,),
      "spk_labels":  float32 (num_frames, num_spk),   # 80 ms frames
      "utt_id":      str,
    }

Training data is typically FastMSS-simulated LibriSpeech meetings; evaluation
data is AMI mixed-headset windows (see ``src/data_prep.py``). Frame count matches
the model's output rate: ``ceil(ceil(num_samples / hop) / subsampling_factor)``.

Splits map to lhotse ``CutSet`` paths via ``dataset/config.yaml``.
"""

import math
from pathlib import Path
from typing import Any, Dict, Optional

import lhotse
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import Dataset as TorchDataset


def num_frames_from_samples(
    num_samples: int, hop: int = 160, subsampling_factor: int = 8
) -> int:
    """Compute the model output frame count for a number of audio samples.

    Mirrors the model's front end: STFT framing followed by encoder subsampling,
    i.e. ``ceil(ceil(num_samples / hop) / subsampling_factor)``.

    Args:
        num_samples: Number of audio samples in the window.
        hop: STFT hop length in samples (160 = 10 ms at 16 kHz).
        subsampling_factor: Encoder subsampling factor (8 for the 80 ms rate).

    Returns:
        The number of output frames.
    """
    mel_frames = math.ceil(num_samples / hop)
    return int(math.ceil(mel_frames / subsampling_factor))


class LhotseDiarDataset(TorchDataset):
    """Lhotse-backed diarization dataset (exported to the recipe as ``Dataset``).

    Wraps a lhotse ``CutSet`` and yields, per index, a fixed-length audio window
    with a frame-level speaker-activity matrix. Speakers in each cut are ranked by
    total speaking time and the top ``num_spk`` are kept (one column each).

    Each item is a dict::

        {
          "speech":     float32 (num_samples,),          # mono, one channel
          "spk_labels": float32 (num_frames, num_spk),   # frame_dur-wide frames
          "utt_id":     str,                             # the cut id
        }

    where ``num_frames`` matches the model's output rate (see
    :func:`num_frames_from_samples`).

    The split's cuts are taken from the ``cuts`` argument when given, otherwise
    resolved from ``dataset/config.yaml`` via the ``splits`` mapping (which also
    supplies ``num_spk`` and ``frame_dur`` defaults).
    """

    def __init__(
        self,
        split: str,
        recipe_dir: Optional[str] = None,
        cuts: Optional[str] = None,
        num_spk: int = 4,
        frame_dur: float = 0.08,
        sample_rate: int = 16000,
        channel: int = 0,
    ) -> None:
        """Load the split's cuts and configure framing.

        Args:
            split: Split name; looked up in ``dataset/config.yaml`` ``splits``
                when ``cuts`` is not given.
            recipe_dir: Recipe root for resolving ``config.yaml`` and relative
                cut paths; defaults to this file's parent recipe directory.
            cuts: Optional explicit path to a lhotse cut file, bypassing the
                config lookup.
            num_spk: Max speakers kept per cut (overridden by the config when
                resolving from ``config.yaml``).
            frame_dur: Label frame duration in seconds (default 80 ms).
            sample_rate: Audio sample rate in Hz.
            channel: Channel index to read from multi-channel cuts.

        Raises:
            ValueError: If ``split`` is unknown in the config.
            FileNotFoundError: If the resolved cut file does not exist (run the
                ``data_preparation`` stage first).
        """
        self.split = split
        self.num_spk = num_spk
        self.frame_dur = frame_dur
        self.sample_rate = sample_rate
        self.channel = channel
        self.hop = int(round(frame_dur * sample_rate / 8))  # 160 for 80ms/8x

        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        if cuts is None:
            cfg = OmegaConf.load(recipe_root / "dataset" / "config.yaml")
            cfg = OmegaConf.to_container(cfg, resolve=True)
            self.num_spk = cfg.get("num_spk", num_spk)
            self.frame_dur = cfg.get("frame_dur", frame_dur)
            splits = cfg["splits"]
            if split not in splits:
                raise ValueError(f"Unknown split '{split}'. Known: {sorted(splits)}")
            cuts = splits[split]
        cuts_path = Path(cuts)
        if not cuts_path.is_absolute():
            cuts_path = recipe_root / cuts_path
        if not cuts_path.is_file():
            raise FileNotFoundError(
                f"CutSet for split '{split}' not found: {cuts_path}. "
                "Run the `data_preparation` stage first."
            )
        self.cuts = lhotse.load_manifest(str(cuts_path))
        self.ids = list(self.cuts.ids)

    def __len__(self) -> int:
        return len(self.ids)

    def _build_labels(self, cut, num_frames: int) -> np.ndarray:
        """Build the ``(num_frames, num_spk)`` activity matrix for a cut.

        Keeps the ``num_spk`` most-talkative speakers and marks each frame
        overlapping a kept supervision as active for that speaker's column.
        """
        # Rank speakers by total speaking time; keep the top `num_spk`.
        durs: Dict[str, float] = {}
        for s in cut.supervisions:
            durs[s.speaker] = durs.get(s.speaker, 0.0) + float(s.duration)
        speakers = sorted(durs, key=lambda k: durs[k], reverse=True)[: self.num_spk]
        spk_idx = {spk: i for i, spk in enumerate(speakers)}

        labels = np.zeros((num_frames, self.num_spk), dtype=np.float32)
        for s in cut.supervisions:
            if s.speaker not in spk_idx:
                continue
            st = max(0, int(math.floor(s.start / self.frame_dur)))
            en = min(
                num_frames, int(math.ceil((s.start + s.duration) / self.frame_dur))
            )
            if en > st:
                labels[st:en, spk_idx[s.speaker]] = 1.0
        return labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return the ``{speech, spk_labels, utt_id}`` item for index ``idx``."""
        cut = self.cuts[self.ids[int(idx)]]
        audio = cut.load_audio()  # (C, N)
        wav = np.asarray(audio[self.channel], dtype=np.float32)
        num_frames = num_frames_from_samples(wav.shape[0], hop=self.hop)
        labels = self._build_labels(cut, num_frames)
        return {
            "speech": wav,
            "spk_labels": labels,
            "utt_id": cut.id,
        }
