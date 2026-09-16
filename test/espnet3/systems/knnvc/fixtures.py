"""Lightweight stand-ins for the audio dataset and the WavLM encoder.

Referenced by dotted path (``data_src`` / ``_target_``) from the VC system
tests so that no WavLM checkpoint has to be downloaded.
"""

from __future__ import annotations

import numpy as np
import torch

SAMPLE_RATE = 16000
HOP_LENGTH = 320
FEATURE_DIM = 8

# (utt_id, speaker, seconds)
UTTERANCES = [
    ("spkA-0001", "spkA", 0.5),
    ("spkA-0002", "spkA", 0.7),
    ("spkA-0003", "spkA", 0.4),
    ("spkB-0001", "spkB", 0.6),
    ("spkB-0002", "spkB", 0.5),
    ("spkC-0001", "spkC", 0.3),  # single-utterance speaker: nothing to prematch
]


class TinyAudioDataset:
    """Synthetic audio dataset implementing the ``prepare_features`` contract."""

    def __init__(self, split: str = "train", **_kwargs) -> None:
        self.split = split
        self.utterances = list(UTTERANCES)

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> dict:
        utt_id, _speaker, seconds = self.utterances[int(idx)]
        rng = np.random.RandomState(abs(hash(utt_id)) % (2**31))
        return {"speech": rng.randn(int(seconds * SAMPLE_RATE)).astype(np.float32)}

    def get_pool_key(self, idx: int) -> str:
        return self.utterances[int(idx)][1]

    def get_feature_name(self, idx: int) -> str:
        utt_id, speaker, _ = self.utterances[int(idx)]
        return f"{self.split}/{speaker}/{utt_id}"


class DatasetWithoutContract:
    """Dataset lacking ``get_pool_key`` / ``get_feature_name``."""

    def __init__(self, **_kwargs) -> None:
        pass

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict:
        return {"speech": np.zeros(SAMPLE_RATE, dtype=np.float32)}


# Recipe-module style aliases so `data_src: <this module>` works.
Dataset = TinyAudioDataset


class DummyEncoder(torch.nn.Module):
    """Deterministic frame encoder: mean-pools hop-sized windows into 8 dims."""

    sample_rate = SAMPLE_RATE
    hop_length = HOP_LENGTH

    def __init__(self, device: str = "cpu", **_kwargs) -> None:
        super().__init__()
        self.device = torch.device(device)
        generator = torch.Generator().manual_seed(0)
        self.projection = torch.nn.Parameter(
            torch.randn(HOP_LENGTH, FEATURE_DIM, generator=generator),
            requires_grad=False,
        )

    @property
    def output_dim(self) -> int:
        return FEATURE_DIM

    @torch.inference_mode()
    def encode(self, speech, pad_to_hop: bool = False) -> torch.Tensor:
        if isinstance(speech, np.ndarray):
            speech = torch.from_numpy(speech)
        speech = speech.to(self.device, dtype=torch.float32).flatten()
        if pad_to_hop:
            remainder = speech.numel() % HOP_LENGTH
            speech = torch.nn.functional.pad(speech, (0, HOP_LENGTH - remainder))
        n_frames = speech.numel() // HOP_LENGTH
        frames = speech[: n_frames * HOP_LENGTH].view(n_frames, HOP_LENGTH)
        return frames @ self.projection.to(self.device)

    def forward(self, speech):
        return self.encode(speech)
