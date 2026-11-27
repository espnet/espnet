"""Enhance recordings with a model exported by the SNOAR system.

``SNOAREnhancer`` is the inference backend: it loads an exported model
directory (``config.yaml``, ``model.pth``, ``snoar.yaml``), reads its outputs
with the stored ``readout``, and processes recordings of any length in
overlapping chunks. ``SNOARInferenceRunner`` adapts
``SNOAREnhancer`` to the shared runner.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from espnet3.systems.base.inference_runner import InferenceRunner
from espnet3.systems.snoar.src.net import load_net, resolve_model_dir
from espnet3.systems.snoar.src.readout import read_speech

logger = logging.getLogger(__name__)


class SNOAREnhancer:
    """Speech enhancer backed by an exported SNOAR/DNF/MixIT/supervised model.

    Args:
        model_dir: Directory written by the ``export`` stage or by
            ``espnet snoar adapt``, or an ESPnet2 enhancement model tag.
        readout: Override the stored readout (``dnf``, ``mixit``, ``raw``).
            ESPnet2 models without ``snoar.yaml`` default to ``raw``.
        device: Torch device, or ``auto`` (CUDA when available).
        chunk_seconds: Longest piece processed at once. Longer recordings
            are split into chunks with 50 % overlap and joined with a Hann
            cross-fade; each chunk is gain-aligned to its input first, so
            the free output gain of an SI-SDR-trained model does not jump
            at chunk borders. ``None`` processes whole recordings.
        fs: Sampling rate of the audio, passed to the network's encoder.
        tau_rel: Guard of the DNF projection.

    Examples:
        >>> enhancer = SNOAREnhancer("exp/snoar/export")  # doctest: +SKIP
        >>> clean = enhancer(noisy)  # doctest: +SKIP
    """

    def __init__(
        self,
        model_dir: str,
        readout: Optional[str] = None,
        device: str = "cpu",
        chunk_seconds: Optional[float] = 20.0,
        fs: int = 16000,
        tau_rel: float = 1e-3,
    ) -> None:
        """Load the model."""
        model_dir = resolve_model_dir(model_dir)
        meta = {}
        sidecar = Path(model_dir) / "snoar.yaml"
        if sidecar.is_file():
            with open(sidecar, encoding="utf-8") as f:
                meta = yaml.safe_load(f) or {}
        self.readout = readout or meta.get("readout", "raw")
        self.fs = int(meta.get("fs", fs))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = load_net(model_dir, fs=self.fs).to(device).eval()
        self.device = device
        self.chunk = int(chunk_seconds * self.fs) if chunk_seconds else None
        self.tau_rel = tau_rel

    @torch.no_grad()
    def _enhance_piece(self, x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]
        t = t.to(self.device)
        out = self.net(t).float()
        return read_speech(out, t, self.readout, self.tau_rel)[0].cpu().numpy()

    def __call__(self, speech: np.ndarray) -> np.ndarray:
        """Return the enhanced version of one mono recording.

        Args:
            speech: Samples, shape ``(T,)``.

        Returns:
            Enhanced samples, shape ``(T,)``, float32.

        Raises:
            ValueError: If ``speech`` is not one-dimensional.
        """
        x = np.asarray(speech, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError(f"expected mono samples (T,), got shape {x.shape}")
        if self.chunk is None or len(x) <= self.chunk:
            return self._enhance_piece(x).astype(np.float32)
        hop = self.chunk // 2
        win = np.hanning(self.chunk + 2)[1:-1].astype(np.float32)
        out = np.zeros(len(x), dtype=np.float32)
        norm = np.zeros(len(x), dtype=np.float32)
        starts = list(range(0, len(x) - self.chunk + 1, hop))
        if starts[-1] + self.chunk < len(x):
            starts.append(len(x) - self.chunk)
        for i, st in enumerate(starts):
            w = win.copy()
            # Edges of the recording get full weight, not a fade-in from zero.
            if i == 0:
                w[: self.chunk // 2] = 1.0
            if i == len(starts) - 1:
                w[self.chunk // 2 :] = 1.0
            out[st : st + self.chunk] += w * self._enhance_piece(
                x[st : st + self.chunk]
            )
            norm[st : st + self.chunk] += w
        return out / np.maximum(norm, 1e-8)


def _read_sidecar(model_dir: str) -> dict:
    sidecar = Path(model_dir) / "snoar.yaml"
    if not sidecar.is_file():
        return {}
    with open(sidecar, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class SNOARInferenceRunner(InferenceRunner):
    """Pass checked audio to :class:`SNOAREnhancer` and name its output."""

    @staticmethod
    def prepare_model_inputs(inputs):
        """Replace the ``Audio`` of ``speech`` by its mono samples.

        Raises:
            ValueError: If ``speech`` is a batch (a list of recordings).
        """
        audio = inputs["speech"]
        if isinstance(audio, (list, tuple)):
            raise ValueError(
                "SNOAREnhancer runs one recording at a time: call the model on "
                "each recording instead of forward_batch (batch_size: null)"
            )
        return {**inputs, "speech": audio.mono().array}

    @staticmethod
    def format_output(result, *, batched):
        """Return ``{"enhanced": samples}``.

        Raises:
            ValueError: If the call was a batch.
        """
        if batched:
            raise ValueError(
                "SNOAREnhancer runs one recording at a time: batch_size: null"
            )
        return {"enhanced": np.asarray(result, dtype=np.float32)}
