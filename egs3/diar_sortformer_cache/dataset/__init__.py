"""Lhotse-backed diarization dataset module for the Sortformer recipe."""

from egs3.librispeech_sortformer.diar.dataset.dataset import (  # noqa: F401
    LhotseDiarDataset,
    num_frames_from_samples,
)

# ESPnet3 loads a class named ``Dataset`` from the recipe dataset module.
Dataset = LhotseDiarDataset
