#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Audio data loading utilities using Lhotse library for efficient audio processing."""

from pathlib import Path
from typing import Tuple

import numpy as np

try:
    from lhotse import CutSet, RecordingSet
except ImportError:
    raise ImportError(
        "lhotse is not installed. Please install it with: pip install lhotse"
    )


class LhotseAudioReader:
    """Dict-like lazy audio reader using Lhotse manifests.

    This reader supports both single-channel and multi-channel audio data:
    - Single-channel audio (MonoCut): Returns shape [1, num_samples]
    - Multi-channel audio (MultiCut): Returns shape [num_channels, num_samples]

    The output shape is consistent regardless of the input type, always returning
    a 2D array with shape [num_channels, num_samples].

    Args:
        manifest_dir: Directory containing Lhotse manifest files
            (recordings.jsonl.gz and optionally cuts.jsonl.gz)
        valid_ids: List of valid IDs to keep (optional, keeps all if None)
    """

    def __init__(self, manifest_dir: str, valid_ids: list = None):
        manifest_path = Path(manifest_dir)
        cuts_path = manifest_path / "cuts.jsonl.gz"
        recordings_path = manifest_path / "recordings.jsonl.gz"

        # Prefer cuts over recordings if available
        if cuts_path.exists():
            full_manifest = CutSet.from_file(cuts_path)
        elif recordings_path.exists():
            full_manifest = RecordingSet.from_file(recordings_path)
        else:
            raise FileNotFoundError(f"No manifest files found in {manifest_dir}")

        # Filter manifest by valid_ids
        if valid_ids is not None:
            valid_ids_set = set(valid_ids)
            selected_items = [
                item for item in full_manifest if item.id in valid_ids_set
            ]
        else:
            selected_items = list(full_manifest)

        # Create new manifest with only selected items
        if isinstance(full_manifest, CutSet):
            self.manifest = CutSet.from_cuts(selected_items)
        else:
            self.manifest = RecordingSet.from_recordings(selected_items)

    def __getitem__(self, key: str) -> Tuple[np.ndarray, int]:
        """Get audio data by ID.

        Returns:
            Tuple of (audio_array, sample_rate) where audio_array has shape
            [num_channels, num_samples]. For single-channel audio, shape will be
            [1, num_samples].
        """
        item = self.manifest[key]
        audio = item.load_audio()
        sample_rate = item.sampling_rate

        # Ensure consistent shape [num_channels, num_samples]
        # MonoCut.load_audio() returns 1D array, MultiCut returns 2D array
        if audio.ndim == 1:
            # Single-channel audio (MonoCut) - add channel dimension
            audio = audio[np.newaxis, :]  # Shape: [1, num_samples]
        elif audio.ndim == 2:
            # Multi-channel audio (MultiCut) - already has correct shape
            pass  # Shape: [num_channels, num_samples]
        else:
            raise ValueError(f"Unexpected audio shape: {audio.shape} for item {key}")

        return audio, sample_rate

    def __contains__(self, key: str) -> bool:
        """Check if ID exists in manifest."""
        return key in self.manifest

    def __len__(self) -> int:
        """Return number of items in manifest."""
        return len(self.manifest)

    def keys(self):
        """Return iterator over IDs."""
        return self.manifest.ids

    def values(self):
        """Return iterator over items."""
        return iter(self.manifest)

    def items(self):
        """Return iterator over (id, item) pairs."""
        for item in self.manifest:
            yield item.id, item
