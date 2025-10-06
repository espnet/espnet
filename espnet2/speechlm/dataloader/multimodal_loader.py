#!/usr/bin/env python3
import json
import logging
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


class TextReader:
    """Dict-like text reader supporting plain and JSONL formats.

    Plain format: <id> <text content>
    JSONL format: {"id": "<id>", "text": "<text content>"}

    Format is determined by file suffix (.jsonl for JSONL, otherwise plain).

    Args:
        text_file: Path to text file (plain or JSONL format)
        valid_ids: List of valid IDs to keep (optional, keeps all if None)
    """

    def __init__(self, text_file: str, valid_ids: list = None):
        self.data = {}
        text_path = Path(text_file)

        # Determine format by file suffix
        is_jsonl = text_path.suffix == ".jsonl"

        # Convert valid_ids to set for faster lookup
        valid_ids_set = set(valid_ids) if valid_ids is not None else None

        # Load and filter lines
        with open(text_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                if is_jsonl:
                    item = json.loads(line)
                    if "id" not in item or "text" not in item:
                        logging.warning(
                            f"Skipping line {line_idx}: missing 'id' or 'text' key"
                        )
                        continue
                    example_id = item["id"]
                    content = item["text"]
                else:
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        example_id, content = parts
                    else:
                        continue

                # Only keep if in valid_ids (or if valid_ids is None)
                if valid_ids_set is None or example_id in valid_ids_set:
                    self.data[example_id] = content

    def __getitem__(self, key: str) -> str:
        """Get text by ID."""
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        """Check if ID exists."""
        return key in self.data

    def __len__(self) -> int:
        """Return number of items."""
        return len(self.data)

    def keys(self):
        """Return iterator over IDs."""
        return self.data.keys()

    def values(self):
        """Return iterator over texts."""
        return self.data.values()

    def items(self):
        """Return iterator over (id, text) pairs."""
        return self.data.items()
