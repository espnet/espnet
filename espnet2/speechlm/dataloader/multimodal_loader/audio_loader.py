#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Audio data loading utilities using Lhotse library for efficient audio processing."""

from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pyarrow as pa

try:
    import kaldiio
except ImportError:
    kaldiio = None

try:
    from arkive import audio_read
except ImportError:
    raise ImportError(
        "arkive is not installed. Install at https://github.com/wanchichen/arkive"
    )

try:
    import duckdb
except ImportError:
    raise ImportError(
        "duckdb is not installed. Please install it with: pip install duckdb"
    )

try:
    from lhotse import CutSet, RecordingSet
except ImportError:
    raise ImportError(
        "lhotse is not installed. Please install it with: pip install lhotse"
    )


class ArkiveAudioReader:
    """Dict-like lazy audio reader using arkive parquets.

    Reads audio data from arkive parquet files. Audio is accessed via byte
    offsets and time boundaries stored in the parquet metadata.

    Returns:
        Tuple of (audio_array, sample_rate) where audio_array has shape
        [num_samples, num_channels].

    Args:
        parquet_path: Path to the parquet file containing audio metadata.
        valid_ids: List of valid IDs to keep (optional, keeps all if None).
    """

    def __init__(
        self,
        parquet_path: str,
        valid_ids: list = None,
    ):
        # Use a dedicated connection to avoid polluting global state
        conn = duckdb.connect()

        # Register valid_ids as a table for memory-efficient filtering
        # Avoids building a huge SQL string with millions of IDs
        if valid_ids is not None:
            # Create PyArrow table and register - more memory efficient
            valid_ids_table = pa.table({"utt_id": pa.array(valid_ids)})
            conn.register("valid_ids_tbl", valid_ids_table)

            # Use semi-join for efficient filtering
            result = conn.execute(
                f"""
                SELECT p.* FROM read_parquet('{parquet_path}') p
                SEMI JOIN valid_ids_tbl v ON p.utt_id = v.utt_id
                """
            )
        else:
            result = conn.execute(f"SELECT * FROM read_parquet('{parquet_path}')")

        self.data = result.pl()

        # Build index without calling to_list() to avoid extra memory copy
        self.index = {
            utt_id: idx for idx, utt_id in enumerate(self.data["utt_id"].to_physical())
        }

        # Clean up connection
        conn.close()

    def __getitem__(self, key: str) -> Tuple[np.ndarray, int]:
        """Get audio by ID. Returns (audio_array, sample_rate)."""
        idx = self.index[key]
        row = self.data.row(idx, named=True)

        start_time = row.get("start_time", None)
        end_time = row.get("end_time", None)

        data = audio_read(
            row["path"],
            start_offset=row["start_byte_offset"],
            file_size=row["file_size_bytes"],
            start_time=start_time,
            end_time=end_time,
        )

        return data.array.T, data.sample_rate

    def __contains__(self, key: str) -> bool:
        """Check if ID exists in manifest."""
        return key in self.index

    def __len__(self) -> int:
        """Return number of items in manifest."""
        return len(self.data)

    def keys(self) -> Iterator[str]:
        """Return iterator over IDs."""
        return iter(self.index.keys())

    def values(self) -> Iterator[Tuple[np.ndarray, int]]:
        """Return iterator over (audio_array, sample_rate) tuples."""
        for key in self.index:
            yield self[key]

    def items(self) -> Iterator[Tuple[str, Tuple[np.ndarray, int]]]:
        """Return iterator over (id, (audio_array, sample_rate)) pairs."""
        for key in self.index:
            yield key, self[key]


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


class KaldiAudioReader:
    """Dict-like lazy audio reader using Kaldi ark files via kaldiio.

    Reads audio data from Kaldi ark files using an index file. The index file
    should contain one entry per line in the format: "example_id ark_path:offset"

    Returns:
        Tuple of (audio_array, sample_rate) where audio_array has shape
        [num_samples,] or [num_samples, num_channels].

    Args:
        index_path: Path to the index file containing "example_id ark:offset"
            per line.
        valid_ids: List of valid IDs to keep (optional, keeps all if None).
    """

    def __init__(
        self,
        index_path: str,
        valid_ids: list = None,
    ):
        if kaldiio is None:
            raise ImportError(
                "kaldiio is not installed. "
                "Please install it with: pip install kaldiio"
            )

        self.index = {}

        valid_ids_set = set(valid_ids) if valid_ids is not None else None

        with open(index_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid index line format: '{line}'. "
                        "Expected 'example_id ark_path:offset'"
                    )

                example_id, ark_index = parts

                # Filter by valid_ids if provided
                if valid_ids_set is not None and example_id not in valid_ids_set:
                    continue

                self.index[example_id] = ark_index

    def __getitem__(self, key: str) -> Tuple[np.ndarray, int]:
        """Get audio by ID. Returns (audio_array, sample_rate).

        Returns:
            Tuple of (audio_array, sample_rate) where audio_array has shape
            [num_channels, num_samples]. For single-channel audio, shape will
            be [1, num_samples].
        """
        if key not in self.index:
            raise KeyError(f"Key '{key}' not found in index")

        ark_index = self.index[key]
        sample_rate, audio = kaldiio.load_mat(ark_index)

        # Ensure consistent shape [num_channels, num_samples]
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]  # Shape: [1, num_samples]

        return audio, sample_rate

    def __contains__(self, key: str) -> bool:
        """Check if ID exists in index."""
        return key in self.index

    def __len__(self) -> int:
        """Return number of items in index."""
        return len(self.index)

    def keys(self) -> Iterator[str]:
        """Return iterator over IDs."""
        return iter(self.index.keys())

    def values(self) -> Iterator[Tuple[np.ndarray, int]]:
        """Return iterator over (audio_array, sample_rate) tuples."""
        for key in self.index:
            yield self[key]

    def items(self) -> Iterator[Tuple[str, Tuple[np.ndarray, int]]]:
        """Return iterator over (id, (audio_array, sample_rate)) pairs."""
        for key in self.index:
            yield key, self[key]
