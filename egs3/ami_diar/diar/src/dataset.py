"""Lhotse-based diarization dataset for ESPnet3.

This dataset loads audio from lhotse manifests and generates frame-level
speaker labels from RTTM annotations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

try:
    from lhotse import CutSet, Recording, SupervisionSegment
    from lhotse.cut import Cut
except ImportError:
    raise ImportError(
        "lhotse is not installed. Please install it with: pip install lhotse"
    )


class DiarizationDataset(TorchDataset):
    """Diarization dataset using lhotse for data loading.

    This dataset:
    1. Loads audio from lhotse CutSet manifests
    2. Generates frame-level speaker labels from RTTM annotations
    3. Handles chunking for long recordings
    4. Supports variable number of speakers per recording

    Args:
        manifest_path: Path to lhotse CutSet manifest (cuts.jsonl.gz)
        chunk_duration: Duration of chunks in seconds (None for whole recording)
        chunk_shift: Shift between chunks in seconds
        frame_shift: Frame shift for labels in seconds (default: 0.02 = 50 Hz)
        max_speakers: Maximum number of speakers to handle
        min_chunk_duration: Minimum chunk duration (skip shorter chunks)
        sample_rate: Target sample rate (default: 16000)
    """

    def __init__(
        self,
        manifest_path: str,
        chunk_duration: Optional[float] = 8.0,
        chunk_shift: Optional[float] = 6.0,
        frame_shift: float = 0.02,
        max_speakers: int = 4,
        min_chunk_duration: float = 0.5,
        sample_rate: int = 16000,
    ):
        super().__init__()

        self.manifest_path = Path(manifest_path)
        self.chunk_duration = chunk_duration
        self.chunk_shift = chunk_shift if chunk_shift else chunk_duration
        self.frame_shift = frame_shift
        self.max_speakers = max_speakers
        self.min_chunk_duration = min_chunk_duration
        self.sample_rate = sample_rate

        # Load cuts from manifest
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        logging.info(f"Loading diarization manifest from {manifest_path}")
        # Load eagerly to support indexing by ID
        self.cuts = CutSet.from_file(manifest_path).to_eager()
        logging.info(f"Loaded {len(self.cuts)} cuts")

        # Generate chunk indices for all cuts
        self.chunks = self._generate_chunks()
        logging.info(f"Generated {len(self.chunks)} chunks")

    def _generate_chunks(self) -> List[Tuple[str, float, float]]:
        """Generate list of (cut_id, start_time, end_time) for all chunks.

        Returns:
            List of tuples (cut_id, start, end) for each chunk
        """
        chunks = []

        for cut in self.cuts:
            duration = cut.duration

            if self.chunk_duration is None:
                # Use whole recording
                chunks.append((cut.id, 0.0, duration))
            else:
                # Generate overlapping chunks
                start = 0.0
                while start < duration:
                    end = min(start + self.chunk_duration, duration)

                    # Skip if chunk is too short
                    if end - start >= self.min_chunk_duration:
                        chunks.append((cut.id, start, end))

                    # Move to next chunk
                    start += self.chunk_shift

                    # Break if we've covered the whole recording
                    if end >= duration:
                        break

        return chunks

    def _generate_frame_labels(
        self,
        cut: Cut,
        start_time: float,
        end_time: float,
    ) -> np.ndarray:
        """Generate frame-level speaker labels from RTTM annotations.

        Args:
            cut: Lhotse cut with supervisions (speaker segments)
            start_time: Chunk start time
            end_time: Chunk end time

        Returns:
            labels: Frame-level speaker labels
                Shape: (num_frames, max_speakers)
                Binary values indicating speaker activity
        """
        # Calculate number of frames
        duration = end_time - start_time
        num_frames = int(np.ceil(duration / self.frame_shift))

        # Initialize labels (all zeros = no speaker)
        labels = np.zeros((num_frames, self.max_speakers), dtype=np.float32)

        # Get speaker segments that overlap with this chunk
        if cut.supervisions:
            # Map speaker IDs to indices
            speaker_ids = sorted(set(sup.speaker for sup in cut.supervisions if sup.speaker))

            # Limit to max_speakers (keep most active speakers)
            if len(speaker_ids) > self.max_speakers:
                # Count total duration per speaker
                speaker_durations = {}
                for speaker_id in speaker_ids:
                    duration = sum(
                        sup.duration
                        for sup in cut.supervisions
                        if sup.speaker == speaker_id
                    )
                    speaker_durations[speaker_id] = duration

                # Keep top max_speakers by duration
                speaker_ids = sorted(
                    speaker_durations.keys(),
                    key=lambda x: speaker_durations[x],
                    reverse=True
                )[:self.max_speakers]

            speaker_to_idx = {spk: idx for idx, spk in enumerate(speaker_ids)}

            # Fill in labels
            for sup in cut.supervisions:
                if sup.speaker not in speaker_to_idx:
                    continue

                speaker_idx = speaker_to_idx[sup.speaker]

                # Get segment times relative to cut
                seg_start = sup.start
                seg_end = sup.start + sup.duration

                # Adjust to chunk coordinates
                seg_start_chunk = seg_start - start_time
                seg_end_chunk = seg_end - start_time

                # Skip if segment doesn't overlap with chunk
                if seg_end_chunk <= 0 or seg_start_chunk >= duration:
                    continue

                # Clip to chunk boundaries
                seg_start_chunk = max(0, seg_start_chunk)
                seg_end_chunk = min(duration, seg_end_chunk)

                # Convert to frame indices
                start_frame = int(seg_start_chunk / self.frame_shift)
                end_frame = int(np.ceil(seg_end_chunk / self.frame_shift))

                # Clip to valid frame range
                start_frame = max(0, start_frame)
                end_frame = min(num_frames, end_frame)

                # Set label to 1 for this speaker in these frames
                if start_frame < end_frame:
                    labels[start_frame:end_frame, speaker_idx] = 1.0

        return labels

    def __len__(self) -> int:
        """Return number of chunks."""
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a chunk with audio and labels.

        Args:
            idx: Chunk index

        Returns:
            Dictionary with:
                - speech: Audio waveform (1D tensor)
                - labels: Frame-level speaker labels (2D tensor)
                - utt_id: Utterance ID (string)
        """
        cut_id, start_time, end_time = self.chunks[idx]

        # Get the cut
        cut = self.cuts[cut_id]

        # Truncate cut to the chunk
        if self.chunk_duration is not None:
            cut = cut.truncate(
                offset=start_time,
                duration=end_time - start_time,
                preserve_id=False,
            )

        # Load audio
        audio = cut.load_audio()  # Shape: (num_samples,) for mono

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        # Resample if needed
        if cut.sampling_rate != self.sample_rate:
            from lhotse.audio import resample_audio
            audio = resample_audio(
                audio,
                from_sr=cut.sampling_rate,
                to_sr=self.sample_rate,
            )

        # Generate frame-level labels
        labels = self._generate_frame_labels(cut, start_time, end_time)

        # Create utterance ID
        utt_id = f"{cut_id}_{int(start_time*100):06d}_{int(end_time*100):06d}"

        return {
            "speech": torch.from_numpy(audio).float(),
            "labels": torch.from_numpy(labels).float(),
            "utt_id": utt_id,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for batching diarization data.

    Handles variable-length audio and labels by padding.

    Args:
        batch: List of dictionaries from __getitem__

    Returns:
        Batched dictionary with:
            - speech: Padded audio (batch, max_audio_len)
            - speech_lengths: Audio lengths (batch,)
            - labels: Padded labels (batch, max_frames, num_speakers)
            - labels_lengths: Label lengths (batch,)
            - utt_ids: List of utterance IDs
    """
    # Get batch size
    batch_size = len(batch)

    # Get max lengths
    max_audio_len = max(item["speech"].shape[0] for item in batch)
    max_frames = max(item["labels"].shape[0] for item in batch)
    num_speakers = batch[0]["labels"].shape[1]

    # Initialize padded tensors
    speech = torch.zeros(batch_size, max_audio_len)
    speech_lengths = torch.zeros(batch_size, dtype=torch.long)
    labels = torch.zeros(batch_size, max_frames, num_speakers)
    labels_lengths = torch.zeros(batch_size, dtype=torch.long)
    utt_ids = []

    # Fill in data
    for i, item in enumerate(batch):
        audio_len = item["speech"].shape[0]
        label_len = item["labels"].shape[0]

        speech[i, :audio_len] = item["speech"]
        speech_lengths[i] = audio_len

        labels[i, :label_len, :] = item["labels"]
        labels_lengths[i] = label_len

        utt_ids.append(item["utt_id"])

    return {
        "speech": speech,
        "speech_lengths": speech_lengths,
        "labels": labels,
        "labels_lengths": labels_lengths,
        "utt_ids": utt_ids,
    }


if __name__ == "__main__":
    # Test the dataset
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset.py <manifest_path>")
        print("Example: python dataset.py /path/to/cuts.jsonl.gz")
        sys.exit(1)

    manifest_path = sys.argv[1]

    # Create dataset
    dataset = DiarizationDataset(
        manifest_path=manifest_path,
        chunk_duration=8.0,
        chunk_shift=6.0,
        max_speakers=4,
    )

    print(f"Dataset size: {len(dataset)}")

    # Get first sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Speech shape: {sample['speech'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print(f"Utterance ID: {sample['utt_id']}")

    # Test collate function
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collate_fn,
        shuffle=False,
    )

    batch = next(iter(loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Batch speech shape: {batch['speech'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
    print(f"Speech lengths: {batch['speech_lengths']}")
    print(f"Labels lengths: {batch['labels_lengths']}")
