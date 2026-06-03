#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Script for preparing audio metadata from Kaldi format using Lhotse."""

import argparse
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    from lhotse import CutSet, MonoCut, MultiCut, Recording, RecordingSet
except ImportError:
    raise ImportError(
        "lhotse is not installed. Please install it with: pip install lhotse"
    )


def process_recording(recording_id: str, audio_path: str) -> Optional[Recording]:
    """Process a single recording to extract metadata."""
    try:
        recording = Recording.from_file(
            path=audio_path,
            recording_id=recording_id,
        )
        return recording
    except Exception as e:
        logging.error(f"Error processing {recording_id}: {e}")
        return None


def create_cut_from_recording(
    cut_id: str,
    recording: Recording,
    start: float,
    duration: float,
) -> Union[MonoCut, MultiCut]:
    """Create a cut (MonoCut or MultiCut) from a recording.

    Args:
        cut_id: ID for the cut
        recording: Recording object
        start: Start time in seconds
        duration: Duration in seconds

    Returns:
        MonoCut for single-channel or MultiCut for multi-channel recordings
    """
    if recording.num_channels == 1:
        # Single channel: create MonoCut
        return MonoCut(
            id=cut_id,
            start=start,
            duration=duration,
            channel=0,
            recording=recording,
        )
    else:
        # Multi-channel: create MonoCuts for each channel and combine
        mono_cuts = []
        for channel in range(recording.num_channels):
            mono_cut = MonoCut(
                id=f"{cut_id}_c{channel}",
                start=start,
                duration=duration,
                channel=channel,
                recording=recording,
            )
            mono_cuts.append(mono_cut)
        # Create MultiCut from all channels
        multi_cut = MultiCut.from_mono(*mono_cuts)
        # Override the ID to use the original cut_id
        multi_cut.id = cut_id
        return multi_cut


def parse_segments_file(
    segments_path: str, recording_dict: Dict[str, Recording]
) -> List[Tuple[str, Recording, float, float]]:
    """Parse and validate segments file.

    Args:
        segments_path: Path to segments file
        recording_dict: Dictionary mapping recording IDs to Recording objects

    Returns:
        List of validated (segment_id, recording, start, duration) tuples
    """
    segments = []
    invalid_count = 0

    with open(segments_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 4:
                logging.warning(f"Line {line_num}: Invalid format in segments: {line}")
                invalid_count += 1
                continue

            segment_id, recording_id, start_str, end_str = parts

            # Parse times
            try:
                start = float(start_str)
                end = float(end_str)
            except ValueError:
                logging.warning(
                    f"Line {line_num}: Invalid time values: {start_str}, {end_str}"
                )
                invalid_count += 1
                continue

            # Validate times
            if start < 0:
                logging.warning(
                    f"Line {line_num}: Negative start time for {segment_id}: {start}"
                )
                invalid_count += 1
                continue

            if end <= start:
                logging.warning(
                    f"Line {line_num}: End time <= start time for {segment_id}: "
                    f"{start} -> {end}"
                )
                invalid_count += 1
                continue

            # Check if recording exists
            if recording_id not in recording_dict:
                logging.warning(
                    f"Line {line_num}: Recording {recording_id} not found for "
                    f"segment {segment_id}"
                )
                invalid_count += 1
                continue

            recording = recording_dict[recording_id]
            duration = end - start

            # Validate against recording duration
            if end > recording.duration:
                logging.warning(
                    f"Line {line_num}: Segment {segment_id} exceeds recording duration "
                    f"({end:.2f}s > {recording.duration:.2f}s)"
                )
                # Optionally truncate instead of skipping
                duration = min(recording.duration - start, duration)
                if duration <= 0:
                    invalid_count += 1
                    continue

            segments.append((segment_id, recording, start, duration))

    if invalid_count > 0:
        logging.warning(f"Skipped {invalid_count} invalid segments")

    return segments


def print_statistics(
    cut_set: CutSet, recording_set: RecordingSet, multi_channel_count: int
):
    """Print summary statistics for the processed data."""
    # Cut statistics
    total_duration = sum(cut.duration for cut in cut_set)
    logging.info(f"Total cuts: {len(cut_set)}")
    logging.info(
        f"Total audio duration: {total_duration:.2f} seconds "
        f"({total_duration / 3600:.2f} hours)"
    )

    if multi_channel_count > 0:
        logging.info(f"Multi-channel cuts: {multi_channel_count}")

    # Recording statistics
    logging.info(f"Total recordings: {len(recording_set)}")

    # Sampling rate distribution
    sr_counts = Counter(rec.sampling_rate for rec in recording_set)
    if len(sr_counts) > 1:
        logging.info("Sampling rate distribution:")
        for sr, count in sorted(sr_counts.items()):
            logging.info(f"  {sr} Hz: {count} recordings")
    else:
        logging.info(f"Sampling rate: {list(sr_counts.keys())[0]} Hz (all recordings)")

    # Channel count distribution
    channel_counts = Counter(rec.num_channels for rec in recording_set)
    if len(channel_counts) > 1:
        logging.info("Channel count distribution:")
        for num_channels, count in sorted(channel_counts.items()):
            logging.info(f"  {num_channels} channel(s): {count} recordings")
    else:
        num_ch = list(channel_counts.keys())[0]
        logging.info(f"Channels: {num_ch} channel(s) (all recordings)")


def prepare_audio_lhotse(
    wav_scp: str,
    segments: Optional[str],
    output_dir: str,
    num_jobs: int,
    log_level: str,
):
    """Process Kaldi wav.scp and segments to create Lhotse manifests.

    Args:
        wav_scp: Path to Kaldi wav.scp file
        segments: Path to Kaldi segments file (optional)
        output_dir: Directory to save Lhotse manifest files
        num_jobs: Number of parallel jobs for processing
        log_level: Logging level
    """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Reading wav.scp from: {wav_scp}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Number of jobs: {num_jobs}")

    # Read Kaldi wav.scp file
    audio_paths = {}
    with open(wav_scp, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                logging.warning(f"Line {line_num}: Invalid format in wav.scp: {line}")
                continue
            recording_id, path = parts
            audio_paths[recording_id] = path

    if not audio_paths:
        raise ValueError("No valid recordings found in wav.scp")

    logging.info(f"Found {len(audio_paths)} recordings in wav.scp")

    # Process recordings in parallel
    recordings = []
    logging.info("Processing recordings with multiprocessing...")

    with ProcessPoolExecutor(max_workers=num_jobs) as executor:
        futures = [
            executor.submit(process_recording, rid, path)
            for rid, path in audio_paths.items()
        ]

        for i, future in enumerate(as_completed(futures), 1):
            recording = future.result()
            if recording is not None:
                recordings.append(recording)

            if i % 10000 == 0:
                logging.info(f"Processed {i}/{len(audio_paths)} recordings")

    if not recordings:
        raise ValueError("No recordings could be processed successfully")

    logging.info(
        f"Successfully processed {len(recordings)}/{len(audio_paths)} recordings"
    )

    # Create RecordingSet
    recording_set = RecordingSet.from_recordings(recordings)
    recording_dict = {rec.id: rec for rec in recording_set}

    # Create cuts based on whether segments are provided
    cuts = []
    multi_channel_count = 0

    if segments is not None:
        logging.info(f"Reading segments from: {segments}")

        # Parse and validate segments
        segment_data = parse_segments_file(segments, recording_dict)

        if not segment_data:
            raise ValueError("No valid segments found in segments file")

        logging.info(f"Processing {len(segment_data)} valid segments")

        # Create cuts from segments
        for segment_id, recording, start, duration in segment_data:
            cut = create_cut_from_recording(segment_id, recording, start, duration)
            cuts.append(cut)
            if isinstance(cut, MultiCut):
                multi_channel_count += 1

    else:
        logging.info("No segments file provided - creating cuts from whole recordings")

        # Create cuts from full recordings
        for recording in recording_set:
            cut = create_cut_from_recording(
                recording.id, recording, start=0, duration=recording.duration
            )
            cuts.append(cut)
            if isinstance(cut, MultiCut):
                multi_channel_count += 1

    if not cuts:
        raise ValueError("No cuts could be created")

    # Create and save CutSet
    cut_set = CutSet.from_cuts(cuts)
    cut_manifest_path = output_path / "cuts.jsonl.gz"
    cut_set.to_file(cut_manifest_path)
    logging.info(f"Saved CutSet to: {cut_manifest_path}")

    # Save RecordingSet
    recording_manifest_path = output_path / "recordings.jsonl.gz"
    recording_set.to_file(recording_manifest_path)
    logging.info(f"Saved RecordingSet to: {recording_manifest_path}")

    # Print statistics
    print_statistics(cut_set, recording_set, multi_channel_count)


def get_parser():
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Prepare audio metadata from Kaldi wav.scp and segments files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--wav_scp",
        type=str,
        required=True,
        help="Path to Kaldi wav.scp file",
    )
    parser.add_argument(
        "--segments",
        type=str,
        default=None,
        help="Path to Kaldi segments file (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for Lhotse manifest files",
    )
    parser.add_argument(
        "--num_jobs",
        type=int,
        default=4,
        help="Number of parallel jobs for processing",
    )
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="Logging level",
    )
    return parser


def main(cmd=None):
    """Run the main function."""
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    prepare_audio_lhotse(**kwargs)


if __name__ == "__main__":
    main()
