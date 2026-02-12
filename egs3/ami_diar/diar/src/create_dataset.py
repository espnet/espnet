"""Create lhotse manifests for diarization datasets.

This script creates lhotse CutSet manifests from diarization data with RTTM annotations.

Supports common diarization datasets:
- AMI Corpus
- CALLHOME
- DIHARD
- VoxConverse
- Custom datasets with wav.scp and RTTM files
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

try:
    from lhotse import CutSet, Recording, RecordingSet, SupervisionSegment, SupervisionSet
    from lhotse.recipes.utils import read_manifests_if_cached
    from lhotse.serialization import SequentialJsonlWriter
except ImportError:
    raise ImportError(
        "lhotse is not installed. Please install it with: pip install lhotse"
    )


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def read_rttm(rttm_path: Path) -> Dict[str, List[SupervisionSegment]]:
    """Read RTTM file and return supervisions grouped by recording ID.

    RTTM format:
    SPEAKER <file-id> <channel-id> <start-time> <duration> <NA> <NA> <speaker-id> <NA> <NA>

    Args:
        rttm_path: Path to RTTM file

    Returns:
        Dictionary mapping recording_id to list of SupervisionSegments
    """
    supervisions = {}

    with open(rttm_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 8:
                logger.warning(f"Skipping invalid RTTM line: {line}")
                continue

            # Parse RTTM fields
            label_type = parts[0]
            recording_id = parts[1]
            channel = parts[2]
            start_time = float(parts[3])
            duration = float(parts[4])
            speaker_id = parts[7]

            if label_type != "SPEAKER":
                continue

            # Create supervision segment
            supervision = SupervisionSegment(
                id=f"{recording_id}_{speaker_id}_{int(start_time*1000):06d}",
                recording_id=recording_id,
                start=start_time,
                duration=duration,
                channel=int(channel) if channel.isdigit() else 0,
                speaker=speaker_id,
            )

            if recording_id not in supervisions:
                supervisions[recording_id] = []
            supervisions[recording_id].append(supervision)

    logger.info(f"Read {sum(len(v) for v in supervisions.values())} segments from {rttm_path}")
    return supervisions


def read_wav_scp(wav_scp_path: Path) -> Dict[str, Path]:
    """Read wav.scp file (Kaldi format).

    Format:
    <recording-id> <path-to-audio>

    Args:
        wav_scp_path: Path to wav.scp file

    Returns:
        Dictionary mapping recording_id to audio path
    """
    wav_files = {}

    with open(wav_scp_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                logger.warning(f"Skipping invalid wav.scp line: {line}")
                continue

            recording_id, audio_path = parts
            wav_files[recording_id] = Path(audio_path)

    logger.info(f"Read {len(wav_files)} recordings from {wav_scp_path}")
    return wav_files


def create_manifests_from_data(
    wav_scp: Path,
    rttm: Path,
    output_dir: Path,
    split_name: str = "train",
) -> CutSet:
    """Create lhotse manifests from wav.scp and RTTM files.

    Args:
        wav_scp: Path to wav.scp file
        rttm: Path to RTTM file
        output_dir: Output directory for manifests
        split_name: Name of the data split (train/dev/test)

    Returns:
        CutSet with audio and annotations
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read input files
    logger.info(f"Creating {split_name} manifests...")
    wav_files = read_wav_scp(wav_scp)
    supervisions_dict = read_rttm(rttm)

    # Create recordings
    recordings = []
    for recording_id, audio_path in wav_files.items():
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}")
            continue

        recording = Recording.from_file(
            path=audio_path,
            recording_id=recording_id,
        )
        recordings.append(recording)

    recording_set = RecordingSet.from_recordings(recordings)
    logger.info(f"Created {len(recording_set)} recordings")

    # Create supervisions
    all_supervisions = []
    for recording_id, segments in supervisions_dict.items():
        if recording_id not in recording_set:
            logger.warning(f"Recording {recording_id} not found in wav.scp")
            continue
        all_supervisions.extend(segments)

    supervision_set = SupervisionSet.from_segments(all_supervisions)
    logger.info(f"Created {len(supervision_set)} supervision segments")

    # Create cuts (combines recordings and supervisions)
    cuts = CutSet.from_manifests(
        recordings=recording_set,
        supervisions=supervision_set,
    )
    logger.info(f"Created {len(cuts)} cuts")

    # Save manifests
    recordings_path = output_dir / f"{split_name}_recordings.jsonl.gz"
    supervisions_path = output_dir / f"{split_name}_supervisions.jsonl.gz"
    cuts_path = output_dir / f"{split_name}_cuts.jsonl.gz"

    recording_set.to_file(recordings_path)
    supervision_set.to_file(supervisions_path)
    cuts.to_file(cuts_path)

    logger.info(f"Saved manifests to {output_dir}")
    logger.info(f"  - Recordings: {recordings_path}")
    logger.info(f"  - Supervisions: {supervisions_path}")
    logger.info(f"  - Cuts: {cuts_path}")

    return cuts


def create_dataset(
    data_dir: Path,
    output_dir: Path,
    dataset_name: str = "custom",
) -> None:
    """Create lhotse manifests for diarization dataset.

    Expected directory structure:
        data_dir/
            train/
                wav.scp
                rttm
            dev/
                wav.scp
                rttm
            test/
                wav.scp
                rttm

    Args:
        data_dir: Root directory containing data splits
        output_dir: Output directory for lhotse manifests
        dataset_name: Name of the dataset
    """
    logger.info(f"Creating manifests for {dataset_name}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Process each split
    splits = ["train", "dev", "test"]

    for split in splits:
        split_dir = data_dir / split

        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue

        wav_scp = split_dir / "wav.scp"
        rttm = split_dir / "rttm"

        if not wav_scp.exists():
            logger.warning(f"wav.scp not found in {split_dir}")
            continue

        if not rttm.exists():
            logger.warning(f"rttm not found in {split_dir}")
            continue

        # Create manifests for this split
        create_manifests_from_data(
            wav_scp=wav_scp,
            rttm=rttm,
            output_dir=output_dir,
            split_name=split,
        )

    logger.info("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Create lhotse manifests for diarization datasets"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Data directory containing train/dev/test splits",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for lhotse manifests",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="custom",
        help="Name of the dataset (for logging)",
    )

    args = parser.parse_args()

    create_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
    )


if __name__ == "__main__":
    main()
