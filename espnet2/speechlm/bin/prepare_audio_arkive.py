#!/usr/bin/env python3
# Copyright 2025 William Chen (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Script for preparing audio metadata from Kaldi format using Arkive."""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

try:
    from arkive import Arkive
except ImportError:
    raise ImportError(
        "arkive is not installed. Install at https://github.com/wanchichen/arkive"
    )


def parse_segments_file(
    segments_path: str,
) -> List[Tuple[str, str, float, float]]:

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

            segments.append((segment_id, recording_id, start, end))

    if invalid_count > 0:
        logging.warning(f"Skipped {invalid_count} invalid segments")

    return segments


def prepare_audio_arkive(
    wav_scp: str,
    segments: Optional[str],
    output_dir: str,
    log_level: str,
):
    """Process Kaldi wav.scp and segments to create Arkive manifests.

    Args:
        wav_scp: Path to Kaldi wav.scp file
        segments: Path to Kaldi segments file (optional)
        output_dir: Directory to save arkive manifest files
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
            audio_paths[recording_id] = os.path.abspath(path)

    if not audio_paths:
        raise ValueError("No valid recordings found in wav.scp")

    logging.info(f"Found {len(audio_paths)} recordings in wav.scp")

    paths = [audio_paths[recording_id] for recording_id in audio_paths]
    ark = Arkive(output_dir)
    ark.append(paths)

    logging.info(
        f"Successfully processed {len(ark.data)}/{len(audio_paths)} recordings"
    )

    df = ark.data
    cols = df.columns.tolist()
    cols.remove("original_file_path")
    cols_to_add = [df[col] for col in cols]
    data = dict(zip(df["original_file_path"], zip(*cols_to_add)))

    new_data = []
    if segments is not None:
        logging.info(f"Reading segments from: {segments}")

        # Parse and validate segments
        segment_data = parse_segments_file(segments)

        if not segment_data:
            raise ValueError("No valid segments found in segments file")

        for segment in segment_data:
            segment_id, recording_id, start, end = segment
            audio_path = audio_paths[recording_id]

            dump_data = data[audio_path]
            dump_data = [audio_path, segment_id, recording_id, start, end] + list(
                dump_data
            )

            new_data.append(dump_data)

    else:
        for recording_id in audio_paths:
            audio_path = audio_paths[recording_id]
            dump_data = data[audio_path]
            dump_data = [audio_path, recording_id, recording_id, None, None] + list(
                dump_data
            )

            new_data.append(dump_data)

    out_df = pd.DataFrame(
        new_data,
        columns=["original_file_path", "utt_id", "doc_id", "start_time", "end_time"]
        + cols,
    )
    out_df.to_parquet(f'{output_dir.rstrip(os.path.sep)}_processed.parquet')


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
        help="Output directory for arkive manifest files",
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
    prepare_audio_arkive(**kwargs)


if __name__ == "__main__":
    main()
