#!/usr/bin/env python3
"""VAD data directory for one AMI split, using every microphone.

Each wav of a meeting (Headset-*, Lapel-*, Array1-0*, Array2-0*, Mix-Headset)
is its own recording, labelled with the union of all speakers' segments from
the manual annotations (segments/<meeting>.<speaker>.segments.xml). The
meeting lists come from the AMI ASR recipe (full-corpus-ASR partition).
"""

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path

from vad_data_utils import (
    clip_intervals,
    iter_annotation_files,
    merge_intervals,
    readable_audio_info,
    write_skipped,
    write_vad_dir,
)

SEG_TAG_RE = re.compile(r"<segment\b[^>]*>")
START_RE = re.compile(r'transcriber_start="([0-9.]+)"')
END_RE = re.compile(r'transcriber_end="([0-9.]+)"')


def read_segments(annotations: Path):
    """meeting -> list of (start, end) over all speakers."""
    meetings = defaultdict(list)
    for name, text in iter_annotation_files(annotations, ".segments.xml"):
        meeting = name.split(".")[0]
        for tag in SEG_TAG_RE.findall(text):
            start, end = START_RE.search(tag), END_RE.search(tag)
            if start and end:
                meetings[meeting].append((float(start[1]), float(end[1])))
    if not meetings:
        raise SystemExit(f"no *.segments.xml in {annotations}")
    return meetings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ami_root", type=Path, help="holds <meeting>/audio/*.wav")
    parser.add_argument(
        "annotations", type=Path, help="manual annotations: directory or zip"
    )
    parser.add_argument("meeting_list", type=Path, help="e.g. split_dev.orig")
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # wav.scp is read from other directories later, so store absolute paths
    args.ami_root = args.ami_root.resolve()

    segments = read_segments(args.annotations)
    meetings = args.meeting_list.read_text().split()
    recordings, speech, skipped = {}, {}, []
    for meeting in meetings:
        if meeting not in segments:
            logging.warning("%s has no segment annotation, skipped", meeting)
            continue
        merged = merge_intervals(segments[meeting])
        wavs = sorted((args.ami_root / meeting / "audio").glob(f"{meeting}.*.wav"))
        if not wavs:
            logging.warning("%s has no audio, skipped", meeting)
        for wav in wavs:
            mic = wav.name[len(meeting) + 1 : -len(".wav")]
            info = readable_audio_info(wav, skipped)
            if info is None:
                continue
            duration, channels = info
            if channels != 1:
                logging.warning("%s has %d channels, skipped", wav, channels)
                continue
            rec = f"ami_{meeting}_{mic}"
            recordings[rec] = str(wav)
            speech[rec] = clip_intervals(merged, duration)

    n = write_vad_dir(args.out_dir, recordings, speech)
    write_skipped(args.out_dir, skipped)
    logging.info(
        "%s: %d meetings listed, %d recordings -> %s",
        args.meeting_list.name,
        len(meetings),
        n,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
