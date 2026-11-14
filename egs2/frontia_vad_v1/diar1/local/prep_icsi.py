#!/usr/bin/env python3
"""VAD data directory for one ICSI split, every channel.

Each meeting has one SPHERE file per microphone (chan*.sph: head-mounted,
lapel and table microphones) and a Mix-Headset.wav. Every file is a
recording labelled with the union of all participants' segments from the NXT
annotations (Segments/<meeting>.<participant>.segs.xml). The split follows
the ICSI partition of lhotse (recipes/icsi.py, lhotse 1.33.0), listed in
local/icsi_split_{train,dev,eval}.txt.
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
    wav_scp_entry,
    write_skipped,
    write_vad_dir,
)

SEG_TAG_RE = re.compile(r"<segment\b[^>]*>")
START_RE = re.compile(r'starttime="([0-9.]+)"')
END_RE = re.compile(r'endtime="([0-9.]+)"')


def read_segments(annotations: Path):
    """meeting -> list of (start, end) over all participants."""
    meetings = defaultdict(list)
    for name, text in iter_annotation_files(annotations, ".segs.xml"):
        meeting = name.split(".")[0]
        for tag in SEG_TAG_RE.findall(text):
            start, end = START_RE.search(tag), END_RE.search(tag)
            if start and end:
                meetings[meeting].append((float(start[1]), float(end[1])))
    if not meetings:
        raise SystemExit(f"no *.segs.xml in {annotations}")
    return meetings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("icsi_root", type=Path, help="holds <meeting>/*.sph")
    parser.add_argument(
        "annotations",
        type=Path,
        help="NXT annotations: ICSI_core_NXT.zip or its directory",
    )
    parser.add_argument("meeting_list", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--sph2pipe", default="sph2pipe", help="sph2pipe command written to wav.scp"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.icsi_root = args.icsi_root.resolve()

    segments = read_segments(args.annotations)
    meetings = args.meeting_list.read_text().split()
    recordings, speech, skipped = {}, {}, []
    for meeting in meetings:
        if meeting not in segments:
            logging.warning("%s has no segment annotation, skipped", meeting)
            continue
        merged = merge_intervals(segments[meeting])
        files = sorted(
            p
            for p in (args.icsi_root / meeting).glob("*")
            if p.suffix.lower() in (".sph", ".wav")
        )
        if not files:
            logging.warning("%s has no audio, skipped", meeting)
        for path in files:
            info = readable_audio_info(path, skipped)
            if info is None:
                continue
            duration, channels = info
            if channels != 1:
                logging.warning("%s has %d channels, skipped", path, channels)
                continue
            rec = f"icsi_{meeting}_{path.stem}"
            recordings[rec] = wav_scp_entry(path, args.sph2pipe)
            speech[rec] = clip_intervals(merged, duration)

    n = write_vad_dir(args.out_dir, recordings, speech)
    write_skipped(args.out_dir, skipped)
    logging.info(
        "%s: %d meetings, %d recordings -> %s",
        args.meeting_list.name,
        len(meetings),
        n,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
