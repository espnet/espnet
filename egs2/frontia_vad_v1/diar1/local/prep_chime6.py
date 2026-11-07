#!/usr/bin/env python3
"""VAD data directory for one CHiME-6 split, using every microphone channel.

Every channel of every device (the six Kinect arrays, 4 channels each, and the
binaural in-ear microphones, 2 channels each) becomes its own recording, and
each recording is labelled with the union of all speakers' utterances in its
session. CHiME-6 audio is synchronised, so one set of times serves all devices.
Utterances whose words are "[redacted]" are still speech and are kept.
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from vad_data_utils import (
    clip_intervals,
    merge_intervals,
    readable_audio_info,
    write_skipped,
    write_vad_dir,
)

# S02_U01.CH1.wav (array channel) or S02_P05.wav (binaural in-ear, 2 channels)
WAV_RE = re.compile(r"^(S\d+)_(U\d+\.CH\d|P\d+)\.wav$")


def hms_to_seconds(hms: str) -> float:
    hours, minutes, seconds = hms.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chime6_root", type=Path)
    parser.add_argument("split", choices=["train", "dev", "eval"])
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # wav.scp is read from other directories later, so store absolute paths
    args.chime6_root = args.chime6_root.resolve()

    jsons = sorted(args.chime6_root.glob(f"**/transcriptions/{args.split}/S*.json"))
    if not jsons:
        raise SystemExit(
            f"no transcriptions/{args.split}/S*.json under {args.chime6_root}"
        )
    sessions = {}
    for js in jsons:
        utts = json.loads(js.read_text(encoding="utf-8"))
        sessions[js.stem] = merge_intervals(
            (hms_to_seconds(u["start_time"]), hms_to_seconds(u["end_time"]))
            for u in utts
        )

    recordings, speech, skipped = {}, {}, []
    n_wav = defaultdict(int)
    for wav in sorted(args.chime6_root.glob(f"**/audio/{args.split}/S*.wav")):
        m = WAV_RE.match(wav.name)
        if not m or m.group(1) not in sessions:
            logging.warning("skipping %s", wav)
            continue
        session, device = m.groups()
        info = readable_audio_info(wav, skipped)
        if info is None:
            continue
        duration, channels = info
        segs = clip_intervals(sessions[session], duration)
        for ch in range(channels):
            if channels == 1:
                rec, entry = f"chime6_{session}_{device}", str(wav)
            else:
                rec = f"chime6_{session}_{device}.CH{ch + 1}"
                entry = f"sox {wav} -t wav - remix {ch + 1} |"
            recordings[rec] = entry
            speech[rec] = segs
            n_wav[session] += 1

    missing = sorted(set(sessions) - set(n_wav))
    if missing:
        logging.warning("sessions with no audio found: %s", " ".join(missing))
    n = write_vad_dir(args.out_dir, recordings, speech)
    write_skipped(args.out_dir, skipped)
    logging.info(
        "%s: %d sessions, %d recordings (channels) -> %s",
        args.split,
        len(n_wav),
        n,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
