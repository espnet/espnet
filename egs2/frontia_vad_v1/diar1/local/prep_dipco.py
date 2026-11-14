#!/usr/bin/env python3
"""VAD data directory for one DiPCo split (dev or eval), every channel.

DiPCo is evaluation-only in this recipe. Each session has five 7-channel arrays
(U01..U05, one file per channel) and one close-talk microphone per
participant (P*). The transcriptions give each utterance's times per device
("U01".."U05", "close-talk"), and every recording is labelled with the union
of all speakers' utterances in its device's time base.
"""

import argparse
import json
import logging
import re
from pathlib import Path

from vad_data_utils import (
    clip_intervals,
    merge_intervals,
    readable_audio_info,
    wav_scp_entry,
    write_skipped,
    write_vad_dir,
)

# S02_U01.CH1.wav (array channel) or S02_P05.wav (close-talk)
WAV_RE = re.compile(r"^(S\d+)_(?:(U\d+)\.CH\d|(P\d+))\.wav$")


def hms_to_seconds(hms: str) -> float:
    hours, minutes, seconds = hms.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dipco_root", type=Path)
    parser.add_argument("split", choices=["dev", "eval"])
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.dipco_root = args.dipco_root.resolve()

    jsons = sorted(args.dipco_root.glob(f"**/transcriptions/{args.split}/S*.json"))
    if not jsons:
        raise SystemExit(
            f"no transcriptions/{args.split}/S*.json under {args.dipco_root}"
        )
    # session -> device time base ("U01", ..., "close-talk") -> intervals
    sessions = {}
    for js in jsons:
        per_device = {}
        for u in json.loads(js.read_text(encoding="utf-8")):
            for dev, start in u["start_time"].items():
                per_device.setdefault(dev, []).append(
                    (hms_to_seconds(start), hms_to_seconds(u["end_time"][dev]))
                )
        sessions[js.stem] = {d: merge_intervals(v) for d, v in per_device.items()}

    recordings, speech, skipped = {}, {}, []
    for wav in sorted(args.dipco_root.glob(f"**/audio/{args.split}/S*.wav")):
        m = WAV_RE.match(wav.name)
        if not m or m.group(1) not in sessions:
            logging.warning("skipping %s", wav)
            continue
        session, array, _ = m.groups()
        base = array if array else "close-talk"
        if base not in sessions[session]:
            logging.warning("%s: no times for device %s, skipped", wav, base)
            continue
        info = readable_audio_info(wav, skipped)
        if info is None:
            continue
        duration, channels = info
        if channels != 1:
            logging.warning("%s has %d channels, skipped", wav, channels)
            continue
        rec = f"dipco_{wav.stem}"
        recordings[rec] = wav_scp_entry(wav)
        speech[rec] = clip_intervals(sessions[session][base], duration)

    n = write_vad_dir(args.out_dir, recordings, speech)
    write_skipped(args.out_dir, skipped)
    logging.info(
        "%s: %d sessions, %d recordings -> %s",
        args.split,
        len(sessions),
        n,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
