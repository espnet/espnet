#!/usr/bin/env python3
"""VAD data directory for one NOTSOFAR-1 split, every channel.

Each meeting (MTG/<id>/) has close-talk microphones (close_talk/CT_*.wav),
7-channel arrays (mc_*/ch0..6.wav) and single-channel devices (sc_*/ch0.wav).
Every file is a recording labelled with the union of all speakers'
utterances in gt_transcription.json (start_time / end_time in seconds).
"""

import argparse
import json
import logging
from pathlib import Path

from vad_data_utils import (
    clip_intervals,
    merge_intervals,
    readable_audio_info,
    wav_scp_entry,
    write_skipped,
    write_vad_dir,
)

SPLIT_DIRS = {"train": "train_set", "dev": "dev_set", "eval": "eval_set"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notsofar_root", type=Path, help="holds benchmark-datasets/")
    parser.add_argument("split", choices=list(SPLIT_DIRS))
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.notsofar_root = args.notsofar_root.resolve()

    # <split>/<release>/MTG/MTG_*: meetings of several releases share their ids,
    # so only the newest release is read
    meetings = sorted(
        args.notsofar_root.glob(f"**/{SPLIT_DIRS[args.split]}/*/MTG/MTG_*")
    )
    releases = sorted({m.parent.parent for m in meetings})
    if len(releases) > 1:
        logging.warning(
            "%d releases of %s; reading %s", len(releases), args.split, releases[-1]
        )
        meetings = [m for m in meetings if m.parent.parent == releases[-1]]
    if not meetings:
        raise SystemExit(
            f"no {SPLIT_DIRS[args.split]}/*/MTG/MTG_* under {args.notsofar_root}"
        )
    recordings, speech, skipped = {}, {}, []
    for mtg in meetings:
        gt = mtg / "gt_transcription.json"
        if not gt.exists():
            logging.warning("%s has no gt_transcription.json, skipped", mtg)
            continue
        merged = merge_intervals(
            (float(u["start_time"]), float(u["end_time"]))
            for u in json.loads(gt.read_text(encoding="utf-8"))
        )
        for wav in sorted(mtg.glob("*/*.wav")):
            info = readable_audio_info(wav, skipped)
            if info is None:
                continue
            duration, channels = info
            if channels != 1:
                logging.warning("%s has %d channels, skipped", wav, channels)
                continue
            rec = f"notsofar_{mtg.name}_{wav.parent.name}_{wav.stem}"
            recordings[rec] = wav_scp_entry(wav)
            speech[rec] = clip_intervals(merged, duration)

    n = write_vad_dir(args.out_dir, recordings, speech)
    write_skipped(args.out_dir, skipped)
    logging.info(
        "%s: %d meetings, %d recordings -> %s",
        args.split,
        len(meetings),
        n,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
