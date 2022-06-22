#!/usr/bin/env python3
import argparse
import logging
from io import BytesIO
from pathlib import Path
from typing import Tuple, Optional

import humanfriendly
import numpy as np
import os
import resampy
import soundfile
from tqdm import tqdm
from typeguard import check_argument_types

from espnet2.utils.cli_utils import get_commandline_args
from espnet2.fileio.read_text import read_2column_text
from espnet2.fileio.midi_scp import MIDIScpWriter, MIDIScpReader


def humanfriendly_or_none(value: str):
    if value in ("none", "None", "NONE"):
        return None
    return humanfriendly.parse_size(value)


def str2int_tuple(integers: str) -> Optional[Tuple[int, ...]]:
    """
    >>> str2int_tuple('3,4,5')
    (3, 4, 5)
    """
    assert check_argument_types()
    if integers.strip() in ("none", "None", "NONE", "null", "Null", "NULL"):
        return None
    return tuple(map(int, integers.strip().split(",")))


def main():
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    parser = argparse.ArgumentParser(
        description='Create waves list from "midi.scp"',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scp")
    parser.add_argument("outdir")
    parser.add_argument(
        "--name",
        default="midi",
        help="Specify the prefix word of output file name " 'such as "wav.scp"',
    )
    parser.add_argument("--segments", default=None)
    parser.add_argument(
        "--fs",
        type=np.int32,
        default=None,
        help="If the sampling rate specified, " "Change the sampling rate.",
    )
    group = parser.add_mutually_exclusive_group()
    # TODO: in midi, the reference channels should be related to track, it is not implemented now
    group.add_argument("--ref-channels", default=None, type=str2int_tuple)
    group.add_argument("--utt2ref-channels", default=None, type=str)
    args = parser.parse_args()

    if args.ref_channels is not None:

        def utt2ref_channels(x) -> Tuple[int, ...]:
            return args.ref_channels

    elif args.utt2ref_channels is not None:
        utt2ref_channels_dict = read_2column_text(args.utt2ref_channels)

        def utt2ref_channels(x, d=utt2ref_channels_dict) -> Tuple[int, ...]:
            chs_str = d[x]
            return tuple(map(int, chs_str.split()))

    else:
        utt2ref_channels = None

    # load segments
    if args.segments is not None:
        segments = {}
        with open(args.segments) as f:
            for line in f:
                if len(line) == 0:
                    continue
                utt_id, recording_id, segment_begin, segment_end = line.strip().split(
                    " "
                )
                segments[utt_id] = (
                    recording_id,
                    float(segment_begin),
                    float(segment_end),
                )

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    out_midiscp = Path(args.outdir) / f"{args.name}.scp"
    if args.segments is not None:
        loader = MIDIScpReader(args.scp, rate=args.fs)
        writer = MIDIScpWriter(args.outdir, out_midiscp, format="midi", rate=args.fs,)

        cache = (None, None, None)
        for utt_id, (recording, start, end) in tqdm(segments.items()):
            # TODO: specify track information here
            if recording == cache[0]:
                note_seq, tempo_seq = cache[1], cache[2]
            else:
                pitch_aug_factor = 0
                time_aug_factor = 1.0
                note_seq, tempo_seq = loader[(recording, pitch_aug_factor, time_aug_factor)]
                cache = (recording, note_seq, tempo_seq)
            if args.fs is not None:
                start = int(start * args.fs)
                end = int(end * args.fs)
                if start < 0:
                    start = 0
                if start > len(note_seq):
                    start = len(note_seq)
                if end > len(note_seq):
                    end = len(note_seq)
            else:
                start = np.searchsorted([item[0] for item in note_seq], start, "left")
                end = np.searchsorted([item[1] for item in note_seq], end, "left")
            sub_note = note_seq[start:end]
            sub_tempo = tempo_seq[start:end]

            writer[utt_id] = sub_note, sub_tempo

    else:
        # midi_scp does not need to change, when no segments is applied
        # Note things will change, after finish other todos in the script
        os.system("cp {} {}".format(args.scp, Path(args.outdir) / f"{args.name}.scp"))


if __name__ == "__main__":
    main()