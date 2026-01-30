#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mapping", type=str, help="mapping file to concatenate multiple utterances"
    )
    parser.add_argument("text", type=str, help="transcription file")
    parser.add_argument("segments", type=str, help="segmentation file")
    args = parser.parse_args()

    segments = {}
    lineno = 1
    session_prev = ""
    with codecs.open(args.segments, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            session = line.split("-")[0]

            # reset counter
            if session != session_prev:
                lineno = 1

            segments[(session, lineno)] = line
            session_prev = session
            lineno += 1

    with codecs.open(
        os.path.join(os.path.dirname(args.segments), "segments"), "w", encoding="utf-8"
    ) as f:
        for line in codecs.open(args.mapping, "r", encoding="utf-8"):
            session, ids = line.strip().split()
            if len(ids.split("_")) == 1:
                line_new = segments[(session, int(ids))]
            elif len(ids.split("_")) >= 2:
                start_id = int(ids.split("_")[0])
                end_id = int(ids.split("_")[-1])
                segment_start, spk_start, start_t, _ = segments[
                    (session, start_id)
                ].split()
                segment_end, spk_end, _, end_t = segments[(session, end_id)].split()
                line_new = " ".join(
                    [
                        "-".join(segment_start.split("-")[:3])
                        + "-"
                        + segment_end.split("-")[-1],
                        spk_start,
                        start_t,
                        end_t,
                    ]
                )
            f.write(line_new + "\n")

    texts = {}
    lineno = 1
    session_prev = ""
    with codecs.open(args.text, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            session = line.split("-")[0]

            # reset counter
            if session != session_prev:
                lineno = 1

            texts[(session, lineno)] = line
            session_prev = session
            lineno += 1

    with codecs.open(
        os.path.join(os.path.dirname(args.text), "text"), "w", encoding="utf-8"
    ) as f:
        for line in codecs.open(args.mapping, "r", encoding="utf-8"):
            session, ids = line.strip().split()
            if len(ids.split("_")) == 1:
                line_new = texts[(session, int(ids))]
            elif len(ids.split("_")) >= 2:
                start_id = int(ids.split("_")[0])
                segment = texts[(session, start_id)].split()[0]
                trans = " ".join(texts[(session, start_id)].split()[1:])
                for i, id in enumerate(map(int, ids.split("_")[1:])):
                    segment_next = texts[(session, id)].split()[0]
                    trans_next = " ".join(texts[(session, id)].split()[1:])
                    trans += " " + trans_next
                    if i == len(ids.split("_")[1:]) - 1:
                        line_new = " ".join(
                            [
                                "-".join(segment.split("-")[:3])
                                + "-"
                                + segment_next.split("-")[-1],
                                trans,
                            ]
                        )
            f.write(line_new + "\n")


if __name__ == "__main__":
    main()
