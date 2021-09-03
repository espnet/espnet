#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import argparse
import codecs
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str, help="input text")
    parser.add_argument("ctm", type=str, help="input ctm file (ASR results)")
    parser.add_argument("set", type=str, help="")
    parser.add_argument("talk_id", type=str, help="")
    args = parser.parse_args()

    refs = []
    with codecs.open(args.text, encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            utt_id = line.split(" ")[0].split("_")[0]
            ref = " ".join(line.split()[1:])
            refs += [(utt_id, ref)]

    ctms = []
    with codecs.open(args.ctm, encoding="utf-8") as f:
        for line in f:
            ctms.append(re.sub(r"[\s]+", " ", line.strip()))
    ctms = sorted(ctms, key=lambda x: float(x.split()[2]))

    threshold = 0.2

    hyps = []
    utt_id = 1
    start_t = None
    end_t = None
    hyp = ""
    num_lines = len(ctms)
    for i, ctm in enumerate(ctms):
        _, _, start_time_w, duration_w, w = ctm.split()[:5]
        w = re.sub(r"([^\(\)]*)\([^\)]+\)", r"\1", w.replace("$", ""))

        if start_t is not None and i < num_lines - 1:
            if (float(start_time_w) - end_t >= threshold) and (end_t - start_t > 0.2):
                # difference utterance
                hyps += [(utt_id, start_t, end_t, hyp[1:])]

                # reset
                hyp = ""
                start_t = None
                end_t = None
                utt_id += 1

        # normalize
        if start_t is None:
            start_t = float(start_time_w)
            end_t = float(start_time_w)
        end_t = float(start_time_w) + float(duration_w)
        if w != "":
            hyp += " " + w.lower()

        # last word in the session
        if i == num_lines - 1:
            hyps += [(utt_id, start_t, end_t, hyp[1:])]

    for i, (utt_id, start_t, end_t, hyp) in enumerate(hyps):
        assert end_t - start_t > 0
        print(
            "%s_%07d_%07d %s %.2f %.2f"
            % (
                args.set + "." + args.talk_id,
                int(start_t * 1000 + 0.5),
                int(end_t * 1000 + 0.5),
                args.set + "." + args.talk_id,
                start_t,
                end_t,
            )
        )


if __name__ == "__main__":
    main()
