#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#           2018 Xuankai Chang (Shanghai Jiao Tong University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import json
import logging
import sys

from espnet.utils.cli_utils import get_commandline_args


def get_parser():
    parser = argparse.ArgumentParser(
        description="convert a json to a transcription file with a token dictionary",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("json", type=str, help="json files")
    parser.add_argument("dict", type=str, help="dict")
    parser.add_argument("--num-spkrs", type=int, default=1, help="number of speakers")
    parser.add_argument("--refs", type=str, nargs="+", help="ref for all speakers")
    parser.add_argument("--hyps", type=str, nargs="+", help="hyp for all outputs")
    return parser


def main(args):
    args = get_parser().parse_args(args)
    convert(args.json, args.dict, args.refs, args.hyps, args.num_spkrs)


def convert(jsonf, dic, refs, hyps, num_spkrs=1):
    n_ref = len(refs)
    n_hyp = len(hyps)
    assert n_ref == n_hyp
    assert n_ref == num_spkrs

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    logging.info("reading %s", jsonf)
    with codecs.open(jsonf, "r", encoding="utf-8") as f:
        j = json.load(f)

    logging.info("reading %s", dic)
    with codecs.open(dic, "r", encoding="utf-8") as f:
        dictionary = f.readlines()
    char_list = [entry.split(" ")[0] for entry in dictionary]
    char_list.insert(0, "<blank>")
    char_list.append("<eos>")

    for ns in range(num_spkrs):
        hyp_file = codecs.open(hyps[ns], "w", encoding="utf-8")
        ref_file = codecs.open(refs[ns], "w", encoding="utf-8")

        for x in j["utts"]:
            # recognition hypothesis
            if num_spkrs == 1:
                seq = [
                    char_list[int(i)]
                    for i in j["utts"][x]["output"][0]["rec_tokenid"].split()
                ]
            else:
                seq = [
                    char_list[int(i)]
                    for i in j["utts"][x]["output"][ns][0]["rec_tokenid"].split()
                ]
            # In the recognition hypothesis,
            # the <eos> symbol is usually attached in the last part of the sentence
            # and it is removed below.
            hyp_file.write(" ".join(seq).replace("<eos>", "")),
            hyp_file.write(
                " (" + j["utts"][x]["utt2spk"].replace("-", "_") + "-" + x + ")\n"
            )

            # reference
            if num_spkrs == 1:
                seq = j["utts"][x]["output"][0]["token"]
            else:
                seq = j["utts"][x]["output"][ns][0]["token"]
            # Unlike the recognition hypothesis,
            # the reference is directly generated from a token without dictionary
            # to avoid to include <unk> symbols in the reference to make scoring normal.
            # The detailed discussion can be found at
            # https://github.com/espnet/espnet/issues/993
            ref_file.write(
                seq + " (" + j["utts"][x]["utt2spk"].replace("-", "_") + "-" + x + ")\n"
            )

        hyp_file.close()
        ref_file.close()


if __name__ == "__main__":
    main(sys.argv[1:])
