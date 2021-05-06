#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import json
import logging
import sys

from espnet.utils.cli_utils import get_commandline_args


def get_parser():
    parser = argparse.ArgumentParser(
        description="concatenate multiple json files for data augmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("jsons", type=str, nargs="+", help="json files")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    # make intersection set for utterance keys
    num_keys = 0
    js = {}
    for i, x in enumerate(args.jsons):
        with codecs.open(x, encoding="utf-8") as f:
            j = json.load(f)
        ks = j["utts"].keys()
        logging.debug(x + ": has " + str(len(ks)) + " utterances")

        num_keys += len(ks)
        if i > 0:
            for k in ks:
                js[k + "." + str(i)] = j["utts"][k]
        else:
            js = j["utts"]
        # js.update(j['utts'])

    # logging.info('new json has ' + str(len(js.keys())) + ' utterances')
    logging.info("new json has " + str(num_keys) + " utterances")

    # ensure "ensure_ascii=False", which is a bug
    jsonstring = json.dumps(
        {"utts": js},
        indent=4,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ": "),
    )
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)
    print(jsonstring)


if __name__ == "__main__":
    main()
