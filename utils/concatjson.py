#!/usr/bin/env python
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function
from __future__ import unicode_literals

import argparse
import codecs
import json
import logging
import sys

from espnet.utils.cli_utils import get_commandline_args

is_python2 = sys.version_info[0] == 2


def get_parser():
    parser = argparse.ArgumentParser(
        description='concatenate json files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('jsons', type=str, nargs='+',
                        help='json files')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    # logging info
    logfmt = '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    # make intersection set for utterance keys
    js = {}
    for x in args.jsons:
        with codecs.open(x, encoding="utf-8") as f:
            j = json.load(f)
        ks = j['utts'].keys()
        logging.debug(x + ': has ' + str(len(ks)) + ' utterances')
        js.update(j['utts'])
    logging.info('new json has ' + str(len(js.keys())) + ' utterances')

    # ensure "ensure_ascii=False", which is a bug
    jsonstring = json.dumps({'utts': js}, indent=4, sort_keys=True,
                            ensure_ascii=False, separators=(',', ': '))
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout if is_python2 else sys.stdout.buffer)
    print(jsonstring)
