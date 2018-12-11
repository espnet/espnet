#!/usr/bin/env python2
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

is_python2 = sys.version_info[0] == 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('jsons', type=str, nargs='+',
                        help='json files')
    args = parser.parse_args()

    # logging info
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    # make intersection set for utterance keys
    js = {}
    for x in args.jsons:
        with open(x, 'r') as f:
            j = json.load(f)
        ks = j['utts'].keys()
        logging.debug(x + ': has ' + str(len(ks)) + ' utterances')
        js.update(j['utts'])
    logging.info('new json has ' + str(len(js.keys())) + ' utterances')

    # ensure "ensure_ascii=False", which is a bug
    jsonstring = json.dumps({'utts': js}, indent=4, sort_keys=True, ensure_ascii=False, separators=(',', ': '))
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout if is_python2 else sys.stdout.buffer)
    print(jsonstring)
