#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import argparse
import logging

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
    jsonstring = json.dumps({'utts': js}, indent=4, sort_keys=True, ensure_ascii=False).encode('utf_8')
    print(jsonstring)
