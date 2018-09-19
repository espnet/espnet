#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function
import sys
import json
import argparse
import codecs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_json', type=str, help='output json')
    parser.add_argument('jsons', type=str, nargs='+',
                        help='json files')
    args = parser.parse_args()

    # Get superset of utterances
    utterances = set(k for j in args.jsons for k in json.load(open(j))['utts'])
    new_json = {'utts':{}}
    warning_num = 0
    utt_num = 1
    json_files = []
    for jfile in args.jsons:
        with codecs.open(jfile, 'r', encoding='utf-8') as f:
            json_files.append(json.load(f))

    for utt in utterances:
        new_json['utts'][utt] = {}
        inputs = {}
        outputs = {}
        for j in json_files:
            if utt in j['utts']:
                # Merge inputs
                for i in j['utts'][utt]['input']:
                    inputs[i['name']] = i
            
                # Merge outputs
                for o in j['utts'][utt]['output']:
                    outputs[o['name']] = o
            
                # Merge the rest of the keys
                for k in j['utts'][utt]:
                    if k not in ('input', 'output'): 
                        # Assumes no overlapping keys. Otherwise, stored value
                        # is the value of the last json file
                        new_json['utts'][utt][k] = j['utts'][utt][k]
                    elif k == 'input':
                        new_json['utts'][utt]['input'] = inputs.values()
                    elif k == 'output':
                        new_json['utts'][utt]['output'] = outputs.values()
        utt_num += 1

    with codecs.open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(new_json, f, indent=4, sort_keys=True)
    
if __name__ == main():
    main()

