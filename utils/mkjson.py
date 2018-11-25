#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function
import sys
import argparse
import codecs
import os
import subprocess
import json
from collections import defaultdict
import logging

def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('feat_scp',
        help='Feature scp',
        type=str)
    parser.add_argument('data',
        help='Kaldi style data directory',
        type=str)
    parser.add_argument('dict',
        help='Dictionary of allowable tokens',
        type=str)
    parser.add_argument('--non-lang-syms',
        help='List of special symbols to process',
        default=None,
        action='store',
        type=str)
    parser.add_argument('--ivectors',
        help='ivector scp',
        default=None,
        required=False,
        action='store',
        type=str)
    parser.add_argument('--oov',
        help='Default oov symbol',
        default='<unk>',
        action='store',
        type=str)
    parser.add_argument('--phonemes',
        help='Flag to request phoneme transcription in json.',
        action='store_true')

    return parser.parse_args()


def main():
    args = parse_input()

    text_file = os.path.join(args.data, 'text')
    utt2spk_file = os.path.join(args.data, 'utt2spk')
    utt2num_frames_file = os.path.join(os.path.dirname(os.path.abspath(args.feat_scp)), "utt2num_frames")

    # Assumes args.feat_scp/../utt2num_frames exists (created by dump.sh)  
    if not os.path.exists(utt2num_frames_file):
        # Create utt2num_frames_file
        subprocess.Popen(['feat-to-len', 'scp:' + args.feat_scp, 'ark,t:{}'.format(utt2num_frames_file)])

    # Check that data/DATASET/{text,utt2spk} exist
    if not os.path.exists(utt2spk_file):
        sys.exit('Expected {} to exist'.format(utt2spk_file))

    if not os.path.exists(text_file):
        sys.exit('Expected {} to exist'.format(text_file))


    # Read in dictionary
    dictionary = {}
    with codecs.open(args.dict, 'r', encoding='utf-8') as f:
        for l in f:
            symbol, val = l.strip().split(None, 1)
            dictionary[symbol] = int(val)

    # Dealing with <unk> symbols
    symbols = defaultdict(lambda: dictionary['<unk>'], dictionary.items())

    num_output_units = len(symbols) + 2 # One for blank, one for <eos>

    if args.phonemes:
        # Read in phoneme dictionary
        phn_dict_fn = "{}.phn".format(args.dict)
        phn_dictionary = {}
        with codecs.open(phn_dict_fn, 'r', encoding='utf-8') as f:
            for l in f:
                symbol, val = l.strip().split(None, 1)
                phn_dictionary[symbol] = int(val)

        # Dealing with <unk> symbols
        phn_symbols = defaultdict(lambda: phn_dictionary['<unk>'],
                                  phn_dictionary.items())

        num_phn_output_units = len(phn_symbols) + 2 # One for blank, one for <eos>

    # Read in non-lang-symbols
    non_lang_symbols = set()
    with codecs.open(args.non_lang_syms, 'r', encoding='utf-8') as f: 
        for l in f:
            non_lang_symbols.add(l.strip())

    # Read in utt2spk
    utt2spk = {}
    with codecs.open(utt2spk_file, 'r', encoding='utf-8') as f: 
        for l in f:
            utt, spk = l.strip().split(None, 1)
            utt2spk[utt] = spk

    # Read in feats.scp
    feats_scp = {}
    with open(args.feat_scp) as f:
        for l in f:
            utt, feat_path = l.strip().split(None, 1)
            feats_scp[utt] = feat_path

    # Get feature dim
    feat_dim = int(subprocess.Popen(['feat-to-dim', 'scp:' + args.feat_scp, '-'],
                                stdout=subprocess.PIPE).communicate()[0])

    # Read in utterance lengths
    utt2num_frames = {}
    with open(utt2num_frames_file) as f:
        for l in f:
            utt, num_frames = l.strip().split(None, 1)
            utt2num_frames[utt] = int(num_frames)

    # Read in (optional) ivectors
    if args.ivectors:
        ivector_scp = {}
        with open(args.ivectors) as f:
            for l in f:
                utt, ivector_path = l.strip().split(None, 1)
                ivector_scp[utt] = ivector_path

        # Get ivector dim
        ivector_dim = int(subprocess.Popen(['feat-to-dim', 'scp:' + args.ivectors, '-'],
                                    stdout=subprocess.PIPE).communicate()[0])


    # Read in targets
    dataset = {}
    with codecs.open(text_file, 'r', encoding='utf-8') as f:
        for i, l in enumerate(f):
            try:
                uttname, text = l.strip().split(None, 1)
            except ValueError:
                # Probably an empty line
                # raise Exception("Line {}, {}".format(i, l.strip()))
                continue

            words = text.split()
            token = ""
            tokenid = []
            for w in words:
                if w in non_lang_symbols:
                    token += w
                    tokenid.append(str(symbols[w]))
                else:
                    token += " ".join(w)
                    tokenid.extend([str(symbols[g]) for g in w])
                token += " <space> "
                tokenid.append(str(symbols['<space>']))

            # Strip trailing space
            space_str = " <space> "
            if token.endswith(space_str):
                token = token[:-len(space_str)]
                tokenid.pop()

            try:
                # input info
                inputs = [
                    {
                        'feat': feats_scp[uttname],
                        'name': 'input1',
                        'shape': [utt2num_frames[uttname], feat_dim]
                    }
                ]
            except KeyError:
                logging.warn("uttname {} not found in feats_scp or utt2num_frames".format(uttname))
                continue

            if args.ivectors:
                inputs.append(
                    {
                        'feat': ivector_scp[uttname],
                        'name': 'ivectors',
                        'shape': [utt2num_frames[uttname], ivector_dim]
                    }
                )

            # output info
            output = [
                {
                    'name': 'grapheme',
                    'shape': [len(tokenid), num_output_units],
                    'text': text,
                    'token': token,
                    'tokenid': ' '.join(tokenid)
                }
            ]

            dataset[uttname] = {
                'input': inputs,
                'output': output,
                'utt2spk': utt2spk[uttname]
            } 

    if args.phonemes:
        phn_text_file = "{}.phn".format(text_file)
        with codecs.open(phn_text_file, 'r', encoding='utf-8') as phn_f:
            for phn_l in phn_f:
                try:
                    phn_uttname, phn_text = phn_l.strip().split(None, 1)
                except ValueError:
                    # Probably an empty line
                    #raise Exception("Line {}, {}".format(i, l.strip()))
                    continue

                phns = phn_text.split()
                phn_token = []
                phn_tokenid = []
                for phn in phns:
                    phn_token.append(phn)
                    phn_tokenid.append(str(phn_symbols[phn]))

                phn_token = " ".join(phn_token)

                try:
                    # output info
                    phn_output = {
                            'name': 'phn',
                            'shape': [len(phn_tokenid), num_phn_output_units],
                            'text': dataset[phn_uttname]['output'][0]['text'],
                            'token': phn_token,
                            'tokenid': ' '.join(phn_tokenid)
                        }

                    dataset[phn_uttname]['output'].append(phn_output)

                except KeyError:
                    logging.warn("phn_uttname {} not found in datasets dict. It either wasn't in the grapheme text file, or wasn't in feats_scp.".format(phn_uttname))


    # Remove all utterances that don't have phoneme output
    for uttname in dataset:
        if len(dataset[uttname]['output']) < 2:
            # Then there can't be both phonemes and graphemes for the
            # utterance. Remove it.
            del(dataset[uttname])
            logging.warn("Removing uttname {} from dataset, since there are less than two outputs (ie. phoneme outputs are missing.".format(uttname))

    # Format output string
    jsonstring = json.dumps({'utts': dataset}, indent=4,
        ensure_ascii=False, sort_keys=True).encode('utf_8')

    print(jsonstring)

if __name__ == "__main__":
    main()
