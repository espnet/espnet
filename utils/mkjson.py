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
        for l in f:
            uttname, text = l.strip().split(None, 1)
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

            # input info
            inputs = [
                {
                    'feat': feats_scp[uttname],
                    'name': 'input1',
                    'shape': [utt2num_frames[uttname], feat_dim]
                }
            ]
            
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

    # Format output string
    jsonstring = json.dumps({'utts': dataset}, indent=4,
        ensure_ascii=False, sort_keys=True).encode('utf_8')

    print(jsonstring)

if __name__ == "__main__":
    main()
    
