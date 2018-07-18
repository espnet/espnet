from __future__ import print_function
import kaldi_io_py as kaldi_io
import numpy as np
import os
import argparse
import sys
import pdb


# CORPUS to LANGAUGE ISOCODE MAP
iso_map = { '102': 'asm',
            '103': 'ben',
            '201': 'hat',
            '205': 'kmr',
            '203': 'lao',
            '104': 'pst',
            '202': 'swh',
            '204': 'tam',
            '106': 'tgl',
            '207': 'tpi',
            '105': 'tur',
            '107': 'vie',
            '206': 'zul',
            '101': 'yue',
            '404': 'kat',
            'librispeech': 'eng',
            'csj': 'jap' 
          }


def main():
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('output',
        help='output file',
        type=str)
    parser.add_argument('feat_scp',
        help='scp kaldi feats file',
        type=str)
    parser.add_argument('langvecs',
        help='langvecs dir (path in conf/lang.conf)',
        type=str)
    parser.add_argument('--corpus_id',
        help='The name of the corpus. For BABEL we just use the 3 digit code.',
        type=str,
        action='store',
        default=None)
    parser.add_argument('--feature', 
        help='feature type',
        type=str,
        choices=[ 'phon+inv+fam+geo',
                  'phon+inv+fam',
                  'phon+inv+geo',
                  'phon+inv'
                ],
        default='phon+inv')

    args = parser.parse_args()
    if args.corpus_id is not None:
        lang = iso_map[args.corpus_id]
  
    
    # Lang to int map
    lang2int = {}
    with open(os.path.sep.join((args.langvecs, 'langcodes.txt'))) as f:
        for i, l in enumerate(f):
            lang2int[l.strip()] = i

    # Get acoustic features and langvecs
    new_features = {} 
    
    if args.corpus_id is not None:
        lang_vec = np.load(os.path.join(args.langvecs, args.feature + '.npy'))[:, lang2int[lang]]

    # Append feats and write
    featpath = os.path.abspath(args.output)
    ark_scp_output = 'ark:| copy-feats --compress=true ark:- ark,scp:{}.{}.ark,{}.{}.scp'.format(featpath, args.feature, featpath, args.feature)
    i = 0
    with kaldi_io.open_or_fd(ark_scp_output, 'wb') as f:
        for key, mat in kaldi_io.read_mat_scp(args.feat_scp): 
            print("Utterance: ", i)
            i += 1
            utt_len = len(mat)
            if args.corpus_id is None:
                lang_vec = np.load(os.path.join(args.langvecs, args.feature + '.npy'))[:, lang2int[iso_map[key.split('_', 1)[0]]]]
            
            kaldi_io.write_mat(f, np.concatenate((mat, np.repeat(lang_vec[np.newaxis, :], utt_len, axis=0)), axis=1), key=key)
    
    print()

if __name__ == "__main__":
    main()


