#!/usr/bin/env python3

# Copyright 2019 University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import csv
import os
import sys
import subprocess

def get_uttid(wav):
    if '/' in wav:
        return wav.split('/')[-4] + '_' + wav[-21:-4].replace('/', '')

idir = sys.argv[1]

bad_utts = set()

for filename in ['bad_trainval_v03.csv', 'bad_public_train_v03.csv']:
    with open(idir + '/' + filename) as bad_utts_list_file:
        bad_utts_list = csv.DictReader(bad_utts_list_file)
        for row in bad_utts_list:
            bad_utts.add(get_uttid(row['disk_db_path']))

test_utts = set()

with open(idir + '/share_results_v02.csv') as test_utts_list_file:
    test_utts_list = csv.DictReader(test_utts_list_file)
    for row in test_utts_list:
        test_utts.add(get_uttid(row['wav']))

subsets = {'train': {}, 'dev': {}, 'test': {}}

with open(idir + '/public_meta_data_v03.csv') as metafile:
    meta = csv.DictReader(metafile)

    for row in meta:
        [dataset, words, wav] = [row[x] for x in ['dataset', 'text', 'wav_path']]
        wav = idir + '/' + wav[:-3] + 'mp3'
        uttid = get_uttid(wav)

        if uttid in bad_utts or not os.path.isfile(wav):
            continue

        subset = 'train'

        if uttid in test_utts:
            subset = 'test'

        if dataset not in subsets[subset]:
            subsets[subset][dataset] = []

        subsets[subset][dataset].append([uttid, words, wav])

for dataset in subsets['test'].keys():
    l = min(len(subsets['train'][dataset]), len(subsets['test'][dataset]))
    subsets['dev'][dataset] = subsets['train'][dataset][:l]
    subsets['train'][dataset] = subsets['train'][dataset][l:]
    subsets['test_' + dataset] = {'all': subsets['test'][dataset][:]}

for subset in subsets.keys():
    odir = 'data/' + subset
    os.makedirs(odir, exist_ok=True)

    with open(odir + '/text', 'w', encoding='utf-8') as text, \
        open(odir + '/wav.scp', 'w') as wavscp, \
        open(odir + '/utt2spk', 'w') as utt2spk:

        for utt in sum(subsets[subset].values(), []):
            [uttid, words, wav] = utt
            text.write('{} {}\n'.format(uttid, words))
            utt2spk.write('{} {}\n'.format(uttid, uttid))
            wavscp.write('{} sox --norm=-1 {} -r 16k -t wav -c 1 -b 16 -e signed - |\n'.format(uttid, wav))

    subprocess.call('utils/fix_data_dir.sh {}'.format(odir), shell=True)
