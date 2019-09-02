#!/usr/bin/env python3

# Copyright 2019 University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import csv
import os
import random
import sys
import subprocess

def get_uttid(wav):
    if '/' in wav:
        return wav.split('/')[-4] + '_' + wav[-21:-4].replace('/', '')

idir = sys.argv[1]

bad_utts = set()

for filename in ['exclude_df_youtube_1120', 'public_exclude_file_v5']:
    with open(idir + '/' + filename + '.csv') as bad_utts_list_file:
        bad_utts_list = csv.DictReader(bad_utts_list_file)
        for row in bad_utts_list:
            bad_utts.add(get_uttid(row['wav']))

subsets = {'train': {}, 'dev': {}, 'test': {}}

words = ''
val_words = set()

for dataset in \
[
# first the validation datasets
'asr_calls_2_val',
'buriy_audiobooks_2_val',
'public_youtube700_val',
# next the training datasets
# (it needs all validation transcripts)
'asr_public_phone_calls_1',
'asr_public_phone_calls_2',
'asr_public_stories_1',
'asr_public_stories_2',
'private_buriy_audiobooks_2',
'public_lecture_1',
'public_series_1',
'public_youtube1120',
'public_youtube1120_hq',
'public_youtube700',
'radio_2',
'ru_RU',
'russian_single',
'tts_russian_addresses_rhvoice_4voices'
]:
    with open(idir + '/' + dataset + '.csv') as metafile:
        meta = csv.reader(metafile)

        for row in meta:
            wav = idir + row[1][19:][:-3] + 'mp3'
            uttid = get_uttid(wav)

            if uttid in bad_utts or not os.path.isfile(wav):
                continue

            with open(wav[:-3] + 'txt', encoding='utf-8') as text_file:
                words = text_file.read().strip().lower()

                subset = 'train'

                if dataset[-4:] == '_val':
                    val_words.add(words)
                    subset = 'test'
                elif words in val_words:
                    continue

                if dataset not in subsets[subset]:
                    subsets[subset][dataset] = []

                subsets[subset][dataset].append([uttid, words, wav])

for dataset in subsets['test'].keys():
    subsets[dataset] = {'all': subsets['test'][dataset][:]}

for subset in subsets.keys():
    if 'all' not in subsets[subset]:
        subsets[subset]['all'] = sum(subsets[subset].values(), [])

random.seed(1)
random.shuffle(subsets['train']['all'])

dev_size = min(int(len(subsets['train']['all']) * 0.1), len(subsets['test']['all']))
subsets['dev']['all'] = subsets['train']['all'][:dev_size]
subsets['train']['all'] = subsets['train']['all'][dev_size:]

del subsets['test']

for subset in subsets.keys():
    odir = 'data/' + subset
    os.makedirs(odir, exist_ok=True)

    with open(odir + '/text', 'w', encoding='utf-8') as text, \
        open(odir + '/wav.scp', 'w') as wavscp, \
        open(odir + '/utt2spk', 'w') as utt2spk:

        for utt in subsets[subset]['all']:
            [uttid, words, wav] = utt
            text.write('{} {}\n'.format(uttid, words))
            utt2spk.write('{} {}\n'.format(uttid, uttid))
            wavscp.write('{} sox --norm=-1 {} -r 16k -t wav -c 1 -b 16 -e signed - |\n'.format(uttid, wav))

    subprocess.call('utils/fix_data_dir.sh {}'.format(odir), shell=True)
