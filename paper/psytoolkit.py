import json
import os
import random
import re
from psytoolkit_templates import *

import shutil

os.chdir('/home/perry/PycharmProjects/espnet/egs2/ljspeech/tts1')

NUM_MOS_FILES = 10
NUM_PREF3_FILES = 8

NATURAL_DIR = 'downloads/LJSpeech-1.1/wavs'
EXP_DIR = 'exp'
SHORT_FORMS = {'downloads/LJSpeech-1.1/wavs': 'natural',
               'fastspeech2_0': '0',
               'fastspeech2_varlr_20max75': '20',
               'fastspeech2_varlr_40max75': '40',
               'fastspeech2_varlr_20to0max75': '20to0',
               'fastspeech2_varlr_40to0max75': '40to0',
               'decode_fastspeech_400epoch': '400epoch',
               'decode_fastspeech_valid.loss.best': 'valid'
               }
MOS_DIRS = ['fastspeech2_0',
            'fastspeech2_varlr_20max75',
            'fastspeech2_varlr_40max75',
            'fastspeech2_varlr_20to0max75',
            'fastspeech2_varlr_40to0max75',
            ]

PREF3_GROUPS = {'20': ('fastspeech2_0', 'fastspeech2_varlr_20max75', 'fastspeech2_varlr_20to0max75'),
                '40': ('fastspeech2_0', 'fastspeech2_varlr_40max75', 'fastspeech2_varlr_40to0max75')
                }

SUBDIRS = ['decode_fastspeech_400epoch', 'decode_fastspeech_valid.loss.best']

WEBAPP_DIR = f'/home/perry/firebase/public'
HOSTING_URL = 'https://icassp2023.web.app'
MOS_DIRNAME = 'mos_test_files'

NONWORDS = re.compile(r'\W')


def main():
    if os.path.exists('test_list'):
        with open('test_list') as f:
            test_list = f.read().splitlines()
    else:
        test_list = []
        with open('data/eval1/wav.scp') as f:
            for line in f:
                test_list.append(line.split()[0])
        random.shuffle(test_list)
        with open('test_list', 'w') as f:
            f.write('\n'.join(test_list))

    mos_files = test_list[:NUM_MOS_FILES]
    num_subdirs = len(SUBDIRS)
    assert NUM_MOS_FILES % num_subdirs == 0
    step = NUM_MOS_FILES // num_subdirs
    subdir_mos_files = [mos_files[i:i + step] for i in range(0, NUM_MOS_FILES, step)]
    dest_dir = f'mos/{SHORT_FORMS[NATURAL_DIR]}'
    os.makedirs(f'{WEBAPP_DIR}/{dest_dir}', exist_ok=True)
    mos_qns = []
    for mos_file in mos_files:
        src_path = f'{NATURAL_DIR}/{mos_file}.wav'
        dest_path = f'{dest_dir}/{mos_file}.wav'
        wav_id = format_wav_str(dest_path)
        webapp_path = f'{WEBAPP_DIR}/{dest_path}'
        shutil.copyfile(src_path, webapp_path)
        print('Copied', src_path, '->', webapp_path)
        mos_qn = MOS_QN_TEMPLATE.format(
            wav_id=wav_id,
            hosting_url=HOSTING_URL,
            wav_path=dest_path,
        )
        mos_qns.append(mos_qn)

    for mos_dir in MOS_DIRS:
        for mos_files, subdir in zip(subdir_mos_files, SUBDIRS):
            dest_dir = f'mos/{SHORT_FORMS[mos_dir]}/{SHORT_FORMS[subdir]}'
            os.makedirs(f'{WEBAPP_DIR}/{dest_dir}', exist_ok=True)
            for mos_file in mos_files:
                src_path = f'{EXP_DIR}/{mos_dir}/{subdir}/eval1_parallel_wavegan/{mos_file}_gen.wav'
                dest_path = f'{dest_dir}/{mos_file}.wav'
                wav_id = format_wav_str(dest_path)
                webapp_path = f'{WEBAPP_DIR}/{dest_path}'
                shutil.copyfile(src_path, webapp_path)
                print('Copied', src_path, '->', webapp_path)
                mos_qn = MOS_QN_TEMPLATE.format(
                    wav_id=wav_id,
                    hosting_url=HOSTING_URL,
                    wav_path=dest_path,
                )
                mos_qns.append(mos_qn)

    mos_section = MOS_SECTION_TEMPLATE.format(mos_questions=''.join(mos_qns))
    with open('survey.txt', 'w') as f:
        f.write(mos_section + '\n\n')

    test_list = test_list[NUM_MOS_FILES:]
    pref3_qns = []
    for group in PREF3_GROUPS:
        for subdir in SUBDIRS:
            for pref3_dir in PREF3_GROUPS[group]:
                os.makedirs(f'{WEBAPP_DIR}/pref3/{group}/{SHORT_FORMS[pref3_dir]}/{SHORT_FORMS[subdir]}', exist_ok=True)
            for pref3_file in test_list[:NUM_PREF3_FILES]:
                pref3_src_paths = []
                pref3_dest_paths = []
                for pref3_dir in PREF3_GROUPS[group]:
                    src_path = f'exp/{pref3_dir}/{subdir}/eval1_parallel_wavegan/{pref3_file}_gen.wav'
                    dest_path = f'pref3/{group}/{SHORT_FORMS[pref3_dir]}/{SHORT_FORMS[subdir]}/{pref3_file}.wav'
                    pref3_src_paths.append(src_path)
                    pref3_dest_paths.append(dest_path)
                    webapp_path = f'{WEBAPP_DIR}/{dest_path}'
                    shutil.copyfile(src_path, webapp_path)
                    print('Copied', src_path, '->', webapp_path)

                group_id = f'{group}_{SHORT_FORMS[subdir]}_{pref3_file}'
                group_id = NONWORDS.sub('_', group_id)
                pref3_qn = PREF3_QN_TEMPLATE.format(
                    group_id=group_id,
                    hosting_url=HOSTING_URL,
                    wav_paths=pref3_dest_paths,
                )
                pref3_qns.append(pref3_qn)

            test_list = test_list[NUM_PREF3_FILES:]

    pref3_section = PREF3_TEST_TEMPLATE.format(pref3_questions=''.join(pref3_qns))
    with open('survey.txt', 'a') as f:
        f.write(pref3_section + '\n\n')


def format_wav_str(s):
    return NONWORDS.sub('_', s[:-4])


if __name__ == '__main__':
    main()
