import os
import re
import sys
import argparse
import logging
import numpy as np
from itertools import chain
from glob import iglob
from pathlib import Path

np.random.seed(0)

def get_parser():
    parser = argparse.ArgumentParser(
        description="LRS-3 Data Preparation steps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train_val_path", type=str, help="Path to the Train/ Validation files"
    )
    parser.add_argument(
        "--test_path", type=str, help="Path to the Test files"
    )
    return parser


def train_val_files(train_val_path: str, train_val_ratio: int = 0.92):
    speaker_folders = os.listdir(train_val_path)

    np.random.shuffle(speaker_folders)
    num_speakers = len(speaker_folders)

    num_train = int(train_val_ratio * num_speakers)
    speakers_train = speaker_folders[0: num_train]
    speakers_val   = speaker_folders[num_train: ]

    speakers_train = [os.path.join(train_val_path, folder) for folder in speakers_train]
    speakers_val = [os.path.join(train_val_path, folder) for folder in speakers_val]

    return speakers_train, speakers_val
    
def test_files(test_path: str):
    speakers_test =  os.listdir(test_path)
    test_files = [os.path.join(test_path, folder) for folder in speakers_test]
    return test_files

def read_text(text_file: str):
    with open(text_file, encoding='ISO-8859-1') as f:
        first_line = f.readline()
    text_val = first_line.split('Text:')[1]
    return text_val.strip('\n')

def generate_espnet_files(speaker_folders: list, dataset: str):
    utt2spk = []
    text = []
    wav = []

    for speaker_folder in speaker_folders:
        
        spk_id = os.path.basename(speaker_folder)

        for wav_file in os.listdir(speaker_folder):
            if not wav_file.endswith('.wav'):
                continue
            text_file = wav_file.replace('wav', 'txt')

            wav_full_path = os.path.join(speaker_folder, wav_file)
            text_full_path = os.path.join(speaker_folder, text_file)

            assert os.path.exists(wav_full_path)
            assert os.path.exists(text_full_path)

            utt_id = spk_id + '_' + Path(wav_full_path).stem
            
            utt2spk.append(utt_id + ' ' + spk_id)   
            wav.append(utt_id + ' ' + wav_full_path)
            text.append(utt_id + ' ' + read_text(text_full_path))

    utt2spk_file = os.path.join('data', dataset, 'utt2spk')
    wav_file     = os.path.join('data', dataset, 'wav.scp')
    text_file    =  os.path.join('data', dataset, 'text')

    with open(utt2spk_file, "w") as f:
        for line in utt2spk:
            f.write(line + "\n")

    with open(wav_file, "w") as f:
        for line in wav:
            f.write(line + "\n")

    with open(text_file, "w") as f:
        for line in text:
            f.write(line + "\n")



def main():
    parser = get_parser()
    args = parser.parse_args()
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    
    train_folders, dev_folders = train_val_files(args.train_val_path)
    test_folders = test_files(args.test_path)

    generate_espnet_files(test_folders, 'test')
    generate_espnet_files(train_folders, 'train')
    generate_espnet_files(dev_folders, 'dev')

if __name__ == '__main__':
    main()
