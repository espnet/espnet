#!/usr/bin/env python3

# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import os
import random
import sys
import librosa


if len(sys.argv) != 3:
    print("Usage: python create_splits.py [src_language-ID] [tgt_language-ID]")
    sys.exit(1)

src_lang = sys.argv[1]
tgt_lang = sys.argv[2]

current_train_dir="data/train."+src_lang+"-"+tgt_lang
current_dev_dir="data/dev."+src_lang+"-"+tgt_lang
current_test_dir="data/test."+src_lang+"-"+tgt_lang

for current_dir in [current_train_dir, current_dev_dir, current_test_dir]:
    new_dir = current_dir+"_new"
    os.popen(f"mkdir -p {new_dir}").read()
    wav_scp_file=open(current_dir+"/wav.scp")
    wavpath_dict={}
    for line in wav_scp_file:
        wavpath_dict[line]=line.split()[3]  
    wav_ids = list(wavpath_dict.keys())
    random.shuffle(wav_ids)
    new_wavpath_dict = {}
    if current_dir == current_train_dir:
        totaldur = 10 * 60 * 60  # (in seconds) 10 hours taken for train split
    else:
        totaldur = 4 * 60 * 60  # (in seconds) 4 hours taken for dev/test split
    wav_scp_file_new=open(new_dir+"/wav.scp","w")
    for wav_id in wav_ids:
        dur = librosa.get_duration(filename=wavpath_dict[wav_id])
        new_wavpath_dict[wav_id] = wavpath_dict.pop(wav_id)
        wav_scp_file_new.write(wav_id)
        totaldur -= dur
        if totaldur < 0:
            break
