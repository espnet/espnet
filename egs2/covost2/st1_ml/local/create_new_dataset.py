#!/usr/bin/env python3

# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import os
import random
import sys
import librosa


if len(sys.argv) != 3:
    print("Usage: python prepare_data.py [src_language-ID] [tgt_language-ID]")
    sys.exit(1)

src_lang = sys.argv[1]
tgt_lang = sys.argv[2]

current_train_dir="data/train."+src_lang+"-"+tgt_lang
current_dev_dir="data/dev."+src_lang+"-"+tgt_lang
current_test_dir="data/test."+src_lang+"-"+tgt_lang

for current_dir in [current_train_dir, current_dev_dir, current_test_dir]:
    new_dir = current_dir+"_new"
    wav_scp_file=open(new_dir+"/wav.scp")
    wavpath_dict={}
    for line in wav_scp_file:
        wavpath_dict[line.split()[0]]=line 
    wav_scp_file_old=open(current_dir+"/wav.scp")
    wav_scp_file_new=open(new_dir+"/wav_new.scp","w")
    for line in wav_scp_file_old:
        if line.split()[0] in wavpath_dict:
            wav_scp_file_new.write(line)
    
    for file_name in ["utt2spk","text.lc."+src_lang,"text.lc."+tgt_lang,"text.lc.rm."+src_lang,"text.lc.rm."+tgt_lang,"text.tc."+src_lang,"text.tc."+tgt_lang]:
        file_old=open(current_dir+"/"+file_name)
        file_new=open(new_dir+"/"+file_name,"w")
        for line in file_old:
            if line.split()[0] in wavpath_dict:
                file_new.write(line)



