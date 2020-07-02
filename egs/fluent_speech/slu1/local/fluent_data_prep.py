#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Carnegie Mellon University (Muqiao Yang)

# Apache License v2.0 - http://www.apache.org/licenses/

"""
Generating wav.scp, utt2spk and text for train/dev/test in Fluent Speech Commands dataset. 
Change default dataset root accordingly.
"""

import pandas as pd
import os
import sys 

dataset_root = sys.argv[1] 

for split in ["train", "valid", "test"]:

    test_data = pd.read_csv(os.path.join(dataset_root, "data/%s_data.csv" % (split)))
    if split != "valid":
        target_path = "data/%s" % (split)
    else:
        target_path = "data/dev"

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    with open(os.path.join(target_path, "wav.scp"), "w") as f:
        for i in range(len(test_data['path'])):
            path_to_file = dataset_root + test_data['path'][i]
            record_id = test_data['path'][i].split("/")[-1].replace(".wav", "").replace("-", "_")
            speaker_id = test_data['speakerId'][i]
            utt_id = speaker_id + "_" + record_id
            f.write(utt_id + " " + path_to_file + "\n")
        f.close()


    with open(os.path.join(target_path, "utt2spk"), "w") as f:
        for i in range(len(test_data['path'])):
            record_id = test_data['path'][i].split("/")[-1].replace(".wav", "").replace("-", "_")
            speaker_id = test_data['speakerId'][i]
            utt_id = speaker_id + "_" + record_id
            f.write(utt_id + " " + speaker_id + "\n")
        f.close()


    with open(os.path.join(target_path, "text"), "w") as f:
        for i in range(len(test_data['path'])):
            record_id = test_data['path'][i].split("/")[-1].replace(".wav", "").replace("-", "_")
            speaker_id = test_data['speakerId'][i]
            utt_id = speaker_id + "_" + record_id
            action = test_data['action'][i].replace(" ", "_")
            obj = test_data['object'][i]
            location = test_data['location'][i]
            f.write(utt_id + " " + action + " " + obj + " " + location + "\n")
        f.close()

