#!/usr/bin/env bash

# Yongyi Zang, November 2022

import json
import os
import string as string_lib
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("Data Preparation for ASVspoof2019 LA")
    print("Usage: python data_prep.py [asvspoof2019LA_root]")
    sys.exit(1)

# Initialize the path to the ASVspoof2019 LA root directory.
ASVSpoofLocation = sys.argv[1]
ASVSpoofLocation = Path(ASVSpoofLocation)

# Grabbing all data directories.
trainDataDir = Path(os.path.join(ASVSpoofLocation, "ASVspoof2019_LA_train", "flac"))
devDataDir = Path(os.path.join(ASVSpoofLocation, "ASVspoof2019_LA_dev", "flac"))
evalDataDir = Path(os.path.join(ASVSpoofLocation, "ASVspoof2019_LA_eval", "flac"))

# Grab the CM Protocols. 
# See the CM protocols for more detail on expected data format.
protocolPaths = Path(os.path.join(ASVSpoofLocation, "ASVspoof2019_LA_cm_protocols"))
devFile = open(os.path.join(protocolPaths, "ASVspoof2019.LA.cm.dev.trl.txt"), "r")
evalFile = open(os.path.join(protocolPaths, "ASVspoof2019.LA.cm.eval.trl.txt"), "r")
trainFile = open(os.path.join(protocolPaths, "ASVspoof2019.LA.cm.train.trn.txt"), "r")

def generate_data(fileText, dataDir):
    """
    This function takes in one of the cm protocols;
    also need dataDir to be able to find the correct audio files.
    **It takes in the file, not the path of the file.**
    It is expected to output the text, wav_scp, and utt2spk arrays.
    """
    text = []
    wav_scp = []
    utt2spk = []
    
    for line in fileText:
        line = line.strip().split(" ")
        speakerID = line[0]
        utteranceID = line[1]
        
        if line[4] == "spoof":
            label = 1
            attackType = line[3] # e.g. A01, A03, etc.
        else:
            label = 0
            attackType = "NA" # Since it's bonafide.
            
        # TODO: attackType need to be incorporated into the text file.
        
        # Text is in the format of "utt_id transcription".
        # Here, transcription is the label (0 or 1).
        text.append("{} {}".format(utteranceID, label))
        
        # utt2spk is in the format of "utt_id speaker_id".
        utt2spk.append("{} {}".format(utteranceID, speakerID))
        
        # wav_scp is in the format of "utt_id path/to/audio".
        wav_scp.append("{} {}".format(utteranceID, os.path.join(dataDir, utteranceID + ".flac")))
    
    return text, wav_scp, utt2spk

# Start writing to the files.
text, wav_scp, utt2spk = generate_data(trainFile, trainDataDir)
with open("data/train/text", "w") as f:
    for item in text:
        f.write(item + "\n")

with open("data/train/wav.scp", "w") as f:
    for item in wav_scp:
        f.write(item + "\n")

with open("data/train/utt2spk", "w") as f:
    for item in utt2spk:
        f.write(item + "\n")
        
text, wav_scp, utt2spk = generate_data(devFile, devDataDir)
with open("data/dev/text", "w") as f:
    for item in text:
        f.write(item + "\n")

with open("data/dev/wav.scp", "w") as f:
    for item in wav_scp:
        f.write(item + "\n")

with open("data/dev/utt2spk", "w") as f:
    for item in utt2spk:
        f.write(item + "\n")

text, wav_scp, utt2spk = generate_data(evalFile, evalDataDir)
with open("data/eval/text", "w") as f:
    for item in text:
        f.write(item + "\n")

with open("data/eval/wav.scp", "w") as f:
    for item in wav_scp:
        f.write(item + "\n")

with open("data/eval/utt2spk", "w") as f:
    for item in utt2spk:
        f.write(item + "\n")
