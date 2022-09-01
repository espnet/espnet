# Copyright 2020 Audio, Speech and Language Processing Group @ NWPU (Author: Xian Shi)
# Apache 2.0

import sys

fin = open(sys.argv[1], "r")
fout_text = open(sys.argv[2], "w")
fout_utt2spk = open(sys.argv[3], "w")

for line in fin.readlines():
    uttid, path = line.strip("\n").split("\t")
    text_path = path.replace(".wav", ".txt")
    text_ori = open(text_path, "r").readlines()[0].strip("\n")
    feild = path.split("/")
    accid = feild[-3]
    spkid = accid + "-" + feild[-2]
    fout_utt2spk.write(uttid + "\t" + spkid + "\n")
    fout_text.write(text_ori + "\n")
