# create train, dev, test split 90/5/5, no speakers in the train set, 4M4F in test/dev sets

# 126 (94M, 32F), 8 (4M, 4F), 8 (4M, 4F)

import glob
import random

m_speakers = []
f_speakers = []

for f in glob.glob("downloads/tedx_spanish_corpus/speech/*.wav"):
    spkr = "_".join(f.split("/")[-1].split("_")[0:3])
    if "M" in spkr and spkr not in m_speakers:
        m_speakers.append(spkr)
    elif "F" in spkr and spkr not in f_speakers:
        f_speakers.append(spkr)

train = open("local/train.txt", "w")
dev = open("local/dev.txt", "w")
test = open("local/test.txt", "w")

random.shuffle(m_speakers)
random.shuffle(f_speakers)

train_list = m_speakers[0:94] + f_speakers[0:32]
dev_list = m_speakers[94:98] + f_speakers[32:36]
test_list = m_speakers[98:102] + f_speakers[36:40]

random.shuffle(train_list)
random.shuffle(dev_list)
random.shuffle(test_list)

for i in range(126):
    train.write(train_list[i] + "\n")

for i in range(8):
    dev.write(dev_list[i] + "\n")
    test.write(test_list[i] + "\n")
