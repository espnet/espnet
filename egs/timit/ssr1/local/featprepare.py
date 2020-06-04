#!/usr/bin/env python3
for dataset in ["data/train", "data/test"]:
    with open(dataset + ".scp", "r") as s:
        utts = s.readlines()
        n_utts = len(utts)
        for i in range(n_utts):
            utts[i] = utts[i].split()[0]
    with open(dataset + ".txt", "r") as t:
        txts = t.readlines()
    text = []
    spk2utt = "Bruce " + " ".join(utts)
    utt2spk = []
    wav = []
    for i in range(n_utts):
        text.append(utts[i] + " " + txts[i])
        utt2spk.append(utts[i] + " Bruce\n")
        wav.append(utts[i] + " " + str(i) + ".wav\n")
    with open(dataset + "/text", "w") as a:
        a.writelines(text)
    with open(dataset + "/spk2utt", "w") as b:
        b.writelines(spk2utt)
    with open(dataset + "/utt2spk", "w") as c:
        c.writelines(utt2spk)
    # wav.scp is not actually used. 
    # It is generated just for fixing data directory.
    with open(dataset + "/wav.scp", "w") as d:
        d.writelines(wav)
