#!/usr/bin/env python3

if __name__ == "__main__":
    for dataset in ["data/train/", "data/test/"]:
        with open(dataset + "text", "r") as t:
            txts = t.readlines()
        with open(dataset + "feats.scp", "r") as s:
            utts = s.readlines()
            n_utts = len(utts)
            for i in range(n_utts):
                utts[i] = utts[i].split()[0]
        text = []
        spk2utt = "Bruce " + " ".join(utts)
        utt2spk = []
        wav = []
        for i in range(n_utts):
            text.append(utts[i] + " " + txts[i])
            utt2spk.append(utts[i] + " Bruce\n")
            wav.append(utts[i] + " " + str(i) + ".wav\n")

        with open(dataset + "text", "w") as t:
            t.writelines(text)
        with open(dataset + "spk2utt", "w") as s:
            s.writelines(spk2utt)
        with open(dataset + "utt2spk", "w") as u:
            u.writelines(utt2spk)
        # wav.scp is not actually used.
        # It is generated just for fixing data directory.
        with open(dataset + "wav.scp", "w") as w:
            w.writelines(wav)
