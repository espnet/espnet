"""
80% 10% 10% split
"""

import os
import random


if __name__ == "__main__":
    tsv_path = "downloads/utt_spk_text.tsv"

    with open(tsv_path, "r") as inf:
        tsv_lines = inf.readlines()
    tsv_lines = [l.strip() for l in tsv_lines]

    spk2utt = {}
    utt2text = {}
    for l in tsv_lines:
        l_list = l.split("\t")
        fid = l_list[0]
        spk = l_list[1]
        text = l_list[2]
        path = "downloads/data/%s/%s.flac" % (fid[:2], fid)
        if os.path.exists(path):
            utt2text[fid] = text
            if spk in spk2utt:
                spk2utt[spk].append(fid)
            else:
                spk2utt[spk] = [fid]

    spks = sorted(list(spk2utt.keys()))
    random.Random(0).shuffle(spks)
    num_train = int(len(spks)*0.8)
    num_dev = int(len(spks)*0.1)
    train_spks = spks[:num_train]
    dev_spks = spks[num_train:num_train+num_dev]
    test_spks = spks[num_train+num_dev:]

    spks_by_phase = {"train": train_spks, 
                     "dev": dev_spks, 
                     "test": test_spks}
    for phase in spks_by_phase:
        spks = spks_by_phase[phase]
        text_strs = []
        wav_scp_strs = []
        spk2utt_strs = []
        for spk in spks:
            fids = sorted(list(set(spk2utt[spk])))
            utts = [spk+'-'+f for f in fids]
            utts_str = " ".join(utts)
            spk2utt_strs.append("%s %s" % (spk, utts_str))
            for fid, utt in zip(fids, utts):
                cmd = "ffmpeg -i downloads/data/"+fid[:2]+"/"+fid+".flac -f wav -ar 16000 -ab 16 -ac 1 - |"
                text_strs.append("%s %s" % (utt, utt2text[fid]))
                wav_scp_strs.append("%s %s" % (utt, cmd))
        phase_dir = "data/%s" % phase
        os.makedirs(phase_dir)
        text_strs = sorted(text_strs)
        wav_scp_strs = sorted(wav_scp_strs)
        spk2utt_strs = sorted(spk2utt_strs)
        with open(os.path.join(phase_dir, "text"), "w+") as ouf:
            for s in text_strs:
                ouf.write("%s\n" % s)
        with open(os.path.join(phase_dir, "wav.scp"), "w+") as ouf:
            for s in wav_scp_strs:
                ouf.write("%s\n" % s)
        with open(os.path.join(phase_dir, "spk2utt"), "w+") as ouf:
            for s in spk2utt_strs:
                ouf.write("%s\n" % s)
