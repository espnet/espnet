#!/usr/bin/env python3

# Copyright 2021 Carnegie Mellon University (Peter Wu)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import argparse
import os
import random
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="downloads directory", type=str, default="downloads")
    args = parser.parse_args()

    # Read all file names in directory
    file_names = os.listdir(args.d)
    file_names = list(set([f.split(".")[0] for f in file_names]))
    
    spk2utt = {}
    utt2text = {}
    for idx, fname in enumerate(file_names):
        
        # File ID
        fid = fname
        
        # Speaker ID – Info not provided so treat each as unique
        spk = fname
        
        # Sometimes there are {lang}.gz files also in the zipped folder, skip those.
        if fname=="TR":
          continue

        # Text – Read from file
        text_path = "%s/%s.txt" % (args.d, fname)
        with open(text_path, "r", encoding="utf-8") as f:
          text = f.read()
        
        path = "%s/%s.flac" % (args.d, fid)
        
        if os.path.exists(path):
            utt2text[fid] = text
            if spk in spk2utt:
                spk2utt[spk].append(fid)
            else:
                spk2utt[spk] = [fid]

    spks = sorted(list(spk2utt.keys()))
    num_fids = 0
    num_test_spks = 0
    test_percentage = 0.2
    for spk in spks:
        num_test_spks += 1
        fids = sorted(list(set(spk2utt[spk])))
        num_fids += len(fids)
        if num_fids >= len(spks)*test_percentage:
            break
    
    # Train-Test Split
    test_spks = spks[:num_test_spks]
    train_dev_spks = spks[num_test_spks:]
    
    # Shuffle Train
    random.Random(0).shuffle(train_dev_spks)
    num_train = int(len(train_dev_spks) * 0.9)
    
    # Train-Dev Split
    train_spks = train_dev_spks[:num_train]
    dev_spks = train_dev_spks[num_train:]

    spks_by_phase = {"train": train_spks, "dev": dev_spks, "test": test_spks}
    flac_dir = "%s/" % args.d
    
    sr = 16000
    for phase in spks_by_phase:
        spks = spks_by_phase[phase]
        text_strs = []
        wav_scp_strs = []
        spk2utt_strs = []
        num_fids = 0
        for spk in spks:
            fids = sorted(list(set(spk2utt[spk])))
            num_fids += len(fids)
            
            if phase == "test" and num_fids > 2000:
                curr_num_fids = num_fids - 2000
                random.Random(1).shuffle(fids)
                fids = fids[:curr_num_fids]
            
            utts = fids
            utts_str = " ".join(utts)
            spk2utt_strs.append("%s %s" % (spk, utts_str))
            
            for fid, utt in zip(fids, utts):
                cmd = "ffmpeg -i %s/%s.flac -f wav -ar %d -ab 16 -ac 1 - |" % (
                    flac_dir,
                    fid,
                    sr,
                )

                text_strs.append("%s %s" % (utt, utt2text[fid]))
                wav_scp_strs.append("%s %s" % (utt, cmd))
        
        phase_dir = "data/mediaspeech_%s" % phase
    
        if not os.path.exists(phase_dir):
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