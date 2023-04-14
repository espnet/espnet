import os
import glob
import sys
import random

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python data_prep.py [root]")
        sys.exit(1)
    root = sys.argv[1]

    all_audio_list = glob.glob(
        os.path.join(root, "*", "*.mp3")
    )
    audio_basename_set = set()
    for audio_path in all_audio_list:
        filename = os.path.basename(audio_path)
        audio_basename_set.add(filename.split('.')[0])
    
    # Read the TSV file 
    tsvPath = os.path.join(root, "tsv", "raw_30s.tsv")
    # eg："raw_30s_test.tsv"
    with open(tsvPath) as fr, open(
        os.path.join('data', 'train', 'text'), 'w+'
    ) as fw_train_text, open(
        os.path.join('data', 'train', 'wav.scp'), 'w+'
    ) as fw_train_wavscp, open(
        os.path.join('data', 'train', 'utt2spk'), 'w+'
    ) as fw_train_utt2spk, open(
        os.path.join('data', 'test', 'text'), 'w+'
    ) as fw_test_text, open(
        os.path.join('data', 'test', 'wav.scp'), 'w+'
    ) as fw_test_wavscp, open(
        os.path.join('data', 'test', 'utt2spk'), 'w+'
    ) as fw_test_utt2spk:
        for line in fr.readlines()[1:]:
            l=line.split()
            tags = " ".join(l[5:])
            #uttid: 'track_0000214-artist_000014-album_000031'
            uttid = "-".join(l[0:3])
            if os.path.basename(l[3]).split('.')[0] in audio_basename_set:
                low_quality_audio_file = os.path.join(os.path.dirname(audio_path), os.path.basename(l[3]).split('.')[0] + ".low.mp3")
                if random.randint(0, 4) == 4: 
                    fw_test_text.write(f"{uttid} {tags}\n")
                    fw_test_wavscp.write(f"{uttid} ffmpeg -i {low_quality_audio_file} -f wav -ar 16000 -ab 16 -ac 1 - |\n")
                    fw_test_utt2spk.write(f"{uttid} {l[1]}\n")
                else:
                    fw_train_text.write(f"{uttid} {tags}\n")
                    fw_train_wavscp.write(f"{uttid} ffmpeg -i {low_quality_audio_file} -f wav -ar 16000 -ab 16 -ac 1 - |\n")
                    fw_train_utt2spk.write(f"{uttid} {l[1]}\n")
