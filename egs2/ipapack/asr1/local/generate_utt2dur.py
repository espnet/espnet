import os
import sys

import kaldiio
from tqdm import tqdm


def generate_utt2dur(scp_file, utt2dur_file, frame_shift=0.01):
    with open(scp_file, "r") as scp, open(utt2dur_file, "w") as dur_f:
        for line in tqdm(scp):
            utt_id, path = line.strip().split(" ", 1)
            try:
                # should be PATH/espnet/egs2/ipapack/asr1
                base_path = os.getcwd()
                features = kaldiio.load_mat(base_path + path)
                num_frames = features.shape[0]
                duration = num_frames * frame_shift  # duration in seconds
                dur_f.write(f"{utt_id} {duration}\n")
            except Exception as e:
                print(f"Error loading features for {utt_id}: {e}")
                continue


ftype = sys.argv[1]
# should be PATH/espnet/egs2/ipapack/asr1/data/
scp_path = (
    os.getcwd()
    + "/data/"
    + ftype
    + "/feats.scp"
)
utt2dur_path = "data/" + ftype + "_utt2dur"
generate_utt2dur(scp_path, utt2dur_path)
