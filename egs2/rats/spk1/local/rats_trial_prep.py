import argparse
import sys
import os
import glob

import numpy as np
import random
np.random.seed(1234)


def main(args):
    data_dir = args.data_dir
    
    d_src2spk = {}
    with open(data_dir + "docs/source_file_info.tab", "r") as f_meta:
        for line in f_meta.readlines()[1:]:
            src_id, lng, spk_id, t, part = line.strip().split()
            if part == "dev":
                d_src2spk[src_id] = spk_id

    l_dev = glob.glob(os.path.join(args.data_dir, "data/dev/audio/*/*/*.flac"))

    d_spk2utt = {}
    for line in l_dev:
        f_name = line.strip().split("/")[-1]
        dir = line.strip().split("/")[-2]
    
        if dir == "src":
            src, lng, ch_info = f_name.strip().split("_")
        else:
            src, trs, lng, ch_info = f_name.strip().split("_")

        spk_id = d_src2spk[src]
        if spk_id not in d_spk2utt.keys():
            d_spk2utt[spk_id] = []

        d_spk2utt[spk_id].append(line)

    print ("# of spk: {}".format(len(d_spk2utt.keys())))
    print ("# of utt: {}".format(len(l_dev)))

    used_trials = set()

    def is_unique_trial(class_idx, utt1, utt2):
        trial = (class_idx, utt1, utt2)
        reverse_trial = (class_idx, utt2, utt1)
        if trial in used_trials or reverse_trial in used_trials:
            return False
        used_trials.add(trial)
        return True

    with open(args.out, "w") as f_out:
        for _ in range(len(d_src2spk.keys()) * 20):
            spk = np.random.choice(list(d_spk2utt.keys()), size=1)[0]
            class_idx = np.random.randint(2)

            if class_idx == 0:
                while True:
                    utt1, utt2 = np.random.choice(list(d_spk2utt[spk]), size=2, replace=False)
                    if is_unique_trial(class_idx, utt1, utt2):
                        break
            elif class_idx == 1:
                while True:
                    tmp_lines = list(set(l_dev) - set(d_spk2utt[spk]))
                    utt1, utt2 = np.random.choice(tmp_lines, size=2, replace=False)
                    if is_unique_trial(class_idx, utt1, utt2):
                        break

            f_out.write("{} {} {}\n".format(class_idx, utt1, utt2))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RATS trial generator")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="data directory of rats",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="output of rats trial",
    )

    args = parser.parse_args()

    sys.exit(main(args))
