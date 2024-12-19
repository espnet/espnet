import argparse
import glob
import os
import random
import sys

import numpy as np
from tqdm import tqdm

np.random.seed(1234)


def main(args):
    data_dir = args.data_dir

    # Build src to spk dictionary
    d_src2spk = {}
    with open(os.path.join(data_dir, "docs/source_file_info.tab"), "r") as f_meta:
        for line in f_meta.readlines()[1:]:
            src_id, lng, spk_id, t, part = line.strip().split()
            if part == "dev":
                d_src2spk[src_id] = spk_id

    # Load all .flac files in the dev directory
    l_dev = glob.glob(os.path.join(args.data_dir, "data/dev/audio/*/src/*.flac"))

    # Build spk to utt dictionary more efficiently
    d_spk2utt = {}
    for line in l_dev:
        f_name = line.split("/")[-1]
        src = f_name.split("_")[0]  # Extract src directly from file name
        spk_id = d_src2spk.get(src, None)

        if spk_id is not None:
            d_spk2utt.setdefault(spk_id, []).append(line)

    print(f"# of spk: {len(d_spk2utt)}")
    print(f"# of utt: {len(l_dev)}")

    # Pre-generate the random speaker and class indices to reduce calls to random
    n_trials = len(d_src2spk) * 20
    batch_size = 100

    random_spks = np.random.choice(spk_keys, size=n_trials)
    class_indices = np.random.randint(0, 2, size=n_trials)

    used_trials = set()

    def is_unique_trial(trials):
        """Check if the trial is unique and add to used trials."""
        unique_trials = []
        for trial in trials:
            class_idx, utt1, utt2 = trial
            reverse_trial = (class_idx, utt2, utt1)
            if trial not in used_trials and reverse_trial not in used_trials:
                used_trials.add(trial)
                unique_trials.append(trial)
        return unique_trials

    spk_keys = list(d_spk2utt.keys())
    dev_set = set(l_dev)

    with open(args.out, "w") as f_out:
        for i in tqdm(range(0, n_trials, batch_size)):

            batch_spks = random_spks[i : i + batch_size]
            batch_classes = class_indices[i : i + batch_size]
            batch_trials = []

            for spk, class_idx in zip(batch_spks, batch_classes):
                if class_idx == 1:
                    # Same-speaker pair
                    utt_list = d_spk2utt[spk]
                    if len(utt_list) >= 2:
                        utt_pairs = np.random.choice(
                            utt_list,
                            size=(min(batch_size, len(utt_list) // 2), 2),
                            replace=False,
                        )
                        batch_trials.extend(
                            [(class_idx, utt1, utt2) for utt1, utt2 in utt_pairs]
                        )
                else:
                    # Different-speaker pair
                    utt_list_spk = set(d_spk2utt[spk])
                    tmp_lines = list(dev_set - utt_list_spk)
                    if len(tmp_lines) >= 2:
                        utt_pairs = np.random.choice(
                            tmp_lines,
                            size=(min(batch_size, len(tmp_lines) // 2), 2),
                            replace=False,
                        )
                        batch_trials.extend(
                            [(class_idx, utt1, utt2) for utt1, utt2 in utt_pairs]
                        )

            # Check uniqueness in batch
            unique_trials = is_unique_trial(batch_trials)

            for class_idx, utt1, utt2 in unique_trials:
                line = f"{class_idx} {utt1} {utt2}\n"
                f_out.write(line)
                # print(line)

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
