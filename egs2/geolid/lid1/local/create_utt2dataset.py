import argparse
import os

def parser():
    parser = argparse.ArgumentParser(
        description="Create dataset2utt map"
    )
    parser.add_argument(
        "--train_sets",
        type=str,
        required=True,
        help="train sets under dump/raw, e.g., dump/raw/train_voxlingua107_lang",
    )
    parser.add_argument(
        "--train_all_dir",
        type=str,
        required=True,
        help="train all dir under dump/raw",
    )
    args = parser.parse_args()
    return args

def main(args):
    train_sets = args.train_sets.strip().split()
    train_all_dir = args.train_all_dir

    utt2dataset = []

    for train_set in train_sets:
        train_set_name = os.path.basename(train_set)
        utt2spk_path = os.path.join(train_set, "utt2spk")
        with open(utt2spk_path, "r") as f:
            for line in f:
                utt, lang = line.strip().split()
                utt2dataset.append(f"{utt} {train_set_name}\n")
    
    # Write the dataset2utt map to a file
    with open(os.path.join(train_all_dir, "utt2dataset"), "w") as f:
        f.writelines(sorted(utt2dataset))


if __name__ == "__main__":
    args = parser()
    main(args)
