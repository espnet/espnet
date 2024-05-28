import argparse
import json
import os
import re
import sys


def parser():
    parser = argparse.ArgumentParser(description="Data preparation for HiFiTTS")
    parser.add_argument("--train_set", type=str, help="train set")
    parser.add_argument("--dev_set", type=str, help="dev set")
    parser.add_argument("--test_set", type=str, help="dev set")
    parser.add_argument("--dataset_path", type=str, help="dataset path")
    parser.add_argument("--dest_path", type=str, help="destination path")
    return parser.parse_args()


def make_kaldi_dir(path, manifests, dataset_path):
    wav_scp_str = ""
    text_str = ""
    uttr2spk_str = ""
    utt2dur_str = ""
    spk2utt_str = ""

    spk2utt = {}

    for manifest in manifests:
        with open(manifest) as fp:
            manifest_data = fp.readlines()

        for line in manifest_data:
            data = json.loads(line)
            text = data["text"]
            audio = data["audio_filepath"]
            spk = audio.split("/")[1].split("_")[0]
            book_id = audio.split("/")[2]
            uniq_id = spk + "_" + book_id + "_" + audio.split("/")[-1].split(".")[0]
            dur = data["duration"]

            wav_scp_str += f"{uniq_id} {dataset_path}/{audio}\n"
            text_str += f"{uniq_id} {text}\n"
            uttr2spk_str += f"{uniq_id} {spk}\n"
            utt2dur_str += f"{uniq_id} {dur}\n"
            if spk not in spk2utt:
                spk2utt[spk] = []
            spk2utt[spk].append(uniq_id)

    with open(os.path.join(path, "wav.scp"), "w") as fp:
        fp.write(wav_scp_str)

    with open(os.path.join(path, "text"), "w") as fp:
        fp.write(text_str)

    with open(os.path.join(path, "utt2spk"), "w") as fp:
        fp.write(uttr2spk_str)

    for spk, utts in spk2utt.items():
        spk2utt_str += f"{spk} {' '.join(utts)}\n"

    with open(os.path.join(path, "spk2utt"), "w") as fp:
        fp.write(spk2utt_str)


def main():
    args = parser()
    dataset_path = args.dataset_path
    dest_path = args.dest_path
    train_set = os.path.join(dest_path, args.train_set)
    dev_set = os.path.join(dest_path, args.dev_set)
    test_set = os.path.join(dest_path, args.test_set)

    if not os.path.exists(dest_path):
        print(f"Destination path {dest_path} does not exist. Creating it.")
        os.makedirs(dest_path)

    if not os.path.exists(train_set):
        os.makedirs(train_set)
    if not os.path.exists(dev_set):
        os.makedirs(dev_set)
    if not os.path.exists(test_set):
        os.makedirs(test_set)

    all_manifests = [
        os.path.join(dataset_path, fname)
        for fname in os.listdir(dataset_path)
        if fname.endswith(".json")
    ]
    train_manifests = [
        manifest for manifest in all_manifests if re.search("train", manifest)
    ]
    dev_manifests = [
        manifest for manifest in all_manifests if re.search("dev", manifest)
    ]
    test_manifests = [
        manifest for manifest in all_manifests if re.search("test", manifest)
    ]

    make_kaldi_dir(train_set, train_manifests, dataset_path)
    make_kaldi_dir(dev_set, dev_manifests, dataset_path)
    make_kaldi_dir(test_set, test_manifests, dataset_path)

    print("Data preparation done.")


if __name__ == "__main__":
    main()
