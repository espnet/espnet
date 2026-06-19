#!/usr/bin/env python3
import argparse
import os
import re
from glob import glob
from random import Random


def get_speaker_id(basename):
    # Return the speaker id for an utterance basename
    # Filenames are "<SRC>_<speaker>_<lineindex>"
    parts = basename.split("_")

    return "_".join(parts[:2])


def collect_split(corpus, split):
    """Return {utt_id: (spk, abs_wav_path, text)} for one official split."""
    audio_dir = os.path.join(corpus, split, "audio_files")
    trans_dir = os.path.join(corpus, split, "trans_files")

    if not os.path.isdir(audio_dir) or not os.path.isdir(trans_dir):
        raise FileNotFoundError(f"Missing {audio_dir} or {trans_dir}")

    utt2info = {}

    for trans_path in sorted(glob(os.path.join(trans_dir, "*.txt"))):
        basename = os.path.splitext(os.path.basename(trans_path))[0]
        wav_path = os.path.join(audio_dir, basename + ".wav")

        if not os.path.exists(wav_path):
            print(f"WARNING: no audio for {basename}, skipping")
            continue

        with open(trans_path, encoding="utf-8") as f:
            # Normalize text by trimming and collapsing whitespace
            text = re.sub(r"\s+", " ", f.read().strip())

        if not text:
            print(f"WARNING: empty transcript for {basename}, skipping")
            continue

        spk = get_speaker_id(basename)

        utt_id = basename if basename.startswith(spk) else f"{spk}-{basename}"
        utt2info[utt_id] = (spk, os.path.abspath(wav_path), text)

    return utt2info


def write_data_dir(dest, utt2info):
    os.makedirs(dest, exist_ok=True)
    utt_ids = sorted(utt2info)

    with open(os.path.join(dest, "text"), "w", encoding="utf-8") as f:
        for utt_id in utt_ids:
            f.write(f"{utt_id} {utt2info[utt_id][2]}\n")

    with open(os.path.join(dest, "wav.scp"), "w", encoding="utf-8") as f:
        for utt_id in utt_ids:
            f.write(f"{utt_id} {utt2info[utt_id][1]}\n")

    with open(os.path.join(dest, "utt2spk"), "w", encoding="utf-8") as f:
        for utt_id in utt_ids:
            f.write(f"{utt_id} {utt2info[utt_id][0]}\n")

    print(f"wrote {len(utt2info)} utts to {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        required=True,
        help="path to the extracted mile_tamil_asr_corpus dir",
    )
    parser.add_argument("--dest", default="data", help="output data root")
    parser.add_argument(
        "--dev_ratio", type=float, default=0.1, help="fraction of train speakers -> dev"
    )
    parser.add_argument("--seed", type=int, default=0, help="dev-split shuffle seed")
    args = parser.parse_args()

    train_all = collect_split(args.corpus, "train")
    test_all = collect_split(args.corpus, "test")

    # Speaker-disjoint dev split out of the official train set
    spks = sorted({spk for spk, _, _ in train_all.values()})
    Random(args.seed).shuffle(spks)
    n_dev = max(1, int(len(spks) * args.dev_ratio))
    dev_spks = set(spks[:n_dev])
    print(
        f"train speakers: {len(spks)} total -> {len(spks) - n_dev} train / {n_dev} dev"
    )

    train_set = {u: v for u, v in train_all.items() if v[0] not in dev_spks}
    dev_set = {u: v for u, v in train_all.items() if v[0] in dev_spks}

    write_data_dir(os.path.join(args.dest, "train_ta"), train_set)
    write_data_dir(os.path.join(args.dest, "dev_ta"), dev_set)
    write_data_dir(os.path.join(args.dest, "test_ta"), test_all)
