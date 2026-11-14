#!/usr/bin/env python3
"""Prepare Kaldi-style data directories for the ALFFA Swahili broadcast-news corpus
(OpenSLR SLR25: https://www.openslr.org/25/).

The downloaded tarball already ships Kaldi-format train/test metadata
(data/{train,test}/{text,utt2spk,wav.scp}), but its wav.scp paths are placeholders
(e.g. "/my_dir/wav/..."). This script reads that shipped metadata and, instead of
trusting those paths, indexes every .wav file under --download_dir by filename and
matches each utterance to its audio by basename. Any utterance whose audio can't be
found is dropped with a warning rather than failing the whole run.

"Speaker" IDs in this corpus are broadcast recording sessions (e.g. SWH-05-20101106),
not individual talkers -- each session/show is one Kaldi "speaker". Since no official
dev split ships with the training audio (only train + test), we hold out two whole
train sessions (one from each of the corpus's two show families) as sw_dev, so no
recording leaks across the train/dev boundary.
"""

import argparse
import os
import sys
from pathlib import Path

# Whole recording sessions held out from train as sw_dev (~6% of train utterances).
# Picked to cover both show families in the corpus (SWH-05-* and SWH-15-*).
DEV_SESSIONS = {"SWH-05-20110327", "SWH-15-20101109"}


def read_kaldi_file(path):
    """Read a Kaldi-style "<id> <rest of line>" file into a dict, id -> rest."""
    entries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            uttid, rest = line.split(maxsplit=1)
            entries[uttid] = rest
    return entries


def index_wav_files(root):
    """Walk `root` once and map basename -> absolute path for every .wav file found."""
    index = {}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".wav"):
                index.setdefault(fn, os.path.join(dirpath, fn))
    return index


def find_metadata_dir(root):
    """Return the directory holding the shipped train/ and test/ Kaldi files."""
    for text in sorted(Path(root).rglob("text")):
        d = text.parent.parent
        if text.parent.name == "train" and (d / "test" / "text").is_file():
            return d
    return None


def check_metadata(meta_dir, split_name):
    """Warn about utterances missing from any of wav.scp, text or utt2spk."""
    files = {
        name: set(read_kaldi_file(meta_dir / split_name / name))
        for name in ("wav.scp", "text", "utt2spk")
    }
    all_ids = set.union(*files.values())
    for name, ids in files.items():
        missing = sorted(all_ids - ids)
        if missing:
            print(
                f"[data_prep] WARNING: {len(missing)} '{split_name}' utterances have "
                f"no {name} entry and will be dropped (showing up to 5): "
                f"{missing[:5]}",
                file=sys.stderr,
            )


def write_kaldi_dir(out_dir, wav_scp, text, utt2spk):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "wav.scp", "w", encoding="utf-8") as f:
        for uttid in sorted(wav_scp):
            f.write(f"{uttid} {wav_scp[uttid]}\n")
    with open(out_dir / "text", "w", encoding="utf-8") as f:
        for uttid in sorted(text):
            f.write(f"{uttid} {text[uttid]}\n")
    with open(out_dir / "utt2spk", "w", encoding="utf-8") as f:
        for uttid in sorted(utt2spk):
            f.write(f"{uttid} {utt2spk[uttid]}\n")


def build_split(split_name, meta_dir, wav_index, keep_uttids=None):
    """Load the shipped metadata for `split_name` ("train" or "test"), remap audio
    paths against `wav_index`, and return (wav_scp, text, utt2spk) dicts restricted
    to `keep_uttids` if given."""
    orig_wav = read_kaldi_file(meta_dir / split_name / "wav.scp")
    text = read_kaldi_file(meta_dir / split_name / "text")
    utt2spk = read_kaldi_file(meta_dir / split_name / "utt2spk")

    complete = orig_wav.keys() & text.keys() & utt2spk.keys()
    wav_scp, missing = {}, []
    for uttid, orig_path in orig_wav.items():
        if uttid not in complete:
            continue  # already reported by check_metadata()
        if keep_uttids is not None and uttid not in keep_uttids:
            continue
        basename = os.path.basename(orig_path)
        real_path = wav_index.get(basename)
        if real_path is None:
            missing.append(basename)
            continue
        wav_scp[uttid] = real_path

    kept = set(wav_scp)
    text = {u: t for u, t in text.items() if u in kept}
    utt2spk = {u: s for u, s in utt2spk.items() if u in kept}

    if missing:
        print(
            f"[data_prep] WARNING: {len(missing)} '{split_name}' utterances had no "
            f"matching .wav file under the downloaded corpus and were dropped "
            f"(showing up to 5): {missing[:5]}",
            file=sys.stderr,
        )
    return wav_scp, text, utt2spk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d",
        "--download_dir",
        required=True,
        help="Directory the ALFFA Swahili tarball was extracted into "
        "(searched recursively for .wav files).",
    )
    parser.add_argument(
        "-o",
        "--data_dir",
        required=True,
        help="ESPnet data/ directory to write sw_train, sw_dev, sw_test into.",
    )
    args = parser.parse_args()

    download_dir = Path(args.download_dir)
    data_dir = Path(args.data_dir)
    if not download_dir.is_dir():
        sys.exit(
            f"[data_prep] {download_dir} is not a directory -- "
            "did the download/extract stage run?"
        )

    print(f"[data_prep] Indexing .wav files under {download_dir} ...")
    wav_index = index_wav_files(download_dir)
    print(f"[data_prep] Found {len(wav_index)} .wav files.")
    if not wav_index:
        sys.exit(
            "[data_prep] No .wav files found -- check that the corpus extracted "
            "correctly."
        )

    meta_dir = find_metadata_dir(download_dir)
    if meta_dir is None:
        sys.exit(
            "[data_prep] Could not find the corpus's data/train and data/test "
            f"metadata under {download_dir}."
        )
    print(f"[data_prep] Using metadata from {meta_dir}")
    for split_name in ("train", "test"):
        check_metadata(meta_dir, split_name)

    # --- test split: used as-is ---
    test_wav, test_text, test_utt2spk = build_split("test", meta_dir, wav_index)
    write_kaldi_dir(data_dir / "sw_test", test_wav, test_text, test_utt2spk)
    print(f"[data_prep] sw_test: {len(test_wav)} utterances")

    # --- train split: carve dev sessions out first so nothing leaks ---
    train_utt2spk_full = read_kaldi_file(meta_dir / "train" / "utt2spk")
    dev_uttids = {u for u, spk in train_utt2spk_full.items() if spk in DEV_SESSIONS}
    train_uttids = {u for u in train_utt2spk_full if u not in dev_uttids}

    dev_wav, dev_text, dev_utt2spk = build_split(
        "train", meta_dir, wav_index, keep_uttids=dev_uttids
    )
    write_kaldi_dir(data_dir / "sw_dev", dev_wav, dev_text, dev_utt2spk)
    print(
        f"[data_prep] sw_dev: {len(dev_wav)} utterances "
        f"(sessions: {sorted(DEV_SESSIONS)})"
    )

    train_wav, train_text, train_utt2spk = build_split(
        "train", meta_dir, wav_index, keep_uttids=train_uttids
    )
    write_kaldi_dir(data_dir / "sw_train", train_wav, train_text, train_utt2spk)
    print(f"[data_prep] sw_train: {len(train_wav)} utterances")


if __name__ == "__main__":
    main()
