#!/usr/bin/env python3
"""Prepare ESPnet-style data directories for the HEROICO corpus (LDC2006S37).

The corpus ships three Spanish subsets:

* ``Answers_Spanish``    - transcript keys look like ``speaker/utt`` (e.g. ``100/10``);
                           audio lives at ``Answers_Spanish/{spk}/{utt}.wav``.
* ``Recordings_Spanish`` - transcript keys are bare utterance numbers (e.g. ``1``);
                           audio lives at ``Recordings_Spanish/{spk}/{utt}.wav``
                           and may be
                           split across nested sub-directories, so we scan recursively.
* ``usma``               - transcript keys look like ``s1``;
                           every speaker folder reads the same prompts,
                           audio lives at ``usma/{speaker_folder}/{s1}.wav`` and
                           the speaker id is the last ``-`` separated segment
                           of the folder name.

Audio is 22050 Hz mono; ``wav.scp`` pipes through ``sox`` to resample to 16000 Hz.

Train/dev/test are split by the (heroico) speaker number:
    train: speakers   1-82
    dev:   speakers  83-92
    test:  speakers 93-102 plus all of USMA
"""

import argparse
import os
import re
import sys

# Characters that are kept verbatim: any unicode letter/number plus whitespace.
# ``re.sub`` with ``[^\w\s]`` and the UNICODE flag removes punctuation while
# preserving Spanish accented letters (á é í ó ú ü ñ ...).
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+", re.UNICODE)


def normalize_text(text):
    """Lowercase, strip punctuation, keep Spanish unicode characters."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def read_transcript(path):
    """Read a Latin-1 transcript file into ``[(key, text), ...]``."""
    entries = []
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                # Key with empty transcript: skip.
                continue
            key, text = parts[0], parts[1]
            text = normalize_text(text)
            if not text:
                continue
            entries.append((key, text))
    return entries


def sox_pipe(wav_path):
    """Return the wav.scp value that resamples to 16 kHz mono via sox."""
    return "sox {} -t wav -r 16000 -c 1 - |".format(wav_path)


def split_for_speaker(spk_num):
    """Map a heroico speaker number to a split name."""
    if 1 <= spk_num <= 82:
        return "train"
    if 83 <= spk_num <= 92:
        return "dev"
    if 93 <= spk_num <= 102:
        return "test"
    return None


def prepare_answers(data_dir, splits):
    """Answers_Spanish: keys are ``speaker/utt``."""
    transcript = os.path.join(data_dir, "data", "transcripts", "heroico-answers.txt")
    audio_root = os.path.join(data_dir, "data", "speech", "heroico", "Answers_Spanish")
    n_missing = 0
    for key, text in read_transcript(transcript):
        if "/" not in key:
            continue
        spk_str, utt_str = key.split("/", 1)
        try:
            spk_num = int(spk_str)
            utt_num = int(utt_str)
        except ValueError:
            continue
        split = split_for_speaker(spk_num)
        if split is None:
            continue
        wav_path = os.path.join(audio_root, spk_str, "{}.wav".format(utt_str))
        if not os.path.isfile(wav_path):
            n_missing += 1
            continue
        utt_id = "heroico_ans_{:03d}_{:04d}".format(spk_num, utt_num)
        spk_id = "heroico_ans_{:03d}".format(spk_num)
        splits[split].append((utt_id, spk_id, wav_path, text))
    return n_missing


def prepare_recordings(data_dir, splits):
    """Recordings_Spanish: keys are bare utt numbers; scan speaker dirs recursively."""
    transcript = os.path.join(data_dir, "data", "transcripts", "heroico-recordings.txt")
    audio_root = os.path.join(
        data_dir, "data", "speech", "heroico", "Recordings_Spanish"
    )
    text_by_utt = {}
    for key, text in read_transcript(transcript):
        try:
            text_by_utt[int(key)] = text
        except ValueError:
            continue

    n_missing = 0
    if not os.path.isdir(audio_root):
        return n_missing
    for spk_str in sorted(os.listdir(audio_root)):
        spk_dir = os.path.join(audio_root, spk_str)
        if not os.path.isdir(spk_dir):
            continue
        try:
            spk_num = int(spk_str)
        except ValueError:
            continue
        split = split_for_speaker(spk_num)
        if split is None:
            continue
        for root, _dirs, files in os.walk(spk_dir):
            for fname in files:
                if not fname.endswith(".wav"):
                    continue
                stem = os.path.splitext(fname)[0]
                try:
                    utt_num = int(stem)
                except ValueError:
                    continue
                text = text_by_utt.get(utt_num)
                if text is None:
                    n_missing += 1
                    continue
                wav_path = os.path.join(root, fname)
                utt_id = "heroico_rec_{:03d}_{:04d}".format(spk_num, utt_num)
                spk_id = "heroico_rec_{:03d}".format(spk_num)
                splits[split].append((utt_id, spk_id, wav_path, text))
    return n_missing


def prepare_usma(data_dir, splits):
    """usma: keys are ``s1``; every speaker folder reads all prompts (all -> test)."""
    transcript = os.path.join(data_dir, "data", "transcripts", "usma-prompts.txt")
    audio_root = os.path.join(data_dir, "data", "speech", "usma")
    prompts = []
    for key, text in read_transcript(transcript):
        m = re.fullmatch(r"s(\d+)", key)
        if not m:
            continue
        prompts.append((key, int(m.group(1)), text))

    n_missing = 0
    if not os.path.isdir(audio_root):
        return n_missing
    for folder in sorted(os.listdir(audio_root)):
        folder_path = os.path.join(audio_root, folder)
        if not os.path.isdir(folder_path):
            continue
        spk_code = folder.split("-")[-1]
        spk_id = "usma_{}".format(spk_code)
        for key, num, text in prompts:
            wav_path = os.path.join(folder_path, "{}.wav".format(key))
            if not os.path.isfile(wav_path):
                n_missing += 1
                continue
            utt_id = "usma_{}_s{:03d}".format(spk_code, num)
            splits["test"].append((utt_id, spk_id, wav_path, text))
    return n_missing


def write_split(out_dir, entries):
    """Write wav.scp, text, utt2spk and spk2utt sorted by utt_id."""
    os.makedirs(out_dir, exist_ok=True)
    entries = sorted(entries, key=lambda e: e[0])

    wav_scp = os.path.join(out_dir, "wav.scp")
    text_f = os.path.join(out_dir, "text")
    utt2spk_f = os.path.join(out_dir, "utt2spk")
    spk2utt_f = os.path.join(out_dir, "spk2utt")

    with (
        open(wav_scp, "w", encoding="utf-8") as fw,
        open(text_f, "w", encoding="utf-8") as ft,
        open(utt2spk_f, "w", encoding="utf-8") as fu,
    ):
        for utt_id, spk_id, wav_path, text in entries:
            fw.write("{} {}\n".format(utt_id, sox_pipe(wav_path)))
            ft.write("{} {}\n".format(utt_id, text))
            fu.write("{} {}\n".format(utt_id, spk_id))

    spk2utt = {}
    for utt_id, spk_id, _wav, _text in entries:
        spk2utt.setdefault(spk_id, []).append(utt_id)
    with open(spk2utt_f, "w", encoding="utf-8") as fs:
        for spk_id in sorted(spk2utt):
            utts = " ".join(sorted(spk2utt[spk_id]))
            fs.write("{} {}\n".format(spk_id, utts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="Path to LDC2006S37 root")
    parser.add_argument("--output_dir", required=True, help="Output data directory")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        sys.exit("Error: data_dir does not exist: {}".format(args.data_dir))

    splits = {"train": [], "dev": [], "test": []}

    miss_ans = prepare_answers(args.data_dir, splits)
    miss_rec = prepare_recordings(args.data_dir, splits)
    miss_usma = prepare_usma(args.data_dir, splits)

    for split in ("train", "dev", "test"):
        out_dir = os.path.join(args.output_dir, split)
        write_split(out_dir, splits[split])
        print("{}: {} utterances".format(split, len(splits[split])))

    if miss_ans or miss_rec or miss_usma:
        print(
            "Skipped (missing audio/text): answers={}, recordings={}, usma={}".format(
                miss_ans, miss_rec, miss_usma
            )
        )


if __name__ == "__main__":
    main()
