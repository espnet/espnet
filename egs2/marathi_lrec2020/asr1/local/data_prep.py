#!/usr/bin/env python3
import argparse
import os
import glob
import random

def get_speaker(file_id):
    # file_id like "0248_02_1" → speaker = "0248"
    return file_id.split("_")[0]

def write_lines(path, lines):
    """Write sorted unique lines with Unix newlines"""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for line in sorted(set(lines)):
            f.write(line.strip() + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datadir", required=True,
                        help="Path to Marathi dataset root")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Target sample rate")
    args = parser.parse_args()

    # Collect all .3gp files recursively across all subfolders
    audio_files = glob.glob(os.path.join(args.datadir, "**", "*.3gp"), recursive=True)

    speakers = {}
    for audio_path in audio_files:
        base = os.path.splitext(os.path.basename(audio_path))[0]   # e.g. 0248_02_1
        txt_path = audio_path.replace(".3gp", ".txt")
        if not os.path.exists(txt_path):
            continue
        spk = get_speaker(base)
        speakers.setdefault(spk, []).append((base, audio_path, txt_path))

    # Train/dev/test split by speakers
    all_spk = sorted(speakers.keys())
    random.seed(0)
    random.shuffle(all_spk)
    n = len(all_spk)
    train_spk = all_spk[: int(0.8 * n)]
    dev_spk   = all_spk[int(0.8 * n): int(0.9 * n)]
    test_spk  = all_spk[int(0.9 * n):]

    splits = {"train": train_spk, "dev": dev_spk, "test": test_spk}

    seen_utts = set()

    for split, spk_list in splits.items():
        outdir = f"data/marathi_{split}"
        os.makedirs(outdir, exist_ok=True)

        wav_scp, text, utt2spk, spk2utt = [], [], [], {}

        for spk in spk_list:
            for base, audio_path, txt_path in speakers[spk]:
                # utt_id: speaker-base + folder hash (ensures uniqueness)
                folder = os.path.basename(os.path.dirname(audio_path))
                utt_id = f"{spk}-{base}-{folder}"

                if utt_id in seen_utts:
                    continue
                seen_utts.add(utt_id)

                # ffmpeg decode command
                cmd = f"ffmpeg -i {audio_path} -f wav -ar {args.sr} -ac 1 - |"

                # read transcript
                with open(txt_path, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()

                wav_scp.append(f"{utt_id} {cmd}")
                text.append(f"{utt_id} {transcript}")
                utt2spk.append(f"{utt_id} {spk}")
                spk2utt.setdefault(spk, []).append(utt_id)

        # Write all files
        write_lines(os.path.join(outdir, "wav.scp"), wav_scp)
        write_lines(os.path.join(outdir, "text"), text)
        write_lines(os.path.join(outdir, "utt2spk"), utt2spk)
        with open(os.path.join(outdir, "spk2utt"), "w", encoding="utf-8", newline="\n") as f:
            for spk, utts in sorted(spk2utt.items()):
                f.write(f"{spk} {' '.join(sorted(set(utts)))}\n")