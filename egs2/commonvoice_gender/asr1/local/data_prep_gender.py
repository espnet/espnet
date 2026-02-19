#!/usr/bin/env python3
# Copyright 2024 CMU WAVLab (Srishti Ginjala)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Data preparation script for CommonVoice with gender-based filtering.

Reads CommonVoice TSV files, filters by gender, and creates
Kaldi-style data directories (wav.scp, text, utt2spk, spk2utt, utt2gender).
"""

import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser(
        description="Prepare CommonVoice data filtered by gender"
    )
    parser.add_argument(
        "--cv_dir",
        type=str,
        required=True,
        help="Path to CommonVoice language directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["validated", "train", "test", "dev"],
        help="Dataset split to process",
    )
    parser.add_argument(
        "--gender",
        type=str,
        required=True,
        choices=["male", "female"],
        help="Gender to filter for",
    )
    parser.add_argument(
        "--out_dir", type=str, required=True, help="Output data directory"
    )
    return parser


def gender_matches(gender_field, target_gender):
    """Check if a gender field matches the target gender.

    CommonVoice has used different gender labels across versions:
    - Older versions: "male", "female", ""
    - Newer versions (2024+): "male_masculine", "female_feminine",
      "intersex", "transgender", "non-binary", "do_not_wish_to_say", ""

    Args:
        gender_field: The gender value from the TSV file
        target_gender: "male" or "female"

    Returns:
        True if the gender matches the target
    """
    gender_lower = gender_field.strip().lower()

    if target_gender == "male":
        return gender_lower in ("male", "male_masculine")
    elif target_gender == "female":
        return gender_lower in ("female", "female_feminine")
    return False


def main():
    parser = get_parser()
    args = parser.parse_args()

    cv_dir = args.cv_dir
    split = args.split
    target_gender = args.gender
    out_dir = args.out_dir

    tsv_path = os.path.join(cv_dir, f"{split}.tsv")
    clips_dir = os.path.join(cv_dir, "clips")

    if not os.path.isfile(tsv_path):
        print(f"Error: {tsv_path} not found", file=sys.stderr)
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    # Open output files
    wav_scp = open(os.path.join(out_dir, "wav.scp"), "w", encoding="utf-8")
    text_f = open(os.path.join(out_dir, "text"), "w", encoding="utf-8")
    utt2spk = open(os.path.join(out_dir, "utt2spk"), "w", encoding="utf-8")
    utt2gender = open(os.path.join(out_dir, "utt2gender"), "w", encoding="utf-8")

    total_count = 0
    filtered_count = 0
    skipped_empty = 0
    skipped_gender = 0

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            total_count += 1
            client_id = row.get("client_id", "")
            filepath = row.get("path", "")
            sentence = row.get("sentence", "")
            gender = row.get("gender", "")

            # Skip if gender doesn't match
            if not gender_matches(gender, target_gender):
                skipped_gender += 1
                continue

            # Skip empty transcriptions
            if not sentence.strip():
                skipped_empty += 1
                continue

            # Skip if audio file doesn't exist or is empty
            audio_path = os.path.join(clips_dir, filepath)
            if not os.path.isfile(audio_path):
                continue
            if os.path.getsize(audio_path) == 0:
                continue

            # Create utterance ID: <speaker_id>-<filename_without_ext>
            utt_base = filepath.replace(".mp3", "").replace("/", "-")
            utt_id = f"{client_id}-{utt_base}"

            # Write to output files
            # Use ffmpeg to convert MP3 to WAV on-the-fly
            # Escape audio_path to prevent command injection
            audio_path_escaped = shlex.quote(audio_path)
            wav_cmd = (
                f"ffmpeg -i {audio_path_escaped} -f wav "
                f"-ar 16000 -ab 16 -ac 1 - |"
            )
            wav_scp.write(f"{utt_id} {wav_cmd}\n")

            # Uppercase text (following CommonVoice convention)
            text_f.write(f"{utt_id} {sentence.upper()}\n")

            utt2spk.write(f"{utt_id} {client_id}\n")

            gender_label = "m" if target_gender == "male" else "f"
            utt2gender.write(f"{utt_id} {gender_label}\n")

            filtered_count += 1

    wav_scp.close()
    text_f.close()
    utt2spk.close()
    utt2gender.close()

    print(f"Split: {split}")
    print(f"  Total rows: {total_count}")
    print(f"  Skipped (gender mismatch): {skipped_gender}")
    print(f"  Skipped (empty text): {skipped_empty}")
    print(f"  Kept (gender={target_gender}): {filtered_count}")

    # Generate spk2utt from utt2spk
    # Use subprocess.run with proper argument handling to prevent command injection
    with open(os.path.join(out_dir, "spk2utt"), "w", encoding="utf-8") as spk2utt_f:
        subprocess.run(
            ["utils/utt2spk_to_spk2utt.pl", os.path.join(out_dir, "utt2spk")],
            stdout=spk2utt_f,
            check=False,
        )

    # Fix and validate data directory
    subprocess.run(["utils/fix_data_dir.sh", out_dir], check=False)
    ret = subprocess.run(
        ["utils/validate_data_dir.sh", "--non-print", "--no-feats", out_dir],
        check=False,
    )
    if ret.returncode != 0:
        print(f"Warning: Data validation failed for {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
