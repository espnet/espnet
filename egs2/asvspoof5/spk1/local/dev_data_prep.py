# dev_data_prep.py
# Prepares ASVspoof5 development or evaluation data

import argparse
import os
import sys
from pathlib import Path

# Create the parser
parser = argparse.ArgumentParser(
    description="Prepares ASVspoof5 development or eval data"
)

# Add the arguments
parser.add_argument(
    "--asvspoof5_root",
    type=str,
    required=True,
    help="The directory of the ASVspoof5 enrolment data",
)
parser.add_argument(
    "--target_dir", type=str, required=True, help="The target directory"
)

# Parse the arguments
args = parser.parse_args()

ASVSpoof_root = Path(args.asvspoof5_root)
target_root = Path(args.target_dir)

# check if the ASVspoof5 directories exist
if not ASVSpoof_root.exists():
    print("ASVspoof5 enrolment directory does not exist")
    sys.exit(1)

# metadata file
metadata_file = ASVSpoof_root / "ASVspoof5.dev.metadata.txt"
if not metadata_file.exists():
    print("ASVspoof5 metadata file does not exist")
    sys.exit(1)

# enrollment file
enrollment_file = ASVSpoof_root / "ASVspoof5.dev.enroll.txt"
if not enrollment_file.exists():
    print("ASVspoof5 enrollment file does not exist")
    sys.exit(1)

# open all the files
with open(metadata_file, "r") as f_meta, open(
    target_root / "wav.scp", "w"
) as f_wav, open(target_root / "utt2spk", "w") as f_utt2spk, open(
    target_root / "utt2spf", "w"
) as f_utt2spf:
    lines = f_meta.readlines()
    for line in lines:
        parts = line.strip().split()
        speakerID = parts[0]
        spoofingID = parts[5]  # bonafide or spoof
        path = ASVSpoof_root / "flac_D" / parts[1]
        path = path.with_suffix(".flac")
        file_name = path.stem
        uttID = f"{speakerID}-{file_name}"

        # write wav.scp
        f_wav.write(f"{uttID} {path}\n")
        # write utt2spk
        f_utt2spk.write(f"{uttID} {speakerID}\n")
        # write utt2spf
        f_utt2spf.write(f"{uttID} {spoofingID}\n")

    with open(enrollment_file, "r") as f_enroll:
        lines = f_enroll.readlines()
        for line in lines:
            parts = line.strip().split()
            speakerID = parts[0]
            # enrollment utterances follow and are comma-separated
            enroll_utt = parts[1].split(",")
            for utt in enroll_utt:
                path = ASVSpoof_root / "flac_D" / utt
                path = path.with_suffix(".flac")
                file_name = path.stem
                uttID = f"{speakerID}-{file_name}"
                # write wav.scp
                f_wav.write(f"{uttID} {path}\n")
                # write utt2spk
                f_utt2spk.write(f"{uttID} {speakerID}\n")
                # write utt2spf
                f_utt2spf.write(f"{uttID} bonafide\n")
