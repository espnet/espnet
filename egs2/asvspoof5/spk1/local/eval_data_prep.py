# eval_data_prep.py
# prepares the ASVspoof5 data for evaluation
# no trial labels are created

import os
import sys
from pathlib import Path
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Prepares ASVspoof5 development or eval data')

# Add the arguments
parser.add_argument('--asvspoof5_root', type=str, required=True, help='The directory of the ASVspoof5 enrolment data')
parser.add_argument('--target_dir', type=str, required=True, help='The target directory')
parser.add_argument('--is_progress', type=bool, required=True, help='Whether it is progress or full evaluation')

# Parse the arguments
args = parser.parse_args()

ASVSpoof_root = Path(args.asvspoof5_root)
target_root = Path(args.target_dir)
is_progress = args.is_progress

# check if the ASVspoof5 directories exist
if not ASVSpoof_root.exists():
    print("ASVspoof5 enrolment directory does not exist")
    sys.exit(1)

if is_progress:
    flac_dir = "flac_E_prog"
else:
    flac_dir = "flac_E"

# find all .flac files in the flac directory and write to wav.scp
with open(target_root / "wav.scp", "w") as f_wav:
    for root, _, files in os.walk(ASVSpoof_root / flac_dir):
        for file in files:
            if file.endswith(".flac"):
                uttID = file.split(".")[0]
                path = os.path.join(root, file)
                f_wav.write(f"{uttID} {path}\n")

# make dummy utt2spk and utt2spf files
# each uttID is also the speakerID, and all spoofingID are bonafide
with open(target_root / "utt2spk", "w") as f_utt2spk, open(
    target_root / "utt2spf", "w") as f_utt2spf, open(
        target_root / "wav.scp", "r") as f_wav:
        for line in f_wav:
            parts = line.strip().split()
            uttID = parts[0]
            f_utt2spk.write(f"{uttID} {uttID}\n")
            f_utt2spf.write(f"{uttID} bonafide\n")


