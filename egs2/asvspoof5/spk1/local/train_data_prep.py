# train_data_prep.py
# Prepares ASVspoof5 training data
# Note: This script makes a new speakerID for each speaker + spoofing type

import os
import sys
from pathlib import Path

if len(sys.argv) != 3:
    print("Usage: python3 train_data_prep.py <ASVspoof5 directory> <target directory>")
    sys.exit(1)

ASVSpoof_root = Path(sys.argv[1])
target_root = Path(sys.argv[2])

# check if the ASVspoof5 directory exists
if not ASVSpoof_root.exists():
    print("ASVspoof5 directory does not exist")
    sys.exit(1)

# metadata file
metadata_file = ASVSpoof_root / "ASVspoof5.train.metadata.txt"
if not metadata_file.exists():
    print("ASVspoof5 metadata file does not exist")
    sys.exit(1)

# open all the files
with open(metadata_file, "r") as f_meta, open(
    target_root / "wav.scp", "w") as f_wav, open(
        target_root / "utt2spk", "w") as f_utt2spk, open(
                target_root / "utt2spf", "w") as f_utt2spf:
    lines = f_meta.readlines()
    for line in lines:
        parts = line.strip().split()
        speakerID = parts[0]
        spoofingID = parts[5] # bonafide or spoof
        path = ASVSpoof_root / 'flac_T' / parts[1]
        path = path.with_suffix(".flac")
        file_name = path.stem
        uttID = f"{speakerID}_{file_name}"

        # write wav.scp file
        f_wav.write(f"{uttID} {path}\n")
        # write the utt2spk file
        f_utt2spk.write(f"{uttID} {speakerID}\n")
        # write the utt2spf file
        f_utt2spf.write(f"{uttID} {spoofingID}\n")
