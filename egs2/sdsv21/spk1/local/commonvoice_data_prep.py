# commonvoice_data_prep.py

# This script is for converting CommonVoice mp3 audio files to wav and
# creating the Kaldi-style utt2spk and wav.scp

import os
import subprocess
import sys


def check_ffmpeg():
    """Ensure FFmpeg is installed."""
    try:
        subprocess.check_output(["which", "ffmpeg"])
    except subprocess.CalledProcessError:
        print("Please install 'ffmpeg' on all worker nodes!")
        sys.exit(1)


def process_line(line, db_base, out_dir):
    """Process each line of the dataset."""
    parts = line.strip().split("\t")

    # verify that there are enough parts to unpack the critical fields
    if len(parts) < 4:
        print(f"Skipping line with insufficient values: {line}")
        return None

    # Assign values to variables, with defaults for missing optional fields
    client_id = parts[0] if len(parts) > 0 else None
    path = parts[1] if len(parts) > 1 else None
    sentence = parts[2] if len(parts) > 2 else None
    up_votes = parts[3] if len(parts) > 3 else None
    down_votes = parts[4] if len(parts) > 4 else None
    age = parts[5] if len(parts) > 5 else None
    gender = parts[6] if len(parts) > 6 else None
    accents = parts[7] if len(parts) > 7 else None
    variant = parts[8] if len(parts) > 8 else None
    locale = parts[9] if len(parts) > 9 else None
    segment = parts[10] if len(parts) > 10 else None

    # Check if critical fields are missing and skip the line if so
    if not client_id or not path:
        print(f"Skipping line due to missing required field: {line}")
        return None

    uttId = path.replace(".mp3", "").replace("/", "-")
    uttId = f"{client_id}-{uttId}"

    # Additional checks and processing
    if os.path.getsize(os.path.join(db_base, "clips", path)) == 0:
        print(f"null file {path}")
        return None

    # return uttId, client_id, path
    return uttId, client_id, path


def main(db_base, dataset, out_dir):
    check_ffmpeg()

    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(
        os.path.join(db_base, f"{dataset}.tsv"), "r", encoding="utf-8"
    ) as csv_file, open(
        os.path.join(out_dir, "utt2spk"), "w", encoding="utf-8"
    ) as f_utt2spk, open(
        os.path.join(out_dir, "wav.scp"), "w", encoding="utf-8"
    ) as f_wav:

        next(csv_file)  # Skip header line
        for line in csv_file:
            result = process_line(line, db_base, out_dir)
            if result:
                uttId, client_id, path = result
                f_wav.write(
                    f"{uttId} ffmpeg -i {db_base}/clips/{path} -f wav -ar 16000 "
                    "-ab 16 -ac 1 - |\n"
                )
                f_utt2spk.write(f"{uttId} {client_id}\n")

    # Post-processing and validation
    subprocess.run(
        [
            "utils/utt2spk_to_spk2utt.pl",
            os.path.join(out_dir, "utt2spk"),
            ">",
            os.path.join(out_dir, "spk2utt"),
        ]
    )
    subprocess.run(["env", "LC_COLLATE=C", "utils/fix_data_dir.sh", out_dir])
    subprocess.run(
        [
            "env",
            "LC_COLLATE=C",
            "utils/validate_data_dir.sh",
            "--non-print",
            "--no-feats",
            out_dir,
        ]
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            f"Usage: {sys.argv[0]} <path-to-commonvoice-corpus> "
            f"<dataset> <valid-train|valid-dev|valid-test>"
        )
        print(
            f"e.g. {sys.argv[0]} /export/data/cv_corpus_v1 cv-valid-train "
            f"valid-train"
        )
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
