# sampledeepmine_prep.py

# This script makes wav.scp utt2spk and spk2utt files from the
#  DeepMine sample subset

import argparse
import os


def generate_output(source_dir, output_dir, input_file):
    spk2utt = {}
    utt2spk = []
    wav_list = []

    with open(input_file, "r") as f:
        paths = f.read().splitlines()

    for path in paths:
        # Extract speaker, session, and (optional) subsession from the path
        parts = path.split("/")
        spk = parts[0]
        session = parts[1]
        if len(parts) > 3:  # Check if subsession exists
            subsession = parts[2]
            utt_id = "/".join(parts)
        else:
            subsession = None
            utt_id = "/".join(parts)

        # Add WAV file path to wav_list
        wav_path = os.path.join(source_dir, f"{path}.wav")
        wav_list.append([utt_id, wav_path])

        # Update spk2utt and utt2spk mappings
        spk2utt.setdefault(spk, []).append(utt_id)
        utt2spk.append([utt_id, spk])

    # Write mappings to output files
    with open(os.path.join(output_dir, "spk2utt"), "w") as f_spk2utt, open(
        os.path.join(output_dir, "utt2spk"), "w"
    ) as f_utt2spk, open(os.path.join(output_dir, "wav.scp"), "w") as f_wav:
        for spk, utts in spk2utt.items():
            f_spk2utt.write(f'{spk} {" ".join(utts)}\n')
        for utt in utt2spk:
            f_utt2spk.write(f"{utt[0]} {utt[1]}\n")
        for utt in wav_list:
            f_wav.write(f"{utt[0]} {utt[1]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate output files from input paths"
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Path to the source directory containing WAV files",
    )
    parser.add_argument(
        "--dst", type=str, required=True, help="Path to the output directory"
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input file"
    )
    args = parser.parse_args()

    generate_output(args.src, args.dst, args.input_file)
