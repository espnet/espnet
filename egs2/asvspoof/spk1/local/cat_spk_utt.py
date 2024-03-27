# cat_spk_utt.py

# This script concatenates the speaker utterances into a single .flac file

import argparse
import os
import subprocess


def concatenate_speaker_utterances(utterances_dir, txt_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(txt_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            speaker_id = parts[0]
            utterances = parts[1].split(",")

            # ffmpeg input arg list
            input_files = []
            for utterance in utterances:
                input_files.append(f"file '{utterances_dir}/{utterance}.flac'")

            # temporary file list for ffmpeg
            temp_file_path = os.path.join(output_dir, f"{speaker_id}_temp.txt")
            with open(temp_file_path, "w") as f:
                f.write("\n".join(input_files))

            # ffmpeg command for concatenation
            ffmpeg_cmd = [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                temp_file_path,
                "-c",
                "copy",
                os.path.join(output_dir, f"{speaker_id}.flac"),
            ]

            subprocess.run(ffmpeg_cmd, check=True)

            os.remove(temp_file_path)


def main():
    parser = argparse.ArgumentParser(description="Concatenate speaker utts")
    parser.add_argument("--in_dir", required=True, help="input dir")
    parser.add_argument("--in_file", required=True, help="txt with spk n utt")
    parser.add_argument("--out_dir", required=True, help="output dir")

    args = parser.parse_args()

    concatenate_speaker_utterances(args.in_dir, args.in_file, args.out_dir)


if __name__ == "__main__":
    main()
