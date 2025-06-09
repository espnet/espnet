# asv_data_prep.py
# Makes the utt2spk, spk2utt, and wav.scp files for ASVspoofLA eval and dev

import argparse
import os
import re


def process_files(src, dst, partition):
    # regex to match eval file names (skip ones used in enrollment)
    if partition == "eval":
        valid_file_regex = re.compile(r"^LA(_E)?_\d+\.flac$")
    else:  # dev
        valid_file_regex = re.compile(r"^LA(_D)?_\d+\.flac$")

    spk2utt = {}
    utt2spk = []
    wav_list = []

    for f in os.listdir(src):
        if not valid_file_regex.match(f):
            continue  # skip files not matching the criteria

        full_path = os.path.join(src, f)
        speaker_id = f.split(".")[0]  # file name without extension as speaker ID
        utt_id = speaker_id  # in this case, speaker ID is the same as utterance ID

        if speaker_id not in spk2utt:
            spk2utt[speaker_id] = []
        spk2utt[speaker_id].append(utt_id)
        utt2spk.append([utt_id, speaker_id])
        wav_list.append([utt_id, full_path])

    with open(os.path.join(dst, "spk2utt"), "w") as f_spk2utt, open(
        os.path.join(dst, "utt2spk"), "w"
    ) as f_utt2spk, open(os.path.join(dst, "wav.scp"), "w") as f_wav:

        for spk in spk2utt:
            f_spk2utt.write(f"{spk} {' '.join(spk2utt[spk])}\n")
        for utt, spk in utt2spk:
            f_utt2spk.write(f"{utt} {spk}\n")
        for utt, path in wav_list:
            f_wav.write(f"{utt} {path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get spk utt mapping")
    parser.add_argument("--src", type=str, required=True, help="src dir")
    parser.add_argument("--dst", type=str, required=True, help="dest dir")
    parser.add_argument("--partition", type=str, required=True, help="partition")
    args = parser.parse_args()

    process_files(args.src, args.dst, args.partition)
