# voxceleb_data_prep.py
# adapted for SASV from espnet/egs2/voxceleb/spk1/local/data_prep.py
import argparse
import os
import sys


def main(args):
    src = args.src
    dst = args.dst

    spk2utt = {}
    utt2spk = []
    wav_list = []

    for r, ds, fs in os.walk(src):
        for f in fs:
            if os.path.splitext(f)[1] != ".wav":
                continue

            utt_dir = os.path.join(r, f)
            spk, vid, utt = utt_dir.split("/")[-3:]  # speaker, video, utterance
            utt_id = "/".join([spk, vid, utt.split(".")[0]])
            utt2spk.append([utt_id, spk])
            wav_list.append([utt_id, utt_dir])

    with open(os.path.join(dst, "utt2spk"), "w" ) as f_utt2spk, open(
        os.path.join(dst, "utt2spf"), "w" ) as f_utt2spf, open(
            os.path.join(dst, "wav.scp"), "w") as f_wav:
        for utt_id, spk in utt2spk:
            f_utt2spk.write(f"{utt_id} {spk}\n")
            # voxceleb2 data is bonafide
            f_utt2spf.write(f"{utt_id} bonafide\n")
        for utt_id, wav in wav_list:
            f_wav.write(f"{utt_id} {wav}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoxCeleb 2 data prep")
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="source directory of voxceleb",
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="destination directory of voxceleb",
    )
    args = parser.parse_args()

    sys.exit(main(args))