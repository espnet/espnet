import argparse
import glob
import os
import pickle as pk
import sys


def main(args):
    src = args.src
    dst = args.dst

    head_dir = "/".join(args.src.split("/")[:-3])
    # please check the db.sh file to see the filetree of RATS dataset
    f_info = head_dir + "/docs/source_file_info.tab"
    src2spk = {}
    for line in open(f_info, "r").readlines()[1:]:
        src_id, lng, spk, t, part = line.strip().split()
        src2spk[src_id] = spk
    pk.dump(src2spk, open(dst + "src2spk", "wb"))

    spk2utt = {}
    utt2spk = []
    wav_list = []

    fs = glob.glob(os.path.join(src, "audio/*/*/*.flac"))

    for f in fs:

        utt_dir = "/".join(f.split("/")[-4:])
        lng, ch, utt = utt_dir.split("/")[-3:]  # language, channel, utterance
        utt_id = "/".join([lng, ch, utt.split(".")[0]])

        if ch == "src":
            src, lng, ch = utt.strip().split("_")
        else:
            src, trs, lng, ch = utt.strip().split("_")
        ch = ch.split(".")[0]

        spk = src2spk[src]
        spk_pref_utt = "/".join([spk, lng, ch, utt.split(".")[0]])
        if spk not in spk2utt:
            spk2utt[spk] = []
        spk2utt[spk].append(spk_pref_utt)
        utt2spk.append([spk_pref_utt, spk])
        wav_list.append([spk + "/" + utt_id, utt_dir])

    sorted(spk2utt.items(), key=lambda x: x[1])
    utt2spk.sort()
    wav_list.sort()

    with open(os.path.join(dst, "spk2utt"), "w") as f_spk2utt, open(
        os.path.join(dst, "utt2spk"), "w"
    ) as f_utt2spk, open(os.path.join(dst, "wav.scp"), "w") as f_wav:
        for spk in spk2utt:
            f_spk2utt.write(f"{spk}")
            for utt in spk2utt[spk]:
                f_spk2utt.write(f" {utt}")
            f_spk2utt.write("\n")

        for utt in utt2spk:
            f_utt2spk.write(f"{utt[0]} {utt[1]}\n")

        for utt in wav_list:
            f_wav.write(f"{utt[0]} {args.src}{utt[1]}\n")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RATS data preparation")
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="source directory of rats",
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="destination directory of rats",
    )
    args = parser.parse_args()

    sys.exit(main(args))
