import argparse
import os
import sys


def main(args):
    src = args.src
    dst = args.dst
    spk_list = args.spk
    if spk_list is not None:
        # read speaker list
        with open(spk_list, "r") as f:
            spk_list = [line.strip() for line in f.readlines()]
    else:
        spk_list = None
    utt_list = args.utt
    if utt_list is not None:
        # read utterance list
        with open(utt_list, "r") as f:
            utt_list = [
                line.strip().split(".")[0].replace("/", "-") for line in f.readlines()
            ]
    else:
        utt_list = None

    spk2utt = {}
    utt2spk = []
    wav_list = []

    for r, ds, fs in os.walk(src):
        for f in fs:
            if os.path.splitext(f)[1] != ".wav":
                continue

            utt_dir = os.path.join(r, f)
            spk, utt = utt_dir.split("/")[-2:]
            if utt.startswith("id"):  # id00001-entertainment-05-003.wav
                spk = utt.split("-")[0]
                utt_id = utt.split(".")[0]
            else:  # entertainment-05-003.wav
                utt_id = "-".join([spk, utt.split(".")[0]])

            # filter by speaker or utterance lists
            if spk_list is not None and spk not in spk_list:
                continue
            if utt_list is not None and utt_id not in utt_list:
                continue

            if spk not in spk2utt:
                spk2utt[spk] = []
            spk2utt[spk].append(utt_id)
            utt2spk.append([utt_id, spk])
            wav_list.append([utt_id, utt_dir])

    with (
        open(os.path.join(dst, "spk2utt"), "w") as f_spk2utt,
        open(os.path.join(dst, "utt2spk"), "w") as f_utt2spk,
        open(os.path.join(dst, "wav.scp"), "w") as f_wav,
    ):
        for spk in spk2utt:
            f_spk2utt.write(f"{spk}")
            for utt in spk2utt[spk]:
                f_spk2utt.write(f" {utt}")
            f_spk2utt.write("\n")

        for utt in utt2spk:
            f_utt2spk.write(f"{utt[0]} {utt[1]}\n")

        for utt in wav_list:
            f_wav.write(f"{utt[0]} {utt[1]}\n")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CN-celeb data preparation")
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="source directory of cnceleb",
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="destination directory of cnceleb",
    )
    parser.add_argument("--spk", type=str, default=None, help="speaker list")
    parser.add_argument("--utt", type=str, default=None, help="utterance list")
    args = parser.parse_args()
    assert not (
        args.spk is not None and args.utt is not None
    ), "Provide either speaker list or utterance list, not both."

    sys.exit(main(args))
