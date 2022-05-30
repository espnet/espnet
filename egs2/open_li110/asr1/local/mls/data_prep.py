import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser(
        description="Process Raw data to Kaldi-like format"
    )
    parser.add_argument("--source", type=str, default="downloads/mls_spanish")
    parser.add_argument("--lang", type=str, default="es")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--target_dir", type=str, default="data")
    return parser


def pad_zero(char, size):
    pad = "0" * (size - len(char))
    return pad + char


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    count = 0
    for subdir in ["train", "dev", "test"]:
        target_dir = os.path.join(args.target_dir, f"{subdir}_{args.lang}_mls")
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        work_dir = os.path.join(args.source, subdir)
        transcription = os.path.join(work_dir, "transcripts.txt")
        transcription_file = open(transcription, "r", encoding="utf-8")

        wavscp = open(os.path.join(target_dir, "wav.scp"), "w", encoding="utf-8")
        text = open(os.path.join(target_dir, "text"), "w", encoding="utf-8")
        utt2spk = open(os.path.join(target_dir, "utt2spk"), "w", encoding="utf-8")
        spk2utt = open(os.path.join(target_dir, "spk2utt"), "w", encoding="utf-8")
        spk_save = {}

        while True:
            line = transcription_file.readline()
            if not line:
                break
            [org_header, trans] = line.split("\t")
            [org_spk, book, seg] = org_header.split("_")
            spk = pad_zero(org_spk, 6)
            header = f"mls_{spk}_{book}_{seg}"
            wavdir = (
                os.path.join(work_dir, "audio", org_spk, book, org_header) + ".flac"
            )
            wavscp.write(f"{header} flac -c -d --silent {wavdir} |\n")
            text.write(f"{header} {trans}")
            utt2spk.write(f"{header} {spk}\n")
            spk_save[spk] = spk_save.get(spk, []) + [header]
            count += 1
            if count % 10000 == 0:
                print(f"process {count} waves")

        for spk in spk_save.keys():
            utts = " ".join(spk_save[spk])
            spk2utt.write(f"{spk} {utts}\n")

    print("pre-processing finished")
