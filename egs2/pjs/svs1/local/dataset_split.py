import argparse
import os
import shutil

UTT_PREFIX = "pjs"
DEV_LIST = [
    "pjs002",
    "pjs012",
    "pjs022",
    "pjs032",
    "pjs042",
    "pjs052",
    "pjs062",
    "pjs072",
    "pjs082",
    "pjs092",
]
TEST_LIST = [
    "pjs007",
    "pjs017",
    "pjs027",
    "pjs037",
    "pjs047",
    "pjs057",
    "pjs067",
    "pjs077",
    "pjs087",
    "pjs097",
]


def train_check(song):
    return (song not in DEV_LIST) and (song not in TEST_LIST)


def dev_check(song):
    return song in DEV_LIST


def test_check(song):
    return song in TEST_LIST


def makedir(data_url):
    if os.path.exists(data_url):
        shutil.rmtree(data_url)

    os.makedirs(data_url)


def process_pho_info(phone):
    info = open(phone, "r", encoding="utf-8")
    label_info = []
    pho_info = []
    for line in info.readlines():
        line = line.strip().split()
        label_info.append(
            "{} {} {}".format(
                float(line[0]) / 1e7, float(line[1]) / 1e7, line[2].strip()
            )
        )
        pho_info.append(line[2].strip())
    return " ".join(label_info), " ".join(pho_info)


def process_subset(src_data, subset, check_func, fs, wav_dump):
    subfolder = os.listdir(src_data)
    makedir(subset)
    wavscp = open(os.path.join(subset, "wav.scp"), "w", encoding="utf-8")
    utt2spk = open(os.path.join(subset, "utt2spk"), "w", encoding="utf-8")
    label_scp = open(os.path.join(subset, "label"), "w", encoding="utf-8")
    musicxmlscp = open(os.path.join(subset, "score.scp"), "w", encoding="utf-8")

    for folder in subfolder:
        if not os.path.isdir(os.path.join(src_data, folder)):
            continue
        if not check_func(folder):
            continue
        if folder == "background_noise":
            continue
        utt_id = folder

        cmd = "sox {}_song.wav -c 1 -t wavpcm -b 16 -r {} {}_bits16.wav".format(
            os.path.join(src_data, folder, folder),
            fs,
            os.path.join(wav_dump, folder),
        )
        os.system(cmd)

        wavscp.write(
            "{} {}\n".format(
                utt_id, os.path.join(wav_dump, "{}_bits16.wav".format(folder))
            )
        )
        utt2spk.write("{} {}\n".format(utt_id, UTT_PREFIX))
        label_info, pho_info = process_pho_info(
            os.path.join(src_data, folder, "{}.lab".format(folder))
        )
        label_scp.write("{} {}\n".format(utt_id, label_info))
        musicxmlscp.write(
            "{} {}\n".format(
                utt_id, os.path.join(src_data, folder, "{}.musicxml".format(folder))
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Data for Oniku Database")
    parser.add_argument("src_data", type=str, help="source data directory")
    parser.add_argument("train", type=str, help="train set")
    parser.add_argument("dev", type=str, help="development set")
    parser.add_argument("test", type=str, help="test set")
    parser.add_argument("--fs", type=int, help="frame rate (Hz)")
    parser.add_argument(
        "--wav_dump", type=str, default="wav_dump", help="wav dump directory"
    )
    args = parser.parse_args()

    if not os.path.exists(args.wav_dump):
        os.makedirs(args.wav_dump)

    process_subset(args.src_data, args.train, train_check, args.fs, args.wav_dump)
    process_subset(args.src_data, args.dev, dev_check, args.fs, args.wav_dump)
    process_subset(args.src_data, args.test, test_check, args.fs, args.wav_dump)
