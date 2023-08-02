import argparse
import os
import shutil

UTT_PREFIX = "namine"
DEV_LIST = [
    "subetenovocaloidnisensenfukoku",
    "Choir",
    "shumatsutokiseki",
    "gunjoirono",
    "dokomademomukuni",
]
TEST_LIST = ["hoshifuru", "hanayorikimini", "yume_shigure", "haruwasakura", "reverse"]


def train_check(song):
    return (song not in DEV_LIST) and (song not in TEST_LIST)


def dev_check(song):
    return song in DEV_LIST


def test_check(song):
    return song in TEST_LIST


def pack_zero(string, size=20):
    if len(string) < size:
        string = "0" * (size - len(string)) + string
    return string


def makedir(data_url):
    if os.path.exists(data_url):
        shutil.rmtree(data_url)

    os.makedirs(data_url)


def process_text_info(text):
    info = open(text, "r", encoding="utf-8")
    label_info = []
    text_info = []
    for line in info.readlines():
        line = line.strip().split()
        label_info.append(
            "{} {} {}".format(
                float(line[0]) / 1e7, float(line[1]) / 1e7, line[2].strip()
            )
        )
        text_info.append(line[2].strip())
    return " ".join(label_info), " ".join(text_info)


def process_subset(src_data, subset, check_func, fs, wav_dump):
    src_data = os.path.join(src_data, "「波音リツ」歌声データベースVer2/DATABASE")
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
        utt_id = "{}_{}".format(UTT_PREFIX, pack_zero(folder))

        cmd = "sox {}.wav -c 1 -t wavpcm -b 16 -r {} {}_bits16.wav".format(
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
        label_info, text_info = process_text_info(
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
