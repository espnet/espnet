import sys

import numpy as np
import pandas as pd


def split_speakers(transcripts, train_ratio, dev_ratio, test_ratio):
    # get the groups then divide the speakers
    spk_train, spk_dev, spk_test = set(), set(), set()

    for group_key, group_df in transcripts.groupby(["location", "age_group", "gender"]):
        # we don't have enough speakers in each group to divide by socioeconomic status
        speakers = list(group_df["speaker_id"].unique())
        # print(group_key, 'speakers', len(speakers))

        # decide how many speakers in each group
        num_train, num_dev, _ = np.random.multinomial(
            len(speakers), [train_ratio, dev_ratio, test_ratio], size=1
        ).squeeze()
        # add speakers into the groups
        np.random.shuffle(speakers)
        spk_train.update(speakers[:num_train])
        spk_dev.update(speakers[num_train : num_train + num_dev])
        spk_test.update(speakers[num_train + num_dev :])
        for spk in speakers:
            assert spk in (spk_train | spk_dev | spk_test)

    # ensure different speakers across different splits (no overlap)
    assert len(spk_train & spk_dev) == 0
    assert len(spk_train & spk_test) == 0
    assert len(spk_dev & spk_test) == 0

    return spk_train, spk_dev, spk_test


def split_files(transcripts, spk_train, spk_dev, spk_test):
    train_rows = transcripts[transcripts["speaker_id"].isin(spk_train)]
    dev_rows = transcripts[transcripts["speaker_id"].isin(spk_dev)]
    test_rows = transcripts[transcripts["speaker_id"].isin(spk_test)]

    train_wav = set(train_rows["basefile"])
    dev_wav = set(dev_rows["basefile"])
    test_wav = set(test_rows["basefile"])
    assert len(train_wav & dev_wav) == 0
    assert len(train_wav & test_wav) == 0
    assert len(dev_wav & test_wav) == 0

    train_utt = set(train_rows["segment_filename"])
    dev_utt = set(dev_rows["segment_filename"])
    test_utt = set(test_rows["segment_filename"])
    assert len(train_utt & dev_utt) == 0
    assert len(train_utt & test_utt) == 0
    assert len(dev_utt & test_utt) == 0

    return (
        (train_wav, dev_wav, test_wav),
        (train_utt, dev_utt, test_utt),
        (train_rows, dev_rows, test_rows),
    )


def generate_utt_list(split_file_path, split_utt):
    with open(split_file_path + "_utt.list", "w") as f:
        for utt in split_utt:
            f.write(utt + "\n")


def generate_wav_list(split_file_path, split_wav):
    with open(split_file_path + "_wav.list", "w") as f:
        for utt in split_wav:
            f.write(utt + "\n")


def generate_utt2spk(split_file_path, rows):
    # utt2spk (<utterance_id> <speaker_id>)
    rows[["segment_filename", "speaker_id"]].to_csv(
        split_file_path + ".utt2spk", sep=" ", index=False, header=None
    )


def generate_text(split_file_path, rows):
    # text (<utterance_id> <transcription>)
    rows[["segment_filename", "normalized_text"]].to_csv(
        split_file_path + ".text", sep=" ", index=False, header=None
    )
    # note: need to use sed to remove quotes later


def generate_segments(split_file_path, rows):
    # segments (<utterance_id> <wav_id> <start_time> <end_time>)
    rows[["segment_filename", "basefile", "start_time", "end_time"]].to_csv(
        split_file_path + ".segments", sep=" ", header=False, index=False
    )


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print(
            "Help: python local/train_dev_test_split.py <path_to_transcript> "
            "<train_split> <dev_split> <test_split> "
            "<train_ratio> <dev_ratio> <test_ratio>"
        )
        print(
            "ex: python local/train_dev_test_split.py "
            "downloads/transcript.tsv downloads/train downloads/dev "
            "downloads/test 0.8 0.1 0.1"
        )
        print(
            "Note: This script assumes transcript.tsv contains "
            "age group, gender, and socioeconomic class"
        )
        exit(1)
    path_to_transcript = sys.argv[1]
    train_file, dev_file, test_file = sys.argv[2:5]
    train_ratio, dev_ratio, test_ratio = sys.argv[5:]
    train_ratio, dev_ratio, test_ratio = (
        float(train_ratio),
        float(dev_ratio),
        float(test_ratio),
    )
    assert round(train_ratio + dev_ratio + test_ratio) == 1

    np.random.seed(15213)

    transcripts = pd.read_csv(path_to_transcript, sep="\t")

    spk_train, spk_dev, spk_test = split_speakers(
        transcripts, train_ratio, dev_ratio, test_ratio
    )
    num_spk = len(spk_train | spk_dev | spk_test)
    print(
        "spk distribution: train/dev/test",
        len(spk_train) / num_spk,
        len(spk_dev) / num_spk,
        len(spk_test) / num_spk,
    )

    (
        (train_wav, dev_wav, test_wav),
        (train_utt, dev_utt, test_utt),
        (train_rows, dev_rows, test_rows),
    ) = split_files(transcripts, spk_train, spk_dev, spk_test)
    # utt, wav lists
    generate_utt_list(train_file, train_utt)
    generate_utt_list(dev_file, dev_utt)
    generate_utt_list(test_file, test_utt)
    generate_wav_list(train_file, train_wav)
    generate_wav_list(dev_file, dev_wav)
    generate_wav_list(test_file, test_wav)

    # utt2spk
    generate_utt2spk(train_file, train_rows)
    generate_utt2spk(dev_file, dev_rows)
    generate_utt2spk(test_file, test_rows)
    # text
    generate_text(train_file, train_rows)
    generate_text(dev_file, dev_rows)
    generate_text(test_file, test_rows)
    # segments
    generate_segments(train_file, train_rows)
    generate_segments(dev_file, dev_rows)
    generate_segments(test_file, test_rows)


# http://lingtools.uoregon.edu/coraal/userguide/CORAALUserGuide_current.pdf
# DCA_se2_ag1_m_05_1
# location. socioeconomic group (0 = no splits). age group.
#   gender. speaker number. audio file number.
# age groups inconsistent across locations (page 38)
#   ex: 29 and under ("-29") vs 19 and under ("-19")
# many locations do not mark socioeconomic group
