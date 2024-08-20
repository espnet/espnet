import sys
import pandas as pd
import numpy as np


def split_speakers(transcripts, train_ratio, dev_ratio, test_ratio):
    # get the groups then divide the speakers
    spk_train, spk_dev, spk_test = set(), set(), set()

    for group_key, group_df in transcripts.groupby(['location', 'age_group', 'gender']):
        # we don't have enough speakers in each group to divide by socioeconomic status
        speakers = list(group_df['speaker_id'].unique())
        # print(group_key, 'speakers', len(speakers))

        # decide how many speakers in each group
        num_train, num_dev, _ = np.random.multinomial(len(speakers), [train_ratio, dev_ratio, test_ratio], size=1).squeeze()
        # add speakers into the groups
        np.random.shuffle(speakers)
        spk_train.update(speakers[:num_train])
        spk_dev.update(speakers[num_train:num_train + num_dev])
        spk_test.update(speakers[num_train + num_dev:])
        for spk in speakers:
            assert spk in (spk_train | spk_dev | spk_test)

    # ensure different speakers across different splits (no overlap)
    assert len(spk_train & spk_dev) == 0
    assert len(spk_train & spk_test) == 0
    assert len(spk_dev & spk_test) == 0

    return spk_train, spk_dev, spk_test


def split_files(transcripts, spk_train, spk_dev, spk_test):
    train_rows = transcripts[transcripts['speaker_id'].isin(spk_train)]
    dev_rows = transcripts[transcripts['speaker_id'].isin(spk_dev)]
    test_rows = transcripts[transcripts['speaker_id'].isin(spk_test)]

    train_wav = set(train_rows['basefile'])
    dev_wav = set(dev_rows['basefile'])
    test_wav = set(test_rows['basefile'])
    assert len(train_wav & dev_wav) == 0
    assert len(train_wav & test_wav) == 0
    assert len(dev_wav & test_wav) == 0

    train_utt = set(train_rows['segment_filename'])
    dev_utt = set(dev_rows['segment_filename'])
    test_utt = set(test_rows['segment_filename'])
    assert len(train_utt & dev_utt) == 0
    assert len(train_utt & test_utt) == 0
    assert len(dev_utt & test_utt) == 0

    return (train_wav, dev_wav, test_wav), (train_utt, dev_utt, test_utt)


if __name__ == '__main__':
    if len(sys.argv) != 8:
        print("Help: python local/train_dev_test_split.py <path_to_transcript> <train_split> <dev_split> <test_split> <train_ratio> <dev_ratio> <test_ratio>")
        print("ex: python local/train_dev_test_split.py downloads/transcript.tsv downloads/train downloads/dev downloads/test 0.8 0.1 0.1")
        print("Note: This script assumes transcript.tsv contains age group, gender, and socioeconomic class")
        exit(1)
    path_to_transcript = sys.argv[1]
    train_file, dev_file, test_file = sys.argv[2:5]
    train_ratio, dev_ratio, test_ratio = sys.argv[5:]
    train_ratio, dev_ratio, test_ratio = float(train_ratio), float(dev_ratio), float(test_ratio)
    assert round(train_ratio + dev_ratio + test_ratio) == 1

    np.random.seed(15213)

    transcripts = pd.read_csv(path_to_transcript, sep='\t')

    spk_train, spk_dev, spk_test = split_speakers(transcripts, train_ratio, dev_ratio, test_ratio)
    num_spk = len(spk_train | spk_dev | spk_test)
    print('spk distribution: train/dev/test', len(spk_train) / num_spk, len(spk_dev) / num_spk, len(spk_test) / num_spk)

    (train_wav, dev_wav, test_wav), (train_utt, dev_utt, test_utt) = split_files(transcripts, spk_train, spk_dev, spk_test)
    with open(train_file + '_utt.list', 'w') as f:
        for utt in train_utt:
            f.write(utt + '\n')
    with open(dev_file + '_utt.list', 'w') as f:
        for utt in dev_utt:
            f.write(utt + '\n')
    with open(test_file + '_utt.list', 'w') as f:
        for utt in test_utt:
            f.write(utt + '\n')
    with open(train_file + '_wav.list', 'w') as f:
        for utt in train_wav:
            f.write(utt + '\n')
    with open(dev_file + '_wav.list', 'w') as f:
        for utt in dev_wav:
            f.write(utt + '\n')
    with open(test_file + '_wav.list', 'w') as f:
        for utt in test_wav:
            f.write(utt + '\n')


# http://lingtools.uoregon.edu/coraal/userguide/CORAALUserGuide_current.pdf
# DCA_se2_ag1_m_05_1
# location. socioeconomic group (0 = no splits). age group. gender. speaker number. audio file number.
# age groups inconsistent across locations (page 38), ex: 29 and under ("-29") vs 19 and under ("-19")
# many locations do not mark socioeconomic group
