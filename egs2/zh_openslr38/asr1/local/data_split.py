"""
Split data to train, dev, test
"""
import os
import random
import sys
from collections import defaultdict

train_size = 0.9
random.seed(1)

data_dir = sys.argv[1]  # ST-CMDS-20170001_1-OS

# create speaker id dictionary
d = defaultdict(list)
for fn in os.listdir(data_dir):
    if not fn.endswith(".wav"):
        continue
    # 20170001P00001A0001.wav
    prefix, s = fn.split("P")
    try:
        speaker, s = s.split("A")
        letter = "A"
    except ValueError:
        speaker, s = s.split("I")
        letter = "I"
    utt, _ = s.split(".")
    d[speaker + letter].append(utt)

speaker_ids = list(d.keys())
random.shuffle(speaker_ids)

num_speakers = len(speaker_ids)
assert (
    num_speakers == 855
), "Number of speakers should be 855 in Free ST Chinese Mandarin Corpus."

num_train = int(train_size * num_speakers)
num_test = int((num_speakers - num_train) / 2)

train_speakers = speaker_ids[:num_train]
dev_speakers = speaker_ids[num_train:-num_test]
test_speakers = speaker_ids[-num_test:]

print(
    f"# train: {num_train}, # dev:{num_speakers-num_train-num_test}, # test:{num_test}"
)


def get_transcription(spk_id, utt_id):
    text_fn = get_text_filename(spk_id, utt_id)
    with open(text_fn) as f:
        lines = f.readlines()
    assert len(lines) == 1, f"More than one line in transription file:{text_fn}"
    return lines[0]


def get_text_filename(spk_id, utt_id):
    return f"{data_dir}/20170001P{spk_id}{utt_id}.txt"


def get_wav_filename(spk_id, utt_id):
    return f"{data_dir}/20170001P{spk_id}{utt_id}.wav"


def create_files(speakers, directory):
    text_lines, scp_lines, utt2spk_lines = [], [], []
    for spk_id in speakers:
        for utt_id in d[spk_id]:
            # add spk_id in front to make utt_id unique
            unique_utt_id = spk_id + utt_id

            transcription = get_transcription(spk_id, utt_id)
            text_lines.append(f"{unique_utt_id} {transcription}\n")

            wav_file_path = get_wav_filename(spk_id, utt_id)
            scp_lines.append(f"{unique_utt_id} {wav_file_path}\n")

            utt2spk_lines.append(f"{unique_utt_id} {spk_id}\n")

    # sort
    text_lines.sort()
    scp_lines.sort()
    utt2spk_lines.sort()

    # write to file
    with open(f"{directory}/text", "w+") as text_file:
        text_file.writelines(text_lines)

    with open(f"{directory}/wav.scp", "w+") as scp_file:
        scp_file.writelines(scp_lines)

    with open(f"{directory}/utt2spk", "w+") as utt2spk_file:
        utt2spk_file.writelines(utt2spk_lines)


print("Creating files for train...", end="")
create_files(train_speakers, "data/train")
print("Done.")

print("Creating files for dev...", end="")
create_files(dev_speakers, "data/dev")
print("Done.")

print("Creating files for test...", end="")
create_files(test_speakers, "data/test")
print("Done.")
