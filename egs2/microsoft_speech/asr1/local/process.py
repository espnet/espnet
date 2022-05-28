import contextlib
import os
import random
import sys
import wave

from tqdm import tqdm

microsoft_speech_corpus_path = sys.argv[1]
lang = sys.argv[2]

train_folder = f"{microsoft_speech_corpus_path}/{lang}-in-Train"
test_folder = f"{microsoft_speech_corpus_path}/{lang}-in-Test"

train_audio_folder = os.path.join(train_folder, "Audios")
dev_audio_folder = train_audio_folder
test_audio_folder = os.path.join(test_folder, "Audios")
train_tr_file = os.path.join(train_folder, "transcription.txt")
test_tr_file = os.path.join(test_folder, "transcription.txt")

train_dst_folder = f"data/train_{lang}"
dev_dst_folder = f"data/dev_{lang}"
test_dst_folder = f"data/test_{lang}"

utt_idx = 1


def get_duration(fname):
    with contextlib.closing(wave.open(fname, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration


def get_dev_split():
    train_audio_files = os.listdir(train_audio_folder)
    dev_split = set()
    # randomly split the training set of 40 hours to 37 train and 3 dev
    dur = 7200
    while dur >= 0:
        f = random.choice(train_audio_files)
        while f in dev_split:
            f = random.choice(train_audio_files)
        wav_file = os.path.join(train_audio_folder, f)
        dev_split.add(f)
        dur -= get_duration(wav_file)
    return dev_split


def get_train_dev_trs():
    dev_split = get_dev_split()
    train_transcriptions = []
    dev_transcriptions = []
    train_fnames = []
    dev_fnames = []
    train_utts = []
    dev_utts = []
    with open(train_tr_file, encoding="utf-8") as f:
        for line in f:
            try:
                line = line.strip()
                fname, text = line.split("\t")
                global utt_idx
                utt_id = f"id_{utt_idx:07d}"
                line = utt_id + " " + text
                fpath = fname + ".wav"
                if fpath in dev_split:
                    dev_fnames.append(fname)
                    dev_utts.append(utt_id)
                    dev_transcriptions.append(line)
                else:
                    train_fnames.append(fname)
                    train_utts.append(utt_id)
                    train_transcriptions.append(line)
                utt_idx += 1
            except Exception as e:
                print(f"Cannot process {line}")
    return (
        train_transcriptions,
        train_fnames,
        train_utts,
        dev_transcriptions,
        dev_fnames,
        dev_utts,
    )


def get_test_trs():
    test_transcriptions = []
    test_fnames = []
    test_utts = []
    with open(test_tr_file, encoding="utf-8") as f:
        for line in f:
            try:
                line = line.strip()
                fname, text = line.split("\t")
                global utt_idx
                utt_id = f"id_{utt_idx:07d}"
                line = utt_id + " " + text
                test_fnames.append(fname)
                test_utts.append(utt_id)
                test_transcriptions.append(line)
                utt_idx += 1
            except Exception as e:
                print(f"Cannot process {line}")
    return test_transcriptions, test_fnames, test_utts


def prepare_files(dest_folder, audio_folder, transcriptions, fnames, utt_ids):
    with open(os.path.join(dest_folder, "text"), "w", encoding="utf-8") as f:
        for line in transcriptions:
            f.write(line)
            f.write("\n")
    with open(os.path.join(dest_folder, "spk2utt"), "w", encoding="utf-8") as f:
        for idx in utt_ids:
            line = idx + " " + idx
            f.write(line)
            f.write("\n")
    with open(os.path.join(dest_folder, "utt2spk"), "w", encoding="utf-8") as f:
        for idx in utt_ids:
            line = idx + " " + idx
            f.write(line)
            f.write("\n")
    with open(os.path.join(dest_folder, "utt2gender"), "w", encoding="utf-8") as f:
        for idx in utt_ids:
            line = idx + " " + "m"
            f.write(line)
            f.write("\n")
    with open(os.path.join(dest_folder, "wav.scp"), "w", encoding="utf-8") as f:
        for (idx, fname) in zip(utt_ids, fnames):
            fpath = os.path.join(audio_folder, fname + ".wav")
            line = idx + " " + fpath
            f.write(line)
            f.write("\n")


(
    train_transcriptions,
    train_fnames,
    train_utts,
    dev_transcriptions,
    dev_fnames,
    dev_utts,
) = get_train_dev_trs()
test_transcriptions, test_fnames, test_utts = get_test_trs()
prepare_files(
    train_dst_folder, train_audio_folder, train_transcriptions, train_fnames, train_utts
)
prepare_files(
    dev_dst_folder, dev_audio_folder, dev_transcriptions, dev_fnames, dev_utts
)
prepare_files(
    test_dst_folder, test_audio_folder, test_transcriptions, test_fnames, test_utts
)
