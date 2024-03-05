"""
Reference from ESPnet's egs2/nit_song070/svs1/local/data_prep.py
https://github.com/espnet/espnet/blob/master/egs2/nit_song070/svs1/local/data_prep.py
"""

import argparse
import os
import re

import librosa
import numpy as np
import soundfile as sf

from espnet2.fileio.score_scp import SingingScoreWriter

DEV_LIST = ["003"]
TEST_LIST = ["004", "007", "010"]


def train_check(song):
    return not test_check(song) and not dev_check(song)


def dev_check(song):
    for dev in DEV_LIST:
        if dev in song:
            return True
    return False


def test_check(song):
    for test in TEST_LIST:
        if test in song:
            return True
    return False


def pack_zero(string, size=4):
    if len(string) < size:
        string = "0" * (size - len(string)) + string
    return string


def create_score(phns, midis, syb_dur):
    # Transfer into 'score' format
    assert len(phns) == len(midis)
    assert len(midis) == len(syb_dur)
    st = 0
    index_phn = 0
    note_list = []
    while index_phn < len(phns):
        midi = midis[index_phn]
        note_info = [st]
        st += syb_dur[index_phn]
        syb = [phns[index_phn]]
        index_phn += 1
        if (
            index_phn < len(phns)
            and syb_dur[index_phn] == syb_dur[index_phn - 1]
            and midis[index_phn] == midis[index_phn - 1]
        ):
            syb.append(phns[index_phn])
            index_phn += 1
        syb = "_".join(syb)
        note_info.extend([st, syb, midi, syb])
        note_list.append(note_info)
        # multi notes in one syllable
        while index_phn < len(phns) and phns[index_phn] == phns[index_phn - 1]:
            note_info = [st]
            st += syb_dur[index_phn]
            note_info.extend([st, "—", midis[index_phn], phns[index_phn]])
            note_list.append(note_info)
            index_phn += 1
    return note_list


def process_utterance(
    writer,
    wavscp,
    text,
    utt2spk,
    label,
    audio_dir,
    wav_dumpdir,
    label_id,
    label_file,
    tgt_sr=24000,
):
    labels = open(label_file, "r", encoding="utf-8")
    label_info = labels.read().split("\n")
    prev_end = 0
    phns, notes, phn_dur = [], [], []
    seg_start = 0
    seg_id = 0
    seg_len_phns = 0
    segs = []  # store (seg_id, seg_start, seg_end, seg_len_phns)

    sil = ["pau", "sil"]
    sil = set(sil)

    pattern = re.compile(
        r"([a-z])@([a-zA-Z]+)\^([a-zA-Z]+)-([a-zA-Z]+)\+([a-zA-Z]+)=([a-zA-Z]+)_"
    )  # extract phones from the start of the string

    for i in range(len(label_info)):
        note_label = label_info[i].split(" ")
        start = int(note_label[0])
        end = int(note_label[1])
        if start < prev_end:
            continue
        detailed_info = note_label[2].split("E:")
        # assert len(detailed_info) > 2, "{}".format(label_id)
        if len(detailed_info) < 2:
            print("skip {} for no note label".format(label_id))
            continue
        aligned_note = detailed_info[1].split("]")[0]
        phones = re.findall(pattern, detailed_info[0])
        assert len(phones) == 1, (
            f"len(phones)={len(phones)} is not 1. "
            f"Find more or no patterns from {detailed_info[0]}"
        )
        phn = phones[0][3]
        if phn in sil:
            if seg_len_phns > 0:
                segs.append([pack_zero(str(seg_id)), seg_start, end, seg_len_phns])
                seg_id += 1
            seg_start = end
            seg_len_phns = 0
            continue
        if aligned_note == "xx":
            notes.append(0)  # rest
        else:
            assert (
                aligned_note.isdigit()
            ), f"aligned_note={aligned_note} is not an integer"
            aligned_note = int(aligned_note)
            notes.append(aligned_note)
        phns.append(phn)
        phn_dur.append(end - start)
        seg_len_phns += 1

    start_phn_index = 0
    print(label_id)
    for seg in segs:
        print(seg)
        y, sr = sf.read(
            os.path.join(audio_dir, label_id) + ".wav",
            start=int(seg[1] * 48000 * 16 // 1e7),
            stop=int(seg[2] * 48000 * 16 // 1e7),
        )
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        # estimate a static tempo for midi format
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        tempo = int(tempo)

        # note to midi index
        seg_notes = notes[start_phn_index : start_phn_index + seg[3]]

        # duration type convert
        seg_phn_dur = [
            float(dur / 1e7)
            for dur in phn_dur[start_phn_index : start_phn_index + seg[3]]
        ]

        note_list = create_score(
            phns[start_phn_index : start_phn_index + seg[3]],
            seg_notes,
            seg_phn_dur,
        )

        text.write(
            "jsut_song_{}_{} {}\n".format(
                label_id,
                seg[0],
                " ".join(phns[start_phn_index : start_phn_index + seg[3]]),
            )
        )
        utt2spk.write("jsut_song_{}_{} {}\n".format(label_id, seg[0], "jsut_song"))

        # apply bit convert, there is a known issue in direct convert in format wavscp
        cmd = (
            f"sox -t wav -r 48000 -b 16 -c 1 -L -e signed-integer "
            f"{os.path.join(audio_dir, label_id)}.wav -c 1 -t wavpcm -b 16 -r {tgt_sr} "
            f"{os.path.join(wav_dumpdir, 'jsut_song_'+label_id+'_'+seg[0])}.wav "
            f"trim {float(seg[1] / 1e7)} {float((seg[2]-seg[1]) / 1e7)}"
        )
        os.system(cmd)

        wavscp.write(
            "jsut_song_{}_{} {}.wav\n".format(
                label_id,
                seg[0],
                os.path.join(wav_dumpdir, "jsut_song_" + label_id + "_" + seg[0]),
            )
        )

        running_dur = 0
        assert (
            len(seg_phn_dur)
            == len(phns[start_phn_index : start_phn_index + seg[3]])
            == seg[3]
        )
        label_entry = []
        for i in range(seg[3]):
            start = running_dur
            end = running_dur + seg_phn_dur[i]
            label_entry.append(
                "{:.3f} {:.3f} {}".format(start, end, phns[start_phn_index + i])
            )
            running_dur += seg_phn_dur[i]

        label.write(
            "jsut_song_{}_{} {}\n".format(label_id, seg[0], " ".join(label_entry))
        )
        score = dict(
            tempo=tempo, item_list=["st", "et", "lyric", "midi", "phns"], note=note_list
        )
        writer["jsut_song_{}_{}".format(label_id, seg[0])] = score
        start_phn_index += seg[3]


def process_subset(args, set_name):
    writer = SingingScoreWriter(
        args.score_dump, os.path.join(args.tgt_dir, set_name, "score.scp")
    )
    wavscp = open(
        os.path.join(args.tgt_dir, set_name, "wav.scp"), "w", encoding="utf-8"
    )
    label = open(os.path.join(args.tgt_dir, set_name, "label"), "w", encoding="utf-8")
    text = open(os.path.join(args.tgt_dir, set_name, "text"), "w", encoding="utf-8")
    utt2spk = open(
        os.path.join(args.tgt_dir, set_name, "utt2spk"), "w", encoding="utf-8"
    )

    for label_file in os.listdir(args.lab_srcdir):
        if not label_file.endswith(".lab"):
            continue
        if set_name == "train" and not train_check(label_file):
            continue
        elif set_name == "eval" and not test_check(label_file):
            continue
        elif set_name == "dev" and not dev_check(label_file):
            continue

        label_id = label_file.split(".")[0]
        process_utterance(
            writer,
            wavscp,
            text,
            utt2spk,
            label,
            args.wav_srcdir,
            args.wav_dumpdir,
            label_id,
            os.path.join(args.lab_srcdir, label_file),
            tgt_sr=args.sr,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Data for JSUT-Song Database")
    parser.add_argument("--lab_srcdir", type=str, help="label source directory")
    parser.add_argument("--wav_srcdir", type=str, help="wav source directory")
    parser.add_argument("--tgt_dir", type=str, default="data")
    parser.add_argument(
        "--score_dump", type=str, default="score_dump", help="score dump directory"
    )
    parser.add_argument(
        "--wav_dumpdir", type=str, help="wav dump directoy (rebit)", default="wav_dump"
    )
    parser.add_argument("--sr", type=int, help="sampling rate (Hz)")
    args = parser.parse_args()

    for name in ["train", "dev", "eval"]:
        process_subset(args, name)
