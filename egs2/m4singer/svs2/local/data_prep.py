import argparse
import json
import os
import random
import shutil

import librosa
import miditoolkit
import numpy as np

from espnet2.fileio.score_scp import SingingScoreWriter

"""Generate segments according to structured annotation."""
"""Transfer music score into 'score' format."""


def makedir(data_url):
    if os.path.exists(data_url):
        shutil.rmtree(data_url)
    os.makedirs(data_url)


def load_midi(args, uid, song_name):
    # Note(Yuning): note duration from '.midi' for M4Singer cannot be used here.
    # We only extract tempo from '.midi'.
    midi_path = os.path.join(
        args.src_data, song_name, "{}.mid".format(uid.split("#")[-1])
    )
    midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)
    tempos = midi_obj.tempo_changes
    tempos.sort(key=lambda x: (x.time, x.tempo))
    assert len(tempos) == 1
    tempo = int(tempos[0].tempo + 0.5)
    return tempo


def check_split(phns, index_phn):
    # NOTE(Yuxun): fix single syllabel
    if (phns[index_phn - 1] == "i"
        or (phns[index_phn - 1] == "ia") 
        or (phns[index_phn - 1] == "ian")
        or (phns[index_phn - 1] == "iang")
        or (phns[index_phn - 1] == "iao")
        or (phns[index_phn - 1] == "ie")
        or (phns[index_phn - 1] == "in")
        or (phns[index_phn - 1] == "ing") 
        or (phns[index_phn - 1] == "io")
        or (phns[index_phn - 1] == "iou")
        or (phns[index_phn - 1] == "iong")
        or (phns[index_phn - 1] == "iu")
        or (phns[index_phn - 1] == "iuan")
        or (phns[index_phn - 1] == "iue")
        or (phns[index_phn - 1] == "iun")
        or (phns[index_phn - 1] == "v")
        or (phns[index_phn - 1] == "ve")
        or (phns[index_phn - 1] == "van")
        or (phns[index_phn - 1] == "vn")
        or (phns[index_phn - 1] == "o")
        or (phns[index_phn - 1] == "ou")
        or (phns[index_phn - 1] == "a")
        or (phns[index_phn - 1] == "an")
        or (phns[index_phn - 1] == "ao")
        or (phns[index_phn - 1] == "ai")
        or (phns[index_phn - 1] == "e")
        or (phns[index_phn - 1] == "en")
        or (phns[index_phn - 1] == "ei")
        or (phns[index_phn - 1] == "er")
        or (phns[index_phn - 1] == "u")
        or (phns[index_phn - 1] == "ua")
        or (phns[index_phn - 1] == "uo")
        or (phns[index_phn - 1] == "uan")
        or (phns[index_phn - 1] == "uai")
        or (phns[index_phn - 1] == "uang")
        or (phns[index_phn - 1] == "uen")
        or (phns[index_phn - 1] == "ueng")
        or (phns[index_phn - 1] == "uei")
        or (phns[index_phn - 1] == "uan")
        or (phns[index_phn - 1] == "uang")
    ):
        return True
    return False


def create_score(uid, phns, midis, syb_dur, keep):
    # Transfer into 'score' format
    assert len(phns) == len(midis)
    assert len(midis) == len(syb_dur)
    assert len(syb_dur) == len(keep)
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
            and midis[index_phn] == midis[index_phn - 1]
            and midis[index_phn] != 0
            and keep[index_phn] == 0
            and check_split(phns, index_phn) is False
        ):
            syb.append(phns[index_phn])
            st += syb_dur[index_phn]
            index_phn += 1
        lyric_pinyin = "".join(syb)
        syb = "_".join(syb)

        # NOTE(Yuxun): fix phns error
        if uid == "Alto-5##U4fee#U70bc#U7231#U60c5#0016":
            if syb == "x_ang":
                lyric_pinyin = "zhang"
                syb = "zh_ang"
            if syb == "l_e":
                lyric_pinyin = "lian"
                syb = "l_ian"
        if uid == "Alto-5##U660e#U660e#U5c31#0010":
            if syb == "j_ong":
                lyric_pinyin = "zhong"
                syb = "zh_ong"

        note_info.extend([st, lyric_pinyin, midi, syb])
        note_list.append(note_info)
        # multi notes in one syllable
        while (
            index_phn < len(phns)
            and keep[index_phn] == 1
            and phns[index_phn] == phns[index_phn - 1]
        ):
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
    segment,
    tgt_sr=24000,
):
    name = segment["item_name"]
    lyrics = segment["txt"]
    phns = segment["phs"]
    midis = segment["notes"]
    syb_dur = segment["notes_dur"]
    phn_dur = segment["ph_dur"]
    keep = segment["is_slur"]
    spk = name.split("#")[0]

    # load tempo from midi
    uid = name.encode("unicode_escape").decode().replace("\\u", "#U")
    # NOTE(Yuxun): Please check directory name in m4singer
    song_name = name[: name.rindex("#")] # zh case
    # song_name = uid[: uid.rindex("#")] # unicode case
    uid = uid.replace(" ", "+")

    text.write("m4singer_{} {}\n".format(uid, " ".join(phns)))
    utt2spk.write("m4singer_{} m4singer_{}\n".format(uid, spk))

    # apply bit convert, there is a known issue in direct convert in format wavscp
    cmd = "sox {}.wav -c 1 -t wavpcm -b 16 -r {} {}/m4singer_{}.wav".format(
        os.path.join(
            audio_dir,
            song_name.replace(" ", "\\ "),
            uid.replace("+", "\\ ").split("#")[-1],
        ),
        tgt_sr,
        wav_dumpdir,
        uid,
    )
    os.system(cmd)
    wavscp.write("m4singer_{} {}/m4singer_{}.wav\n".format(uid, wav_dumpdir, uid))

    # write annotation duration info into label
    running_dur = 0
    assert len(phn_dur) == len(phns)
    label_entry = []
    for i in range(len(phns)):
        if phns[i] == "<AP>":
            phns[i] = "AP"
        if phns[i] == "<SP>":
            phns[i] = "SP"
        start = running_dur
        end = running_dur + phn_dur[i]
        label_entry.append("{:.3f} {:.3f} {}".format(start, end, phns[i]))
        running_dur += phn_dur[i]
    label.write("m4singer_{} {}\n".format(uid, " ".join(label_entry)))

    # load tempo from midi
    tempo = load_midi(args, uid.replace("+", " "), song_name)

    # prepare music score
    note_list = create_score(uid, phns, midis, syb_dur, keep)
    score = dict(
        tempo=tempo, item_list=["st", "et", "lyric", "midi", "phns"], note=note_list
    )
    writer["m4singer_{}".format(uid)] = score


def process_subset(args, set_name, data):
    makedir(os.path.join(args.tgt_dir, set_name))
    wavscp = open(
        os.path.join(args.tgt_dir, set_name, "wav.scp"), "w", encoding="utf-8"
    )
    label = open(os.path.join(args.tgt_dir, set_name, "label"), "w", encoding="utf-8")
    text = open(os.path.join(args.tgt_dir, set_name, "text"), "w", encoding="utf-8")
    utt2spk = open(
        os.path.join(args.tgt_dir, set_name, "utt2spk"), "w", encoding="utf-8"
    )
    writer = SingingScoreWriter(
        args.score_dump, os.path.join(args.tgt_dir, set_name, "score.scp")
    )

    for segment in data:
        process_utterance(
            writer,
            wavscp,
            text,
            utt2spk,
            label,
            os.path.join(args.src_data),
            args.wav_dumpdir,
            segment,
            tgt_sr=args.sr,
        )


def split_subset(args, meta):
    overall_data = {}
    item_names = []
    for i in range(len(meta)):
        item_name = meta[i]["item_name"]
        overall_data[item_name] = meta[i]
        item_names.append(item_name)

    item_names = sorted(item_names)

    # Refer to https://github.com/M4Singer/M4Singer
    random.seed(1234)
    random.shuffle(item_names)
    valid_num, test_num = 100, 100

    test_names = item_names[:test_num]
    # NOTE(jiatong): the valid set is different from M4Singer
    # As they include test set in the validation set
    # but we do not.
    valid_names = item_names[test_num : valid_num + test_num]
    train_names = item_names[valid_num + test_num :]

    data = {"tr_no_dev": [], "dev": [], "eval": []}

    for key in overall_data.keys():
        if key in test_names:
            data["eval"].append(overall_data[key])
        elif key in valid_names:
            data["dev"].append(overall_data[key])
        else:
            data["tr_no_dev"].append(overall_data[key])
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Data for M4Singer Database")
    parser.add_argument("src_data", type=str, help="source data directory")
    parser.add_argument("--tgt_dir", type=str, default="data")
    parser.add_argument(
        "--wav_dumpdir", type=str, help="wav dump directoyr (rebit)", default="wav_dump"
    )
    parser.add_argument("--sr", type=int, help="sampling rate (Hz)")
    parser.add_argument("--g2p", type=str, help="g2p", default="None")
    parser.add_argument(
        "--score_dump", type=str, default="score_dump", help="score dump directory"
    )
    args = parser.parse_args()

    with open(os.path.join(args.src_data, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    data = split_subset(args, meta)

    for name in ["tr_no_dev", "dev", "eval"]:
        process_subset(args, name, data[name])
