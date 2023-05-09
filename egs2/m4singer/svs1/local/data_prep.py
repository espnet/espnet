import argparse
import json
import os
import shutil

import librosa
import miditoolkit
import numpy as np

from espnet2.fileio.score_scp import SingingScoreWriter

"""Generate segments according to structured annotation."""
"""Transfer music score into 'score' format."""

SELECTED_SINGER = [
    "Alto-1",
    "Alto-2",
    "Alto-3",
    "Alto-4",
    "Alto-5",
    "Alto-6",
    "Bass-1",
    "Bass-2",
    "Bass-3",
    "Soprano-1",
    "Soprano-2",
    "Soprano-3",
    "Soprano-4",
    "Tenor-1",
    "Tenor-2",
    "Tenor-3",
    "Tenor-4",
    "Tenor-5",
    "Tenor-6",
]


def makedir(data_url):
    if os.path.exists(data_url):
        shutil.rmtree(data_url)
    os.makedirs(data_url)


def load_midi(args, uid, song_name):
    # Note(Yuning): note duration from '.midi' for M4Singer cannot be used here.
    # We only extract tempo from '.midi'.
    midi_path = os.path.join(args.src_data, "midi", song_name, "{}.mid".format(uid))
    midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)
    tempos = midi_obj.tempo_changes
    tempos.sort(key=lambda x: (x.time, x.tempo))
    assert len(tempos) == 1
    tempo = int(tempos[0].tempo + 0.5)
    return tempo


def create_score(uid, phns, midis, syb_dur, keep):
    # Transfer into 'score' format
    assert len(phns) == len(midis)
    assert len(midis) == len(syb_dur)
    assert len(syb_dur) == len(keep)
    lyrics_seq = []
    midis_seq = []
    segs_seq = []
    phns_seq = []
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
            and keep[index_phn] == 0
        ):
            syb.append(phns[index_phn])
            st += syb_dur[index_phn]
            index_phn += 1
        syb = "_".join(syb)
        note_info.extend([st, syb, midi, syb])
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
    midis = segment["notes_pitch"]
    syb_dur = segment["notes_dur"]
    phn_dur = segment["ph_dur"]
    keep = segment["is_slur"]
    spk = segment["singer"]

    # load tempo from midi
    uid = name.encode("unicode_escape").decode().replace("\\u", "#U")
    song_name = uid[: uid.rindex("#")]
    uid = uid.replace(" ", "+")

    text.write("m4singer_{} {}\n".format(uid, " ".join(phns)))
    utt2spk.write("m4singer_{} {}\n".format(uid, spk))

    # apply bit convert, there is a known issue in direct convert in format wavscp
    cmd = "sox {}.wav -c 1 -t wavpcm -b 16 -r {} {}/m4singer_{}.wav".format(
        os.path.join(audio_dir, song_name.replace(" ", "\\ "), uid.replace("+", "\\ ")),
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
            os.path.join(args.src_data, "wav"),
            args.wav_dumpdir,
            segment,
            tgt_sr=args.sr,
        )


def split_subset(args, meta):
    data = {"tr_no_dev": [], "dev": [], "eval": []}
    for i in range(len(meta)):
        if meta[i]["singer"] in SELECTED_SINGER:
            if i % 100 == 0:
                data["dev"].append(meta[i])
            elif (i + 50) % 100 == 0:
                data["eval"].append(meta[i])
            else:
                data["tr_no_dev"].append(meta[i])
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
