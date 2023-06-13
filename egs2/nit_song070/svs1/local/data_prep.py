import argparse
import os
import librosa
import numpy as np
import soundfile as sf

from espnet2.fileio.score_scp import SingingScoreWriter


DEV_LIST = ["f001_003"]
TEST_LIST = ["f001_004", "f001_007", "f001_010"]


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


def pack_zero(string, size=20):
    if len(string) < size:
        string = "0" * (size - len(string)) + string
    return string


def load_midi_note_scp(midi_note_scp):
    # Note(jiatong): midi format is 0-based
    midi_mapping = {}
    with open(midi_note_scp, "r", encoding="utf-8") as f:
        content = f.read().strip().split("\n")
        for key in content:
            key = key.split("\t")
            if "/" in key[1]:
                multi_note = key[1].split("/")
                midi_mapping[multi_note[0]] = int(key[0])
                midi_mapping[multi_note[1]] = int(key[0])
            else:
                midi_mapping[key[1]] = int(key[0])
    return midi_mapping


def create_midi(notes, tempo, duration, sr):
    # we assume the tempo is static tempo with only one value
    assert len(notes) == len(duration)
    note_info = []
    for i in range(len(notes)):
        note_dur = int(duration[i] * sr + 0.5)
        note_info.extend(note_dur * [notes[i]])
    tempo_info = [tempo] * len(note_info)
    return note_info, tempo_info

def create_score(label_id, phns, midis, syb_dur):
    # Transfer into 'score' format
    assert len(phns) == len(midis)
    assert len(midis) == len(syb_dur)
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
            and syb_dur[index_phn] == syb_dur[index_phn - 1]
            and midis[index_phn] == midis[index_phn - 1]
        ):
            syb.append(phns[index_phn])
            index_phn += 1
        syb = "_".join(syb)
        note_info.extend([st, syb, midi, syb])
        note_list.append(note_info)
        # multi notes in one syllable
        while (
            index_phn < len(phns)
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
    label_id,
    label_file,
    mono_file,
    midi_mapping,
    tgt_sr=24000,
):
    labels = open(label_file, "r", encoding="utf-8")
    mono_label = open(mono_file, "r", encoding="utf-8")
    label_info = labels.read().split("\n")
    mono_info = mono_label.read().split("\n")
    prev_end = 0
    phns, notes, phn_dur = [], [], []
    for i in range(len(label_info)):
        note_label = label_info[i].split(" ")
        note_mono = mono_info[i].split()
        if len(note_label) != 3 or len(note_mono) != 3:
            print("mismatch note_label or note_mono at line {} in {}".format(i + 1, label_id))
            print(note_label, note_mono)
            continue
        start = int(note_mono[0])
        end = int(note_mono[1])
        if start < prev_end:
            continue
        detailed_info = note_label[2].split("E:")
        # assert len(detailed_info) > 2, "{}".format(label_id)
        if len(detailed_info) < 2:
            print("skip {} for no note label".format(label_id))
            continue
        aligned_note = detailed_info[1].split("]")[0]
        if aligned_note == "xx":
            notes.append("rest")
        else:
            notes.append(aligned_note)
        phns.append(note_mono[2])
        phn_dur.append(end - start)
        
    y, sr = sf.read(os.path.join(audio_dir, label_id) + ".raw", subtype="PCM_16", channels=1, samplerate=48000, endian="LITTLE")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # estimate a static tempo for midi format
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    tempo = int(tempo)

    # note to midi index
    notes = [midi_mapping[note] if note != "rest" else 0 for note in notes]

    # duration type convert
    phn_dur = [float(dur/ 1e7) for dur in phn_dur]

    note_list = create_score(label_id, phns, notes, phn_dur)

    text.write("nit_song070_{} {}\n".format(label_id, " ".join(phns)))
    utt2spk.write("nit_song070_{} {}\n".format(label_id, "onit_song070"))

    # apply bit convert, there is a known issue in direct convert in format wavscp
    cmd = f"sox -t raw -r 48000 -b 16 -c 1 -L -e signed-integer {os.path.join(audio_dir, label_id)}.raw -c 1 -t wavpcm -b 16 -r {tgt_sr} {os.path.join(wav_dumpdir, label_id)}_bits16.wav"
    os.system(cmd)

    wavscp.write("nit_song070_{} {}_bits16.wav\n".format(label_id, os.path.join(wav_dumpdir, label_id)))

    running_dur = 0
    assert len(phn_dur) == len(phns)
    label_entry = []
    for i in range(len(phns)):
        start = running_dur
        end = running_dur + phn_dur[i]
        label_entry.append("{:.3f} {:.3f} {}".format(start, end, phns[i]))
        running_dur += phn_dur[i]

    label.write("nit_song070_{} {}\n".format(label_id, " ".join(label_entry)))
    score = dict(
        tempo=tempo, item_list=["st", "et", "lyric", "midi", "phns"], note=note_list
    )
    writer["nit_song070_{}".format(label_id)] = score


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

    midi_mapping = load_midi_note_scp(args.midi_note_scp)

    for label_file in os.listdir(os.path.join(args.src_data, "labels", "full")):
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
            os.path.join(args.src_data, "raw"),
            args.wav_dumpdir,
            label_id,
            os.path.join(args.src_data, "labels", "full", label_file),
            os.path.join(args.src_data, "labels", "mono", label_file),
            midi_mapping,
            tgt_sr=args.sr,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Data for Opencpop Database")
    parser.add_argument("src_data", type=str, help="source data directory")
    parser.add_argument("--tgt_dir", type=str, default="data")
    parser.add_argument(
        "--midi_note_scp",
        type=str,
        help="midi note scp for information of note id",
        default="local/midi-note.scp",
    )
    parser.add_argument(
        "--score_dump", type=str, default="score_dump", help="score dump directory"
    )
    parser.add_argument(
        "--wav_dumpdir", type=str, help="wav dump directoyr (rebit)", default="wav_dump"
    )
    parser.add_argument("--sr", type=int, help="sampling rate (Hz)")
    args = parser.parse_args()

    for name in ["train", "dev", "eval"]:
        process_subset(args, name)
