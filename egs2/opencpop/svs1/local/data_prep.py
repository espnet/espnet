import argparse
import os

import librosa
import miditoolkit
import numpy as np

from espnet2.fileio.xml_scp import XMLScpWriter
from espnet2.text.phoneme_tokenizer import pypinyin_g2p_phone_without_prosody


def load_midi_note_scp(midi_note_scp):
    # Note(jiatong): midi format is 0-based
    midi_mapping = {}
    with open(midi_note_scp, "r", encoding="utf-8") as f:
        content = f.read().strip().split("\n")
        for key in content:
            key = key.split("\t")
            midi_mapping[key[1]] = int(key[0])
    return midi_mapping


def load_midi(args):
    midis = {}
    for i in range(2001, 2101):
        midi_path = os.path.join(
            args.src_data, "raw_data", "midis", "{}.midi".format(i)
        )
        midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)
        tempos = midi_obj.tempo_changes
        tempos.sort(key=lambda x: (x.time, x.tempo))
        assert len(tempos) == 1
        tempo = int(tempos[0].tempo + 0.5)
        midis[i] = tempo
    return midis


def create_xml(lyrics, phns, notes, syb_dur, tempo, keep):
    assert len(phns) == len(notes)
    assert len(notes) == len(syb_dur)
    assert len(syb_dur) == len(keep)
    lyrics_seq = []
    notes_seq = []
    segs_seq = []
    index = 0
    st = 0
    et = 0
    while index < len(phns) and notes[index] == 0:
        et += syb_dur[index]
        index += 1
    if st != et:
        lyrics_seq.append("-1")
        notes_seq.append(-1)
        segs_seq.append([st, et])
    for syllable in lyrics:
        et = st
        lyrics_phn = pypinyin_g2p_phone_without_prosody(syllable)
        lyrics_seq.append(syllable)
        notes_seq.append(notes[index])
        for i in range(len(lyrics_phn)):
            assert keep[index + i] == 0
            assert notes[index + i] == notes[index]
            assert syb_dur[index + i] == syb_dur[index]
        et += syb_dur[index]
        segs_seq.append([st, et])
        segs_seq.append([st, et])
        st = et
        index += len(lyrics_phn)
        while index < len(phns) and keep[index] == 1:
            lyrics_seq.append("")
            notes_seq.append(notes[index])
            et += syb_dur[index]
            segs_seq.append([st, et])
            index += 1
            st = et
        while index < len(phns) and notes[index] == 0:
            et += syb_dur[index]
            index += 1
        if st != et:
            lyrics_seq.append("P")
            notes_seq.append(-1)
            segs_seq.append([st, et])
    return lyrics_seq, notes_seq, segs_seq, tempo


def preprocess(phns, notes, syb_dur, phn_dur, keep, min_beat):
    new_phns, new_notes, new_syb_dur, new_phn_dur, new_keep = [], [], [], [], []
    start_index = 0
    st = 0
    while start_index < len(phns) and phns[start_index] in ["SP", "AP"]:
        st += phn_dur[start_index]
        start_index += 1
    end_index = len(phns) - 1
    while end_index >= 0 and phns[end_index] in ["SP", "AP"]:
        end_index -= 1
    index = start_index
    while index <= end_index:
        duration = 0
        merge = 0
        while phns[index] in ["SP", "AP"]:
            duration += phn_dur[index]
            index += 1
        if duration == 0:  # not pause
            if keep[index] == 1 and notes[index] == new_notes[-1]:
                duration = phn_dur[index]
                merge = 1
            else:
                new_phns.append(phns[index])
                new_notes.append(notes[index])
                new_syb_dur.append(syb_dur[index])
                new_phn_dur.append(phn_dur[index])
                new_keep.append(keep[index])
            index += 1
        elif duration >= min_beat:  # add pause
            new_phns.append("P")
            new_notes.append(0)
            new_syb_dur.append(duration)
            new_phn_dur.append(duration)
            new_keep.append(0)
        else:
            merge = 1
        if merge == 1 and duration > 0:  # merge to previous note
            if new_keep[-1] == 1:
                new_syb_dur[-1] += duration
            else:
                if len(new_syb_dur) == 1:
                    new_syb_dur[-1] += duration
                elif new_syb_dur[-1] == new_syb_dur[-2]:
                    new_syb_dur[-1] += duration
                    new_syb_dur[-2] += duration
            new_phn_dur[-1] += duration
    assert (
        len(new_phns) == len(new_notes)
        and len(new_syb_dur) == len(new_notes)
        and len(new_syb_dur) == len(new_phn_dur)
        and len(new_phn_dur) == len(new_keep)
    )
    return new_phns, new_notes, new_syb_dur, new_phn_dur, new_keep, st


def process_utterance(
    xml_scp_writer,
    wavscp,
    text,
    utt2spk,
    label,
    audio_dir,
    wav_dumpdir,
    segment,
    update_segments,
    midi_mapping,
    tempos,
    tgt_sr=24000,
):
    uid, lyrics, phns, notes, syb_dur, phn_dur, keep = segment.strip().split("|")
    phns = phns.split(" ")
    notes = notes.split(" ")
    syb_dur = syb_dur.split(" ")
    phn_dur = phn_dur.split(" ")
    keep = keep.split(" ")

    # load tempo from midi
    id = int(uid[0:4])
    tempo = tempos[id]

    # note to midi index
    notes = [midi_mapping[note] if note != "rest" else 0 for note in notes]

    # type convert
    phn_dur = [float(dur) for dur in phn_dur]
    syb_dur = [float(syb) for syb in syb_dur]
    keep = [int(k) for k in keep]

    min_beat = 1.0 / 4 / (tempo / 60)
    new_phns, new_notes, new_syb_dur, new_phn_dur, new_keep, st = preprocess(
        phns, notes, syb_dur, phn_dur, keep, min_beat
    )

    xml_seq = create_xml(lyrics, new_phns, new_notes, new_syb_dur, tempo, new_keep)
    new_lyrics = xml_seq[0]

    xml_scp_writer["opencpop_{}".format(uid)] = xml_seq
    text.write("opencpop_{} {}\n".format(uid, "".join(new_lyrics)))
    utt2spk.write("opencpop_{} {}\n".format(uid, "opencpop"))

    # apply bit convert, there is a known issue in direct convert in format wavscp
    cmd = f"sox {os.path.join(audio_dir, uid)}.wav -c 1 -t wavpcm -b 16 -r {tgt_sr} {os.path.join(wav_dumpdir, uid)}_bits16.wav"
    os.system(cmd)

    wavscp.write(
        "opencpop_{} {}_bits16.wav\n".format(uid, os.path.join(wav_dumpdir, uid))
    )

    running_dur = 0
    assert len(new_phn_dur) == len(new_phns)
    label_entry = []
    for i in range(len(new_phns)):
        start = running_dur
        end = running_dur + new_phn_dur[i]
        label_entry.append("{:.3f} {:.3f} {}".format(start, end, new_phns[i]))
        running_dur += new_phn_dur[i]

    label.write("opencpop_{} {}\n".format(uid, " ".join(label_entry)))
    update_segments.write(
        "opencpop_{} opencpop_{} {:.3f} {:.3f}\n".format(uid, uid, st, running_dur)
    )


def process_subset(args, set_name, tempos):
    xml_writer = XMLScpWriter(
        args.xml_dumpdir,
        os.path.join(args.tgt_dir, set_name, "musicxml.scp"),
    )
    wavscp = open(
        os.path.join(args.tgt_dir, set_name, "wav.scp"), "w", encoding="utf-8"
    )
    label = open(os.path.join(args.tgt_dir, set_name, "label"), "w", encoding="utf-8")
    text = open(os.path.join(args.tgt_dir, set_name, "text"), "w", encoding="utf-8")
    utt2spk = open(
        os.path.join(args.tgt_dir, set_name, "utt2spk"), "w", encoding="utf-8"
    )
    update_segments = open(
        os.path.join(args.tgt_dir, set_name, "segments"), "w", encoding="utf-8"
    )

    midi_mapping = load_midi_note_scp(args.midi_note_scp)

    with open(
        os.path.join(args.src_data, "segments", set_name + ".txt"),
        "r",
        encoding="utf-8",
    ) as f:
        segments = f.read().strip().split("\n")
        for segment in segments:
            process_utterance(
                xml_writer,
                wavscp,
                text,
                utt2spk,
                label,
                os.path.join(args.src_data, "segments", "wavs"),
                args.wav_dumpdir,
                segment,
                update_segments,
                midi_mapping,
                tempos,
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
        "--xml_dumpdir", type=str, help="xml obj dump directory", default="xml_dump"
    )
    parser.add_argument(
        "--wav_dumpdir", type=str, help="wav dump directoyr (rebit)", default="wav_dump"
    )
    parser.add_argument("--sr", type=int, help="sampling rate (Hz)")
    args = parser.parse_args()

    tempos = load_midi(args)

    for name in ["train", "eval"]:
        process_subset(args, name, tempos)
