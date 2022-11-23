import argparse
import os

import librosa
import miditoolkit
import numpy as np

# from espnet2.fileio.xml_scp import XMLScpWriter
from espnet2.text.build_tokenizer import build_tokenizer

# from espnet2.text.phoneme_tokenizer import pypinyin_g2p_phone_without_prosody


customed_dic = {
    "最长的相拥谁爱得太自由": [
        "z_ui",
        "ch_ang",
        "d_e",
        "x_iang",
        "y_ong",
        "sh_ui",
        "ai",
        "d_e",
        "t_ai",
        "z_i",
        "y_ou",
    ],
    "如小雪落下海岸线": ["r_u", "x_iao", "x_ve", "l_uo", "x_ia", "h_ai", "an", "x_ian"],
}


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


"""
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
"""


def create_score(uid, lyrics, phns, notes, syb_dur, keep):
    assert len(phns) == len(notes)
    assert len(notes) == len(syb_dur)
    assert len(syb_dur) == len(keep)
    tokenizer = build_tokenizer(
        token_type="phn",
        bpemodel=None,
        delimiter=None,
        space_symbol="<space>",
        non_linguistic_symbols=None,
        g2p_type=args.g2p,
    )
    lyrics_seq = []
    notes_seq = []
    segs_seq = []
    phns_seq = []
    st = 0
    index_phn = 0
    index_syb = 0
    text = "".join(lyrics)
    if text in customed_dic:
        text_phns = customed_dic[text]
    else:
        text_phns = tokenizer.g2p("".join(lyrics))
    # text_phns = "_".join(text_phns).strip(" ").split("_ _")
    # print(text_phns)
    while index_phn < len(phns):
        if phns[index_phn] in ["SP", "AP"]:
            lyrics_seq.append(phns[index_phn])
            notes_seq.append(notes[index_phn])
            segs_seq.append([st, st + syb_dur[index_phn]])
            st += syb_dur[index_phn]
            phns_seq.append(phns[index_phn])
            index_phn += 1
        else:
            lyrics_seq.append(lyrics[index_syb])
            notes_seq.append(notes[index_phn])
            segs_seq.append([st, st + syb_dur[index_phn]])
            st += syb_dur[index_phn]
            phns_seq.append(text_phns[index_syb])
            phones = text_phns[index_syb].split("_")
            for k in range(len(phones)):
                if phones[k] != phns[index_phn + k]:
                    raise ValueError(
                        "Mismatch between syllable [{}]->{} and {}-th phonemes ['{}'] in {}. ".format(
                            lyrics[index_syb],
                            phones,
                            index_phn + k,
                            phns[index_phn + k],
                            uid,
                        )
                    )
                assert notes[index_phn + k] == notes[index_phn]
                assert syb_dur[index_phn + k] == syb_dur[index_phn]
            index_phn += len(phones)
            index_syb += 1
            while index_phn < len(phns) and keep[index_phn] == 1:
                lyrics_seq.append("—")
                notes_seq.append(notes[index_phn])
                segs_seq.append([st, st + syb_dur[index_phn]])
                st += syb_dur[index_phn]
                phns_seq.append(phns[index_phn])
                index_phn += 1
    if index_syb < len(lyrics):
        raise ValueError("Syllables are shorter than phonemes in {}: ".format(uid))

    return lyrics_seq, notes_seq, segs_seq, phns_seq


"""
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
"""


def process_utterance(
    wavscp,
    text,
    utt2spk,
    label,
    audio_dir,
    wav_dumpdir,
    segment,
    score,
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

    # min_beat = 1.0 / 4 / (tempo / 60)
    # new_phns, new_notes, new_syb_dur, new_phn_dur, new_keep, st = preprocess(
    #    phns, notes, syb_dur, phn_dur, keep, min_beat
    # )

    # xml_seq = create_xml(lyrics, new_phns, new_notes, new_syb_dur, tempo, new_keep)
    lyrics_seq, notes_seq, segs_seq, phns_seq = create_score(
        uid, lyrics, phns, notes, syb_dur, keep
    )

    # xml_scp_writer["opencpop_{}".format(uid)] = xml_seq
    text.write("opencpop_{} {}\n".format(uid, " ".join(lyrics_seq)))
    utt2spk.write("opencpop_{} {}\n".format(uid, "opencpop"))

    # apply bit convert, there is a known issue in direct convert in format wavscp
    cmd = "sox {}.wav -c 1 -t wavpcm -b 16 -r {} {}_bits16.wav".format(
        os.path.join(audio_dir, uid),
        tgt_sr,
        os.path.join(wav_dumpdir, uid),
    )
    os.system(cmd)

    wavscp.write(
        "opencpop_{} {}_bits16.wav\n".format(uid, os.path.join(wav_dumpdir, uid))
    )

    running_dur = 0
    assert len(phn_dur) == len(phns)
    label_entry = []
    for i in range(len(phns)):
        start = running_dur
        end = running_dur + phn_dur[i]
        label_entry.append("{:.3f} {:.3f} {}".format(start, end, phns[i]))
        running_dur += phn_dur[i]

    label.write("opencpop_{} {}\n".format(uid, " ".join(label_entry)))
    # update_segments.write(
    #    "opencpop_{} opencpop_{} {:.3f} {:.3f}\n".format(uid, uid, st, running_dur)
    # )

    score.write("opencpop_{}  {}".format(uid, tempo))
    for i in range(len(lyrics_seq)):
        score.write(
            "  {:.3f} {:.3f} {} {} {}".format(
                segs_seq[i][0], segs_seq[i][1], lyrics_seq[i], notes_seq[i], phns_seq[i]
            )
        )
    score.write("\n")


def process_subset(args, set_name, tempos):
    # xml_writer = XMLScpWriter(
    #    args.xml_dumpdir,
    #    os.path.join(args.tgt_dir, set_name, "musicxml.scp"),
    # )
    wavscp = open(
        os.path.join(args.tgt_dir, set_name, "wav.scp"), "w", encoding="utf-8"
    )
    label = open(os.path.join(args.tgt_dir, set_name, "label"), "w", encoding="utf-8")
    score = open(os.path.join(args.tgt_dir, set_name, "score"), "w", encoding="utf-8")
    text = open(os.path.join(args.tgt_dir, set_name, "text"), "w", encoding="utf-8")
    utt2spk = open(
        os.path.join(args.tgt_dir, set_name, "utt2spk"), "w", encoding="utf-8"
    )
    # update_segments = open(
    #    os.path.join(args.tgt_dir, set_name, "segments"), "w", encoding="utf-8"
    # )

    midi_mapping = load_midi_note_scp(args.midi_note_scp)

    with open(
        os.path.join(args.src_data, "segments", set_name + ".txt"),
        "r",
        encoding="utf-8",
    ) as f:
        segments = f.read().strip().split("\n")
        for segment in segments:
            process_utterance(
                # xml_writer,
                wavscp,
                text,
                utt2spk,
                label,
                os.path.join(args.src_data, "segments", "wavs"),
                args.wav_dumpdir,
                segment,
                score,
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
    # parser.add_argument(
    #    "--xml_dumpdir", type=str, help="xml obj dump directory", default="xml_dump"
    # )
    parser.add_argument(
        "--wav_dumpdir", type=str, help="wav dump directoyr (rebit)", default="wav_dump"
    )
    parser.add_argument("--sr", type=int, help="sampling rate (Hz)")
    parser.add_argument(
        "--g2p", type=str, help="g2p", default="pypinyin_g2p_phone_without_prosody"
    )
    args = parser.parse_args()

    tempos = load_midi(args)

    for name in ["train", "eval"]:
        process_subset(args, name, tempos)
