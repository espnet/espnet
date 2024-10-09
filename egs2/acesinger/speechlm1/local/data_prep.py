import argparse
import logging
import os
import re
import shutil

import librosa
import miditoolkit
import numpy as np

try:
    from pydub import AudioSegment
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "pydub needed for audio segmentation" "use pip to install pydub"
    )
from tqdm import tqdm

from espnet2.fileio.score_scp import SingingScoreWriter

"""Split audio into segments according to structured annotation.""" ""
"""Generate segments according to structured annotation."""
"""Transfer music score into 'score' format."""


def makedir(data_url):
    if os.path.exists(data_url):
        shutil.rmtree(data_url)
    os.makedirs(data_url)


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
    # Note(Yuning): note duration from '.midi' for Opencpop cannot be used here.
    # We only extract tempo from '.midi'.
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


def load_midi_note_scp(midi_note_scp):
    # Note(jiatong): midi format is 0-based
    midi_mapping = {}
    with open(midi_note_scp, "r", encoding="utf-8") as f:
        content = f.read().strip().split("\n")
        for key in content:
            key = key.split("\t")
            midi_mapping[key[1]] = int(key[0])
    return midi_mapping


def load_and_process_transcriptions(src_data, transcriptions_path, song_folder):
    # Step 1: Load the WAV file and the transcription file
    transcriptions = (
        open(transcriptions_path, "r", encoding="utf-8").read().strip().split("\n")
    )
    midi_obj = miditoolkit.midi.parser.MidiFile(
        os.path.join(src_data, "raw_data", "midis", "{}.midi".format(song_folder))
    )
    midi_notes = []
    midis = []
    tempo = midi_obj.tempo_changes[0].tempo
    ppq = midi_obj.ticks_per_beat
    # Populate the list with start and end times of each note
    for note in midi_obj.instruments[0].notes:
        start_time = note.start * 60000 / tempo / ppq
        end_time = note.end * 60000 / tempo / ppq
        midi_notes.append((start_time, end_time))
        pitch = note.pitch
        midis.append(pitch)
    # Step 2: Split the content in each row to name, lyrics, phns, pitch, duration,
    # is_slur by |
    parsed_transcriptions = [row.split("|") for row in transcriptions]

    # Step 3: Find all the rows' corresponding name in the txt file starting with 2001
    relevant_transcriptions = [
        row for row in parsed_transcriptions if row[0].startswith(str(song_folder))
    ]

    # Step 4: Sum the duration of each file to get the total duration of each subfile
    total_durations = {}
    total_pitches = {}
    for row in relevant_transcriptions:
        name, _, phns, pitches, _, duration, _ = row
        phns = phns.split(" ")
        duration = duration.split(" ")
        pitches = pitches.split(" ")
        while phns[-1] == "AP" or phns[-1] == "SP":
            phns = phns[:-1]
            duration = duration[:-1]
        if phns[0] == "SP" or phns[0] == "AP":
            phns = phns[1:]
            duration = duration[1:]
            pitches = pitches[1:]

        # sum duration by the string of durations, first split the string by space,
        # then convert the string to float, then sum the float
        total_durations[name] = (
            sum(float(d) for d in duration) * 1000
        )  # Convert to milliseconds
        total_pitches[name] = pitches
    return total_durations, total_pitches, midi_notes, midis


def segment_audio(
    audio, total_durations, total_pitches, midi_mapping, midi_notes, midis, output_temp
):
    start_time = 0  # Initialize the current time marker in the audio
    last_end_time = 0  # Initialize end time marker for last segment
    note_idx = 0
    utterance_idx = 0
    total_durations_keys = list(total_durations.keys())
    while utterance_idx < len(total_durations):
        name = total_durations_keys[utterance_idx]
        duration = total_durations[name]
        if note_idx >= len(midi_notes):
            break
        if note_idx == 0:
            start_time = midi_notes[note_idx][0]
            note_idx += 1
            last_end_time = start_time + duration
            utterance_idx += 1
            previous_note_idx = 0
        else:
            if midi_notes[note_idx][0] - last_end_time < -100:
                note_idx += 1
                continue
            pitch_trans = total_pitches[name][0]
            if midis[note_idx] != midi_mapping[pitch_trans]:
                orginal_note_idx = note_idx
                target_midi = midi_mapping[pitch_trans]
                min_distance = float("inf")
                nearest_note_idx = None
                # Search backwards within -5 index range
                for i in range(note_idx, max(note_idx - 5, -1), -1):
                    distance = abs(midis[i] - target_midi)
                    if distance == 0:
                        return i  # Exact match found
                    if distance < min_distance:
                        min_distance = distance
                        nearest_note_idx = i

                # Search within +5 index range and should > previous note index
                for i in range(note_idx, min(note_idx + 5, len(midis))):
                    if i <= previous_note_idx:
                        continue
                    distance = abs(midis[i] - target_midi)
                    if distance == 0:
                        return i  # Exact match found
                    if distance < min_distance:
                        min_distance = distance
                        nearest_note_idx = i
                if nearest_note_idx is not None:
                    note_idx = nearest_note_idx
                else:
                    note_idx = orginal_note_idx
            previous_note_idx = note_idx
            start_time = midi_notes[note_idx][0]
            note_idx += 1
            last_end_time = start_time + duration
            utterance_idx += 1
        segment = audio[start_time:last_end_time]  # Extract the segment
        output_name = output_temp + f"#{name}"
        if os.path.exists(f"{output_name}.wav"):
            print(f"{output_name}.wav already exists. Skipping.")
            continue
        segment.export(
            f"{output_name}.wav", format="wav"
        )  # Export it as a wav file in the output folder
    return None


def segment_dataset(args, dataset):
    root_path = os.path.join(args.src_data, dataset)
    transcriptions_path = os.path.join(args.src_data, "segments", "transcriptions.txt")
    for singer_folder in os.listdir(root_path):
        # starts with number
        if not singer_folder[0].isdigit():
            continue
        singer_path = os.path.join(root_path, singer_folder)
        if not os.path.isdir(singer_path):
            continue  # Skip files
        singer_id = re.findall(r"\d+", singer_folder)[0]
        output_temp = os.path.join(root_path, "segments") + f"/{singer_id}"
        print(f"Processing {singer_folder}")
        for song_folder in tqdm(os.listdir(singer_path)):
            song_path = os.path.join(singer_path, song_folder)
            if not os.path.isdir(song_path):
                continue
            for file in os.listdir(song_path):
                if file.endswith(".wav") and not file.startswith("._"):
                    audio_file_name = file
                    break
                else:
                    continue
            audio_path = os.path.join(song_path, audio_file_name)
            if not os.path.exists(audio_path):
                continue  # Skip if the audio file does not exist
            audio = AudioSegment.from_wav(audio_path)
            (
                total_durations,
                total_pitches,
                midi_notes,
                midis,
            ) = load_and_process_transcriptions(
                args.src_data, transcriptions_path, song_folder
            )
            midi_mapping = load_midi_note_scp(args.midi_note_scp)
            segment_audio(
                audio,
                total_durations,
                total_pitches,
                midi_mapping,
                midi_notes,
                midis,
                output_temp,
            )


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
            and syb_dur[index_phn] == syb_dur[index_phn - 1]
            and midis[index_phn] == midis[index_phn - 1]
            and keep[index_phn] == 0
        ):
            syb.append(phns[index_phn])
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
    midi_mapping,
    tempos,
    tgt_sr=24000,
    dataset="lyrics",
):
    uid, lyrics, phns, midis, syb_dur, phn_dur, keep = segment.strip().split("|")
    phns = phns.split(" ")
    midis = midis.split(" ")
    syb_dur = syb_dur.split(" ")
    phn_dur = phn_dur.split(" ")
    keep = keep.split(" ")

    # load tempo from midi
    id = int(uid[0:4])
    tempo = tempos[id]

    # note to midi index
    midis = [midi_mapping[note] if note != "rest" else 0 for note in midis]

    # type convert
    phn_dur = [float(dur) for dur in phn_dur]
    syb_dur = [float(syb) for syb in syb_dur]
    keep = [int(k) for k in keep]
    note_list = create_score(uid, phns, midis, syb_dur, keep)

    # 1 - 30
    for i in range(1, 31):
        i_str = str(i)  # Convert outer loop index to string
        wav_file = "{}.wav".format(
            os.path.join(audio_dir, dataset, "segments", i_str + "#" + uid)
        )
        if not os.path.exists(wav_file):
            logging.info("cannot find wav file at {}".format(wav_file))
            continue

        text.write("acesinger_{}#{} {}\n".format(i_str, uid, " ".join(phns)))
        utt2spk.write("acesinger_{}#{} {}\n".format(i_str, uid, i_str))

        cmd = "sox {}.wav -c 1 -t wavpcm -b 16 -r {} {}/acesinger_{}#{}.wav".format(
            os.path.join(audio_dir, dataset, "segments", i_str + "#" + uid),
            tgt_sr,
            wav_dumpdir,
            i_str,
            uid,
        )
        os.system(cmd)

        wavscp.write(
            "acesinger_{}#{} {}/acesinger_{}#{}.wav\n".format(
                i_str, uid, wav_dumpdir, i_str, uid
            )
        )

        running_dur = 0
        assert len(phn_dur) == len(phns)
        label_entry = []

        # Changed the variable name to avoid shadowing
        for j in range(len(phns)):
            start = running_dur
            end = running_dur + phn_dur[j]
            label_entry.append("{:.3f} {:.3f} {}".format(start, end, phns[j]))
            running_dur += phn_dur[j]

        label.write("acesinger_{}#{} {}\n".format(i_str, uid, " ".join(label_entry)))
        score = dict(
            tempo=tempo, item_list=["st", "et", "lyric", "midi", "phns"], note=note_list
        )
        writer["acesinger_{}#{}".format(i_str, uid)] = score


def process_subset(args, set_name, tempos, dataset):
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

    midi_mapping = load_midi_note_scp(args.midi_note_scp)

    with open(
        os.path.join(args.src_data, "segments", set_name + ".txt"),
        "r",
        encoding="utf-8",
    ) as f:
        segments = f.read().strip().split("\n")
        for segment in segments:
            process_utterance(
                writer,
                wavscp,
                text,
                utt2spk,
                label,
                args.src_data,
                args.wav_dumpdir,
                segment,
                midi_mapping,
                tempos,
                tgt_sr=args.sr,
                dataset=dataset,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Data for Acesinger Database")
    parser.add_argument("src_data", type=str, help="source data directory")
    parser.add_argument("--tgt_dir", type=str, default="data")
    parser.add_argument(
        "--midi_note_scp",
        type=str,
        help="midi note scp for information of note id",
        default="local/midi-note.scp",
    )
    parser.add_argument(
        "--wav_dumpdir", type=str, help="wav dump directory (rebit)", default="wav_dump"
    )
    parser.add_argument("--sr", type=int, help="sampling rate (Hz)")
    parser.add_argument("--g2p", type=str, help="g2p", default="None")
    parser.add_argument(
        "--score_dump", type=str, default="score_dump", help="score dump directory"
    )
    parser.add_argument(
        "--dataset", type=str, default="lyrics", help="dataset to process"
    )
    args = parser.parse_args()

    tempos = load_midi(args)
    dataset = args.dataset
    if not os.path.exists(os.path.join(args.src_data, dataset, "segments")):
        makedir(os.path.join(args.src_data, dataset, "segments"))
        segment_dataset(args, dataset)
    else:
        logging.info(
            "Found existing data at {}, skip re-generate".format(
                os.path.join(args.src_data, dataset, "segments")
            )
        )
    for name in ["train", "test"]:
        process_subset(args, name, tempos, dataset)
