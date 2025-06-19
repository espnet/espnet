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
from textgrid import TextGrid
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


def detect_leading_silence_duration(
    file_path, threshold=0.0005, frame_length=2048, hop_length=512
):
    # Step 1: Load the audio file
    y, sr = librosa.load(file_path, sr=None)

    # Step 2: Calculate short-time energy for each frame
    energy = np.array(
        [
            np.sum(np.abs(y[i : i + frame_length] ** 2))
            for i in range(0, len(y), hop_length)
        ]
    )

    # Step 3: Find index of the first frame where energy exceeds the threshold
    for i, e in enumerate(energy):
        if e > threshold:
            silence_duration = i * hop_length / sr
            return round(silence_duration, 4)  # round to 4 decimal places
    return 0.0  # All frames are below the threshold -> entire audio is silent


def load_and_process_textgrid(
    textgrid_path, transcriptions_path, song_folder, silence_duration
):
    # Step 1: Load the textgrid file
    tg = TextGrid.fromFile(textgrid_path)

    # Step 2: Get the start and end times of non-silent utterances
    name = tg[0].name
    assert name == "句子"
    silence_bias, idx = 0, 0
    while tg[0][idx].mark == "silence":
        silence_bias = silence_duration - tg[0][idx].maxTime
        idx += 1
    utterance_time = []
    for utt in tg[0]:
        if utt.mark != "silence":
            start_time = (utt.minTime + silence_bias) * 1000  # Convert to milliseconds
            end_time = (utt.maxTime + silence_bias) * 1000  # Convert to milliseconds
            utterance_time.append((start_time, end_time))

    # Step 3: Load the transcription file
    transcriptions = (
        open(transcriptions_path, "r", encoding="utf-8").read().strip().split("\n")
    )

    # Step 4: Split the content in each row
    # to name, lyrics, phns, pitch, duration, is_slur by |
    parsed_transcriptions = [row.split("|") for row in transcriptions]

    # Step 5: Find all the rows' corresponding name in the txt file starting with 2001
    relevant_transcriptions = [
        row for row in parsed_transcriptions if row[0].startswith(str(song_folder))
    ]

    # Step 6: Find the start and end times for the song's corresponding row names
    utterance_idx = 0
    utterance_time_map = {}
    for row in relevant_transcriptions:
        name, _, _, _, _, _, _ = row
        utterance_time_map[name] = [
            utterance_time[utterance_idx][0],
            utterance_time[utterance_idx][1],
        ]
        utterance_idx += 1
    return utterance_time_map


def segment_audio(audio, utterance_time_map, output_temp):
    for name, utt_time in utterance_time_map.items():
        start_time, end_time = utt_time[0], utt_time[1]
        if end_time > len(audio):
            end_time = len(audio)
        segment = audio[start_time:end_time]  # Extract the segment
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
            silence_duration = detect_leading_silence_duration(audio_path)

            textgrid_path = os.path.join(
                args.opencpop_src_data, "raw_data/textgrids", f"{song_folder}.TextGrid"
            )
            if not os.path.exists(textgrid_path):
                logging.warning(f"warning : {textgrid_path} does not exist")
                continue  # Skip if the textgrid file does not exist
            utterance_time_map = load_and_process_textgrid(
                textgrid_path, transcriptions_path, song_folder, silence_duration
            )
            segment_audio(audio, utterance_time_map, output_temp)


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
            and syb_dur[index_phn] == syb_dur[index_phn - 1]
            and midis[index_phn] == midis[index_phn - 1]
            and keep[index_phn] == 0
        ):
            syb.append(phns[index_phn])
            index_phn += 1
        lyrics_pinyin = "".join(syb)
        syb = "_".join(syb)
        note_info.extend([st, lyrics_pinyin, midi, syb])
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
        utt2spk.write("acesinger_{}#{} ace-{}\n".format(i_str, uid, i_str))

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
            if segment.startswith("2026"):
                continue
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
    parser.add_argument(
        "opencpop_src_data", type=str, help="opencpop source data directory"
    )
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
