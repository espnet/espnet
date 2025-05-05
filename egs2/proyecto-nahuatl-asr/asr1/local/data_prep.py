import xml.etree.ElementTree as ET
import os
import glob
import re
from thefuzz import process
import random
import string
import subprocess
import argparse

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data from XML transcriptions and audio.")
    parser.add_argument("audio_data_path", type=str, help="Path to store output audio WAV files.")
    parser.add_argument("data_path", type=str, help="Output directory for Kaldi-style data files.")
    parser.add_argument("--trans_prefix", nargs='*', default=[], metavar="KEY=PATH",
                        help="Override transcription prefix, e.g. Zacatlan=/new/path")
    parser.add_argument("--audio_prefix", nargs='*', default=[], metavar="KEY=PATH",
                        help="Override audio prefix, e.g. Hidalgo=/other/path")
    return parser.parse_args()

def update_prefixes(defaults, overrides):
    updated = defaults.copy()
    for item in overrides:
        key, path = item.split("=", 1)
        updated[key] = path
    return updated


delset = string.punctuation
delset = delset.replace(":", "")
delset = delset.replace("'", "")

def clean_text(text, text_format="underlying_full"):
    text = re.sub(r"\.\.\.|\*|\[.*?\]|\n", "", text.upper())
    delset_specific = delset
    if text_format == "underlying_full":
        for char in "()=-":
            delset_specific = delset_specific.replace(char, "")
    return text.translate(str.maketrans("", "", delset_specific))

def extract_first_channel_16khz(input_wav, output_wav1, output_wav2):
    command = ["soxi", "-c", input_wav]
    num_channels = int(subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip())
    if num_channels == 1:
        if not os.path.exists(output_wav1):
            subprocess.run(["sox", input_wav, "-r", "16000", "-c", "1", "-b", "16", "-e", "signed-integer", output_wav1], check=True)
    else:
        if not os.path.exists(output_wav1):
            subprocess.run(["sox", input_wav, "-r", "16000", "-c", "1", "-b", "16", "-e", "signed-integer", output_wav1, "remix", "1"], check=True)
        if not os.path.exists(output_wav2):
            subprocess.run(["sox", input_wav, "-r", "16000", "-c", "1", "-b", "16", "-e", "signed-integer", output_wav2, "remix", "2"], check=True)

def parse_trans_xml(trans_file, audio_file, out_audio_dir):
    tree = ET.parse(trans_file)
    root = tree.getroot()

    text_lines, wav_scp_lines, utt2spk_lines, spk2utt_lines, segments_lines = [], [], [], [], []
    wav_file = os.path.basename(audio_file).strip(".wav")
    extract_first_channel_16khz(audio_file, f"{out_audio_dir}/{wav_file}_spk1.wav", f"{out_audio_dir}/{wav_file}_spk2.wav")

    for i in [1, 2]:
        path = f"{out_audio_dir}/{wav_file}_spk{i}.wav"
        if os.path.exists(path):
            wav_scp_lines.append(f"{wav_file}_spk{i} {path}")

    speaker_utt, spk2utt_dict = [], {}
    for turn in root.findall(".//Turn"):
        speaker = turn.attrib.get("speaker", None)
        if not speaker:
            continue
        sync_times = [e.attrib["time"].strip() for e in turn.findall("Sync")]
        sync_times.append(turn.get("endTime", "0"))
        spk, i = speaker, 0
        for elem in turn:
            if elem.tag == "Sync" and elem.tail.strip():
                speaker_utt.append([sync_times[i], sync_times[i+1], spk, clean_text(elem.tail.strip())])
                i += 1
            elif elem.tag == "Who":
                spk = f"spk{elem.get('nb', '1')}"
                speaker_utt.append([sync_times[i], sync_times[i+1], spk, clean_text(elem.tail.strip())])
            elif elem.tag == "Comment":
                speaker_utt[-1][-1] += " " + clean_text(elem.tail.strip())

    for start, end, spk, text in speaker_utt:
        if not all([start, end, spk, text]): continue
        utt_id = f"{wav_file}_{spk}_{start}_{end}"
        wav_id = f"{wav_file}_{spk}"
        spk2utt_dict.setdefault(wav_id, []).append(utt_id)
        utt2spk_lines.append(f"{utt_id} {wav_id}")
        text_lines.append(f"{utt_id} {text}")
        segments_lines.append(f"{utt_id} {wav_id} {start} {end}")

    for key, val in spk2utt_dict.items():
        spk2utt_lines.append(f"{key} {' '.join(val)}")

    return text_lines, wav_scp_lines, utt2spk_lines, spk2utt_lines, segments_lines

def write_file(content, out_path):
    with open(out_path, "w") as f:
        f.write("\n".join(content))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare data from XML transcriptions and audio.")
    parser.add_argument("--audio_data_path", type=str, help="Path to store output audio WAV files.")
    parser.add_argument("--data_path", type=str, help="Output directory for Kaldi-style data files.")

    parser.add_argument("--trans_prefix_Zacatlan", type=str, help="Transcription prefix for Zacatlan.")
    parser.add_argument("--trans_prefix_Tequila", type=str, help="Transcription prefix for Tequila.")
    parser.add_argument("--trans_prefix_Hidalgo", type=str, help="Transcription prefix for Hidalgo.")

    parser.add_argument("--audio_prefix_Zacatlan", type=str, help="Audio prefix for Zacatlan.")
    parser.add_argument("--audio_prefix_Tequila", type=str, help="Audio prefix for Tequila.")
    parser.add_argument("--audio_prefix_Hidalgo", type=str, help="Audio prefix for Hidalgo.")

    args = parser.parse_args()
    
    # Defaults
    trans_prefix = {
        "Zacatlan": args.trans_prefix_Zacatlan,
        "Tequila": args.trans_prefix_Tequila,
        "Hidalgo": args.trans_prefix_Hidalgo
    }

    audio_prefix = {
        "Zacatlan": args.audio_prefix_Zacatlan,
        "Tequila": args.audio_prefix_Tequila,
        "Hidalgo": args.audio_prefix_Hidalgo
    }

    # DEFAULT_TRANS_PREFIX = {
    #     "Zacatlan": "/ocean/projects/cis210027p/shared/corpora/Proyecto-Nahuatl-ASR/Zacatlan-Tepetzintla/Transcripciones-finales",
    #     "Tequila": "/ocean/projects/cis210027p/shared/corpora/Proyecto-Nahuatl-ASR/Tequila-Zongolica/Transcripciones-finales",
    #     "Hidalgo": "/ocean/projects/cis210027p/shared/corpora/Proyecto-Nahuatl-ASR/Hidalgo-Transcripciones/Transcripciones-Finales"
    # }

    # DEFAULT_AUDIO_PREFIX = {
    #     "Zacatlan": "/ocean/projects/cis210027p/shared/corpora/Proyecto-Nahuatl-ASR/Zacatlan-Tepetzintla/Grabaciones_Por-dia",
    #     "Tequila": "/ocean/projects/cis210027p/shared/corpora/Proyecto-Nahuatl-ASR/Tequila-Zongolica/Grabaciones",
    #     "Hidalgo": "/ocean/projects/cis210027p/shared/corpora/Proyecto-Nahuatl-ASR/Hidalgo-Grabaciones"
    # }

    data_path = args.data_path
    audio_data_path = args.audio_data_path
    random.seed(2025)

    all_text_lines, all_wav_scp_lines, all_utt2spk_lines, all_spk2utt_lines, all_segments_lines = ({"train": [], "dev": [], "test": {k: [] for k in trans_prefix}} for _ in range(5))
    train_ratio, dev_ratio = 0.7, 0.8

    for lan in trans_prefix:
        all_trans = sorted(glob.glob(os.path.join(trans_prefix[lan], "*.trs")))
        trans_partition = {
            "train": all_trans[:int(train_ratio*len(all_trans))],
            "dev": all_trans[int(train_ratio*len(all_trans)):int(dev_ratio*len(all_trans))],
            "test": all_trans[int(dev_ratio*len(all_trans)):]
        }

        for curr_set in trans_partition:
            for trans in trans_partition[curr_set]:
                date_folder = re.findall(r"\d{4}-\d{2}-\d{2}", trans)[0]
                choices = glob.glob(os.path.join(audio_prefix[lan], date_folder + "*/*.wav"))
                wav = process.extract(trans, choices, limit=1)[0][0]
                text_lines, wav_scp_lines, utt2spk_lines, spk2utt_lines, segments_lines = parse_trans_xml(trans, wav, audio_data_path)
                if curr_set in ["train", "dev"]:
                    all_text_lines[curr_set].extend(text_lines)
                    all_wav_scp_lines[curr_set].extend(wav_scp_lines)
                    all_utt2spk_lines[curr_set].extend(utt2spk_lines)
                    all_spk2utt_lines[curr_set].extend(spk2utt_lines)
                    all_segments_lines[curr_set].extend(segments_lines)
                else:
                    all_text_lines[curr_set][lan].extend(text_lines)
                    all_wav_scp_lines[curr_set][lan].extend(wav_scp_lines)
                    all_utt2spk_lines[curr_set][lan].extend(utt2spk_lines)
                    all_spk2utt_lines[curr_set][lan].extend(spk2utt_lines)
                    all_segments_lines[curr_set][lan].extend(segments_lines)                

    for split in ["train", "dev"]:
        os.makedirs(os.path.join(data_path, split), exist_ok=True)
        write_file(all_text_lines[split], f"{data_path}/{split}/text")
        write_file(all_wav_scp_lines[split], f"{data_path}/{split}/wav.scp")
        write_file(all_utt2spk_lines[split], f"{data_path}/{split}/utt2spk")
        write_file(all_spk2utt_lines[split], f"{data_path}/{split}/spk2utt")
        write_file(all_segments_lines[split], f"{data_path}/{split}/segments")

    for lan in all_text_lines["test"]:
        os.makedirs(os.path.join(data_path, "test", lan), exist_ok=True)
        write_file(all_text_lines["test"][lan], f"{data_path}/test/{lan}/text")
        write_file(all_wav_scp_lines["test"][lan], f"{data_path}/test/{lan}/wav.scp")
        write_file(all_utt2spk_lines["test"][lan], f"{data_path}/test/{lan}/utt2spk")
        write_file(all_spk2utt_lines["test"][lan], f"{data_path}/test/{lan}/spk2utt")
        write_file(all_segments_lines["test"][lan], f"{data_path}/test/{lan}/segments")
