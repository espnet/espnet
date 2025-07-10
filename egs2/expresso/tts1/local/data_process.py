import argparse
import os
import shutil
from pathlib import Path

import librosa
import soundfile as sf
from tqdm import tqdm
import re

def get_audio_duration(filepath):
    info = sf.info(filepath)
    return info.frames / info.samplerate


def remove_and_resample(root_dir, max_duration, resample_rate=None):
    num_removed = 0

    wav_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".wav"):
                wav_files.append(os.path.join(dirpath, fname))

    for fpath in tqdm(wav_files, desc="Removing and resampling audio"):
        duration = get_audio_duration(fpath)

        if duration is not None and duration > max_duration:
            os.remove(fpath)
            num_removed += 1
        elif resample_rate is not None:
            y, sr = librosa.load(fpath, sr=None)
            if sr != resample_rate:
                y_resampled = librosa.resample(y, orig_sr=sr, target_sr=resample_rate)
                sf.write(fpath, y_resampled, resample_rate)

    print(f"✅ Removed {num_removed} audio files.")


def clean_and_save_splits(split_dir, output_split_dir, existing_ids):
    output_split_dir.mkdir(parents=True, exist_ok=True)
    all_files = []

    for split_file in ["train.txt", "dev.txt", "test.txt"]:
        input_path = split_dir / split_file
        output_path = output_split_dir / split_file

        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[1:]  # skip header

        cleaned_lines = []
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) != 1:
                continue

            utt_id = parts[0]
            if utt_id.count("-") >= 2:
                continue
            if utt_id not in existing_ids:
                continue

            speaker_prefix = utt_id.split("_")[0] + "-"
            full_id = speaker_prefix + utt_id
            cleaned_lines.append(full_id)
            all_files.append(utt_id)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(cleaned_lines) + "\n")

    return all_files

def clean_text(text):
    # Remove "*" and "|"
    text = text.replace("*", "")
    text = text.replace("|", "")
    
    # Remove all tags like <...>
    text = re.sub(r"<[^>]+>", "", text)
    
    # Remove multiple spaces and replace with a single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    return text.strip()


def organize_dataset(src_audio_dir, output_base_dir, transcription_path, all_files):
    
    transcriptions = {
        line.split("\t")[0]: clean_text(line.strip().split("\t")[1])
        for line in open(transcription_path, "r", encoding="utf-8")
        if line.strip()
    }

    valid_ids = set(all_files)
    for path in tqdm(
        list(Path(src_audio_dir).rglob("*.wav")),
        desc="Copying and organizing .wav files",
    ):
        file_id = path.stem
        if file_id not in valid_ids:
            continue

        speaker_id = file_id.split("_")[0]
        out_dir = Path(output_base_dir) / "audio" / speaker_id
        out_dir.mkdir(parents=True, exist_ok=True)

        symlink_path = out_dir / path.name
        if not symlink_path.exists():
            symlink_path.symlink_to(path.resolve())

        with open(out_dir / f"{file_id}.txt", "w", encoding="utf-8") as f:
            f.write(transcriptions.get(file_id, ""))


def main():
    parser = argparse.ArgumentParser(
        description="Organize audio dataset and clean splits."
    )
    parser.add_argument("input_folder", help="Path to the input folder")
    parser.add_argument("output_folder", help="Path to the output folder")
    parser.add_argument(
        "--max_duration",
        type=float,
        default=60.0,
        help="Maximum audio duration in seconds",
    )
    parser.add_argument(
        "--resample_rate", type=int, default=None, help="Resample rate for audio files"
    )
    args = parser.parse_args()

    input_path = Path(args.input_folder)
    output_path = Path(args.output_folder)

    src_audio_dir = input_path / "audio_48khz" / "read"
    transcription_path = input_path / "read_transcriptions.txt"
    split_dir = input_path / "splits"
    output_split_dir = output_path / "splits"

    print("Removing long audio files...")
    remove_and_resample(args.input_folder, args.max_duration, args.resample_rate)

    print("Indexing existing audio files...")
    existing_ids = set(p.stem for p in src_audio_dir.rglob("*.wav"))

    print("Cleaning split files...")
    all_files = clean_and_save_splits(split_dir, output_split_dir, existing_ids)

    print("Copying and organizing audio files...")
    organize_dataset(src_audio_dir, output_path, transcription_path, all_files)

    print("Processing completed.")


if __name__ == "__main__":
    main()
