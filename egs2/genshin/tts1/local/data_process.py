import argparse
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pypinyin
from pydub import AudioSegment

FOLDERS_TO_REMOVE = [
    "其它语音 - Others",
    "带变量语音 - Placeholder",
    "战斗语音 - Battle",
    "#Unknown",
]

AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}


def has_chinese(text: str) -> bool:
    """Check if the text contains Chinese characters"""
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def clean_name(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    cleaned = " ".join(cleaned.split())
    return cleaned


def convert_to_pinyin(text: str) -> str:
    try:
        pinyin_list = pypinyin.lazy_pinyin(text, strict=False)
        pinyin = "".join(word.capitalize() for word in pinyin_list)
        return clean_name(pinyin)
    except Exception as e:
        msg = f"Error converting pinyin for '{text}': {str(e)}"
        print(msg)
        return clean_name(text)


def safe_remove_folder(folder_path: str, dry_run: bool = False) -> bool:
    try:
        if dry_run:
            print(f"[Preview] Would remove folder: {folder_path}")
            return True

        shutil.rmtree(folder_path)
        print(f"Successfully removed folder: {folder_path}")
        return True

    except PermissionError:
        print(f"Permission error when removing: {folder_path}")
    except Exception as e:
        print(f"Error removing folder {folder_path}: {str(e)}")
    return False


def safe_rename(old_path: str, new_path: str) -> bool:
    try:
        if old_path == new_path:
            return True

        base_path = new_path
        counter = 1
        while os.path.exists(new_path):
            name, ext = os.path.splitext(base_path)
            new_path = f"{name}_{counter}{ext}"
            counter += 1

        os.rename(old_path, new_path)
        old_name = os.path.basename(old_path)
        new_name = os.path.basename(new_path)
        print(f"Renamed successfully: {old_name} -> {new_name}")
        return True

    except PermissionError:
        print(f"Permission error: {old_path}")
    except OSError as e:
        print(f"Failed to rename {old_path}: {str(e)}")
    except Exception as e:
        print(f"Unknown error for {old_path}: {str(e)}")
    return False


def get_audio_duration(file_path: str) -> float:
    """Get duration of audio file in seconds"""
    try:
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000.0
    except Exception as e:
        print(f"Error getting duration for {file_path}: {str(e)}")
        return 0.0


def convert_audio_format(file_path: str, dry_run: bool = False) -> tuple[str, bool]:
    """Convert audio to mono percific sample rate in WAV format"""
    global resample_audio
    try:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return file_path, False

        if file_path.lower().endswith(".wav"):
            try:
                import soundfile as sf

                data, sample_rate = sf.read(file_path)

                needs_conversion = False
                base_name = os.path.basename(file_path)

                if len(data.shape) > 1 and data.shape[1] > 1:
                    needs_conversion = True
                    print(f"File needs mono conversion: {base_name}")

                if sample_rate != resample_audio:
                    needs_conversion = True
                    msg = f"File needs resampling: {base_name}"
                    print(f"{msg} (current: {sample_rate}Hz)")

                if not needs_conversion:
                    return file_path, False

                if dry_run:
                    print(f"[Preview] Would convert: {file_path}")
                    channels = data.shape[1] if len(data.shape) > 1 else 1
                    msg = (
                        f"         (current: channels={channels}, "
                        f"sample rate={sample_rate}Hz)"
                    )
                    print(msg)
                    return file_path, True

                if len(data.shape) > 1 and data.shape[1] > 1:
                    data = data.mean(axis=1)

                if sample_rate != resample_audio:
                    from scipy import signal

                    new_length = int(len(data) * resample_audio / sample_rate)
                    data = signal.resample(data, new_length)

                sf.write(file_path, data, resample_audio)
                print(f"Converted: {base_name}")
                return file_path, True

            except Exception as e:
                msg = f"Error processing WAV file with soundfile: {str(e)}"
                print(msg)
                pass

        audio = AudioSegment.from_file(file_path)
        base_name = os.path.basename(file_path)

        if audio.channels > 1:
            audio = audio.set_channels(1)
            print(f"Converting to mono: {base_name}")

        if audio.frame_rate != resample_audio:
            audio = audio.set_frame_rate(resample_audio)
            msg = f"Resampling: {base_name} -> {resample_audio}Hz"
            print(msg)

        new_path = str(Path(file_path).with_suffix(".wav"))

        if (
            file_path == new_path
            and audio.channels == 1
            and audio.frame_rate == resample_audio
        ):
            return file_path, False

        if dry_run:
            print(f"[Preview] Would convert: {file_path} -> {new_path}")
            msg = (
                f"         (mono: {audio.channels == 1}, "
                f"sample rate: {audio.frame_rate}Hz)"
            )
            print(msg)
            return new_path, True

        audio.export(new_path, format="wav")

        if file_path != new_path:
            os.remove(file_path)

        old_name = os.path.basename(file_path)
        new_name = os.path.basename(new_path)
        print(f"Converted: {old_name} -> {new_name}")
        return new_path, True

    except Exception as e:
        print(f"Error converting {file_path}: {str(e)}")
        import traceback

        traceback.print_exc()
        return file_path, False


def process_audio_file(args) -> tuple:
    """Process a single audio file and return if it should be removed"""
    file_path, dry_run = args
    global resample_audio

    if resample_audio > 0:
        file_path, _ = convert_audio_format(file_path, dry_run)

    duration = get_audio_duration(file_path)
    if duration < 1.0:
        if dry_run:
            msg = (
                f"[Preview] Would remove short audio: {file_path} "
                f"(duration: {duration:.2f}s)"
            )
            print(msg)
        else:
            try:
                os.remove(file_path)
                os.remove(file_path.replace(".wav", ".lab"))
                msg = (
                    f"Removed short audio: {file_path} " f"(duration: {duration:.2f}s)"
                )
                print(msg)
            except Exception as e:
                print(f"Error removing {file_path}: {str(e)}")
        return file_path, True, 0
    return file_path, False, duration


def remove_short_audio_files(
    target_folder: str,
    dry_run: bool = False,
    remove_short_speaker: int = -1,
    resample_audio: int = -1,
) -> bool:
    print("\n=== Removing audio files shorter than 1 second ===")

    audio_files = []
    for root, _, files in os.walk(target_folder):
        for file in files:
            if Path(file).suffix.lower() in AUDIO_EXTENSIONS:
                audio_files.append((os.path.join(root, file), dry_run))

    if not audio_files:
        print("No audio files found")
        return

    print(f"Processing {len(audio_files)} audio files...")

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(process_audio_file, audio_files))

    removed_count = sum(1 for _, removed, _ in results if removed)
    total_duration = sum(duration for _, _, duration in results)
    msg = f"\nProcessed {len(audio_files)} files, removed {removed_count}"
    print(f"{msg} short audio files")
    if total_duration < remove_short_speaker:
        return True
    return False


def remove_specific_folders(target_folder: str, dry_run: bool = False) -> None:
    """Remove specified folders before renaming"""
    try:
        for root, dirs, _ in os.walk(target_folder, topdown=False):
            for dir_name in dirs:
                if dir_name in FOLDERS_TO_REMOVE:
                    folder_path = os.path.join(root, dir_name)
                    safe_remove_folder(folder_path, dry_run)
    except Exception as e:
        print(f"Error during folder removal: {str(e)}")


def rename_speaker_folders(target_folder: str, dry_run: bool = False) -> None:
    """
    Remove trailing '{num}', '-{num}', or '_{num}'
    from speaker folder names and merge contents if necessary
    """
    print("\n=== Removing trailing numbers from speaker folders ===")
    try:
        pattern = re.compile(r"[_-]?(\d+)$")
        for root, dirs, _ in os.walk(target_folder, topdown=False):
            for dir_name in dirs:
                match = pattern.search(dir_name)
                if match:
                    old_path = os.path.join(root, dir_name)
                    new_name = dir_name[: match.start()]
                    new_path = os.path.join(root, new_name)

                    if dry_run:
                        print(f"[Preview] Will rename: {dir_name} -> {new_name}")
                    else:
                        if os.path.exists(new_path):
                            for item in os.listdir(old_path):
                                src = os.path.join(old_path, item)
                                shutil.move(src, new_path)
                            shutil.rmtree(old_path)
                            msg = f"Merged and removed folder: {old_path} -> {new_path}"
                            print(msg)
                        else:
                            safe_rename(old_path, new_path)

    except Exception as e:
        print(f"Error processing speaker folders: {str(e)}")


def rename_folders_to_pinyin(
    target_folder: str, remove_short_speaker: int = -1, dry_run: bool = False
) -> None:
    try:
        if not os.path.exists(target_folder):
            print(f"Target folder does not exist: {target_folder}")
            return

        print("\n=== Removing specified folders ===")
        remove_specific_folders(target_folder, dry_run)

        print("\n=== Converting folder names to Pinyin ===")
        for root, dirs, _ in os.walk(target_folder, topdown=False):
            for dir_name in dirs:
                if has_chinese(dir_name):
                    old_path = os.path.join(root, dir_name)
                    new_name = convert_to_pinyin(dir_name)

                    if not new_name:
                        msg = f"Skipping {dir_name}: name would be empty after cleaning"
                        print(msg)
                        continue

                    new_path = os.path.join(root, new_name)

                    if dry_run:
                        print(f"[Preview] Will rename: {dir_name} -> {new_name}")
                    else:
                        safe_rename(old_path, new_path)

                dir_path = os.path.join(root, dir_name)
                if remove_short_audio_files(dir_path, dry_run, remove_short_speaker):
                    if remove_short_speaker > 0:
                        print(f"Removed short audio files in {dir_name}")
                        safe_remove_folder(dir_path, dry_run)

    except Exception as e:
        print(f"Error processing folder: {str(e)}")


def rename_files_underscore_to_dash(target_folder: str, dry_run: bool = False) -> None:
    """
    Replace underscores with dashes in all file and folder names
    """
    print("\n=== Converting underscores to dashes in names ===")
    try:
        for root, dirs, files in os.walk(target_folder, topdown=False):
            for file_name in files:
                if "_" in file_name:
                    old_path = os.path.join(root, file_name)
                    new_name = file_name.replace("_", "-")
                    new_path = os.path.join(root, new_name)

                    if dry_run:
                        msg = f"[Preview] Will rename file: {file_name} -> {new_name}"
                        print(msg)
                    else:
                        safe_rename(old_path, new_path)

            for dir_name in dirs:
                if "_" in dir_name:
                    old_path = os.path.join(root, dir_name)
                    new_name = dir_name.replace("_", "-")
                    new_path = os.path.join(root, new_name)

                    if dry_run:
                        msg = (
                            f"[Preview] Will rename directory: {dir_name} -> {new_name}"
                        )
                        print(msg)
                    else:
                        safe_rename(old_path, new_path)

    except Exception as e:
        print(f"Error converting underscores to dashes: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Process audio folders: remove specific folders, "
        "convert Chinese names to Pinyin, and remove short audio files"
    )
    parser.add_argument("target_folder", help="Target folder path to process")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview mode, no actual changes"
    )
    parser.add_argument(
        "--remove-short-speaker",
        type=int,
        default=-1,
        help="Remove audio files shorter than the specified duration in seconds",
    )
    parser.add_argument(
        "--resample-audio",
        type=int,
        default=-1,
        help="Convert audio files to specified sample rate(Hz)",
    )

    args = parser.parse_args()
    global resample_audio
    resample_audio = args.resample_audio

    print(f"Start processing folder: {args.target_folder}")
    print(f"Mode: {'Preview' if args.dry_run else 'Execute'}")

    rename_folders_to_pinyin(
        args.target_folder, args.remove_short_speaker, args.dry_run
    )
    rename_speaker_folders(args.target_folder, args.dry_run)
    rename_files_underscore_to_dash(args.target_folder, args.dry_run)

    print("\nProcessing completed")


if __name__ == "__main__":
    main()
