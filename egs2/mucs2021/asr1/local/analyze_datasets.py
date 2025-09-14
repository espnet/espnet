#!/usr/bin/env python3
#
# This script performs a detailed, reproducible analysis of the three datasets
# required for the Hinglish ASR bootcamp project. It is designed to
# validate the findings from the MUCS 2021 research papers and provide
# the necessary intelligence to design our data preparation pipeline.
#
# Author: Anshul Kumar & Gemini
# Date: September 14, 2025

import tarfile
import pathlib
import collections
import re
import random
import soundfile as sf
import io

# This regular expression helps us identify Hindi words by checking for
# characters within the Devanagari Unicode block.
HINDI_RE = re.compile(r'[\u0900-\u097F]')

def is_hindi(word):
    """Checks if a word contains any Devanagari characters."""
    return bool(HINDI_RE.search(word))

def analyze_mucs2021(archive_path, report_file):
    """
    Analyzes the MUCS 2021 dataset to quantify code-switching challenges
    and validate the language distribution mentioned in the KARI paper.
    """
    print("\n--- Analyzing MUCS 2021 (Hinglish Code-Switching) ---")
    report_file.write("--- Analysis of MUCS 2021 Dataset ---\n")

    total_utterances = 0
    total_words = 0
    hindi_words = 0
    english_words = 0
    example_lines = []

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            # We extract the 'text' file into memory to avoid unpacking the whole archive.
            text_file_member = tar.getmember("train/transcripts/text")
            text_file = tar.extractfile(text_file_member).read().decode("utf-8")

            lines = text_file.strip().split('\n')
            total_utterances = len(lines)
            example_lines = random.sample(lines, 10)

            for line in lines:
                # The format is "utterance_id word1 word2 ..."
                parts = line.strip().split()
                if len(parts) > 1:
                    words = parts[1:]
                    total_words += len(words)
                    for word in words:
                        if is_hindi(word):
                            hindi_words += 1
                        else:
                            english_words += 1

        # --- Generate the report ---
        lang_dist_report = (
            f"  - Total Utterances: {total_utterances}\n"
            f"  - Total Words: {total_words}\n"
            f"  - Hindi Words: {hindi_words} ({hindi_words / total_words:.2%})\n"
            f"  - English Words: {english_words} ({english_words / total_words:.2%})\n"
        )
        print(lang_dist_report)
        report_file.write(lang_dist_report + "\n")

        report_file.write("  - Example Code-Switched Utterances:\n")
        for ex in example_lines:
            report_file.write(f"    - {ex}\n")
        report_file.write("\n")

    except Exception as e:
        error_msg = f"  - ERROR: Could not analyze MUCS 2021. Reason: {e}\n"
        print(error_msg)
        report_file.write(error_msg)

def analyze_librispeech(archive_path, report_file):
    """
    Analyzes the LibriSpeech dataset to confirm its properties as a clean,
    monolingual English foundation.
    """
    print("\n--- Analyzing LibriSpeech (English Monolingual) ---")
    report_file.write("--- Analysis of LibriSpeech (train-clean-100) ---\n")

    total_utterances = 0
    is_all_uppercase = True

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            # Find all transcript files and one audio file for analysis
            transcript_files = [m for m in tar.getmembers() if m.name.endswith(".trans.txt")]
            first_audio_file_member = next(m for m in tar.getmembers() if m.name.endswith(".flac"))

            for member in transcript_files:
                content = tar.extractfile(member).read().decode("utf-8")
                lines = content.strip().split('\n')
                total_utterances += len(lines)
                for line in lines:
                    # Format: "ID-ID-ID WORD1 WORD2..."
                    transcription = " ".join(line.split()[1:])
                    if not transcription.isupper():
                        is_all_uppercase = False

            # Analyze audio properties
            audio_data = tar.extractfile(first_audio_file_member).read()
            info = sf.info(io.BytesIO(audio_data))

            # --- Generate the report ---
            report = (
                f"  - Total Utterances: {total_utterances}\n"
                f"  - Text Format: {'All Uppercase' if is_all_uppercase else 'Mixed Case'}\n"
                f"  - Audio Format (Sample): {info.samplerate} Hz, {info.channels} channel(s), {info.format}\n"
            )
            print(report)
            report_file.write(report + "\n")

    except Exception as e:
        error_msg = f"  - ERROR: Could not analyze LibriSpeech. Reason: {e}\n"
        print(error_msg)
        report_file.write(error_msg)


def analyze_gramvaani(archive_path, report_file, sample_size=100):
    """
    Analyzes the Gramvaani dataset to understand its text and, crucially,
    its varied audio formats.
    """
    print(f"\n--- Analyzing Gramvaani (Hindi Monolingual) ---")
    report_file.write("--- Analysis of Gramvaani (GV_Train_100h) ---\n")

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            # Text analysis
            text_file_member = tar.getmember("GV_Train_100h/text")
            text_file = tar.extractfile(text_file_member).read().decode("utf-8")
            lines = text_file.strip().split('\n')
            text_report = (
                f"  - Total Utterances: {len(lines) - 1}\n" # -1 for header
                f"  - Text Script: Primarily Devanagari\n"
            )
            print(text_report)
            report_file.write(text_report)

            # Audio analysis (this can be slow)
            print(f"  - Analyzing audio formats (sampling {sample_size} files, this may take a moment)...")
            audio_files = [m for m in tar.getmembers() if m.name.endswith(".mp3")]
            sampled_files = random.sample(audio_files, min(sample_size, len(audio_files)))

            samplerate_counts = collections.Counter()
            for member in sampled_files:
                try:
                    audio_data = tar.extractfile(member).read()
                    info = sf.info(io.BytesIO(audio_data))
                    samplerate_counts[info.samplerate] += 1
                except Exception:
                    samplerate_counts["unreadable"] += 1

            # --- Generate the audio report ---
            report_file.write("  - Audio Format Analysis (from sample):\n")
            total_sampled = sum(samplerate_counts.values())
            for rate, count in samplerate_counts.most_common():
                percent = (count / total_sampled) * 100
                audio_report_line = f"    - {rate} Hz: {count} files ({percent:.1f}%)\n"
                print(audio_report_line, end="")
                report_file.write(audio_report_line)
            print()

    except Exception as e:
        error_msg = f"  - ERROR: Could not analyze Gramvaani. Reason: {e}\n"
        print(error_msg)
        report_file.write(error_msg)


if __name__ == "__main__":
    # Define the paths to the downloaded archives
    downloads_dir = pathlib.Path("./downloads")
    report_path = pathlib.Path("./data_analysis_report.txt")

    mucs_archive = downloads_dir / "Hindi-English_train.tar.gz"
    librispeech_archive = downloads_dir / "train-clean-100.tar.gz"
    gramvaani_archive = downloads_dir / "GV_Train_100h.tar.gz"

    print("Starting comprehensive data analysis...")
    print("="*50)

    with open(report_path, "w", encoding="utf-8") as f:
        analyze_mucs2021(mucs_archive, f)
        analyze_librispeech(librispeech_archive, f)
        analyze_gramvaani(gramvaani_archive, f)

    print("="*50)
    print(f"Analysis complete. Full report saved to: {report_path}")