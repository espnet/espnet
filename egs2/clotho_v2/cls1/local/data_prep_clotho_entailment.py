"""Prepares data for Clotho audio entailment dataset.

Run with
python egs2/clotho_v2/cls1/local/data_prep_clotho_entailment.py ${INPUT_DIR} ${OUTPUT_DIR}
"""

import glob
import os
import sys
import pandas as pd
from string import punctuation

strip_punct_table = str.maketrans("", "", punctuation)


def read_hypothesis_text(csv_path):
    """
    Reads hypothesis text from a CSV file and returns a dictionary
    mapping audio files to lists of (hypothesis, label) tuples.
    Logs and counts missing values in the hypothesis columns.
    """
    df = pd.read_csv(csv_path)
    hypothesis_dict = {}
    missing_count = 0

    for _, row in df.iterrows():
        audio_file = row["Audio file"].strip()
        hypotheses = {
            "entailment": row["Entailment"],
            "neutral": row["Neutral"],
            "contradiction": row["Contradiction"],
        }

        if audio_file not in hypothesis_dict:
            hypothesis_dict[audio_file] = []

        for label, hypothesis in hypotheses.items():
            if pd.notna(hypothesis):
                hypothesis_dict[audio_file].append(
                    (hypothesis.strip().lower().translate(strip_punct_table), label)
                )
            else:
                missing_count += 1
                print(
                    f"Warning: Missing hypothesis for {audio_file} under label {label}"
                )

    print(f"Total missing hypotheses: {missing_count}")
    return hypothesis_dict


def prepare_data(input_dir, output_dir, audio_dir, csv_dir):
    """
    Prepares the Clotho entailment dataset for training and evaluation.
    """
    split_names = [
        "development",
        "validation",
        "evaluation",
    ]

    for data_split in split_names:
        missing_audio = []
        audio_regex = os.path.join(input_dir, audio_dir, data_split, "*.wav")
        audio_list = glob.glob(audio_regex)
        hypthesis_path = os.path.join(
            input_dir, csv_dir, f"clotho_entailment_{data_split}.csv"
        )
        hypothesis_dict = read_hypothesis_text(hypthesis_path)

        n_processed = 0
        write_data_split = f"{data_split}_cle"
        os.makedirs(os.path.join(output_dir, write_data_split), exist_ok=True)

        with open(
            os.path.join(output_dir, write_data_split, "wav.scp"), "w", encoding="utf-8"
        ) as wav_scp_f, open(
            os.path.join(output_dir, write_data_split, "utt2spk"), "w", encoding="utf-8"
        ) as utt2spk_f, open(
            os.path.join(output_dir, write_data_split, "text"), "w", encoding="utf-8"
        ) as text_f, open(
            os.path.join(output_dir, write_data_split, "hypothesis.txt"),
            "w",
            encoding="utf-8",
        ) as hyp_f:
            idx = 0
            for file_name, hypotheses in hypothesis_dict.items():
                audio_path = os.path.join(input_dir, audio_dir, data_split, file_name)
                if audio_path not in audio_list:
                    missing_audio.append(audio_path)
                    continue

                for hyp_text, label in hypotheses:
                    uttid = f"{write_data_split}_clotho_{idx}"
                    print(f"{uttid} dummy", file=utt2spk_f)
                    print(f"{uttid} {audio_path}", file=wav_scp_f)
                    print(f"{uttid} {label}", file=text_f)
                    print(f"{uttid} {hyp_text}", file=hyp_f)
                    idx += 1

                n_processed += 1
                if n_processed % 1000 == 0:
                    print(f"Processed {n_processed} audio files and {idx} examples.")

        print(
            f"For split {data_split}: Processed {n_processed} examples. {len(missing_audio)} audio files were missing."
        )


if __name__ == "__main__":
    INPUT_DIR = sys.argv[1]
    OUTPUT_DIR = sys.argv[2]
    AUDIO_DIR = os.path.join("CLOTHO_v2.1", "clotho_audio_files")
    CSV_DIR = os.path.join("AudioEntailment", "data", "CLE")

    prepare_data(INPUT_DIR, OUTPUT_DIR, AUDIO_DIR, CSV_DIR)
