"""Prepares data for Clotho AQA dataset.

Run with
python egs2/clotho_v2/cls1/local/data_prep_clotho_aqa.py ${INPUT_DIR} ${OUTPUT_DIR}
"""

import glob
import os
import sys
from string import punctuation

import pandas as pd

# strip_punct_table = str.maketrans("", "", punctuation)


def prepare_data(input_dir, output_dir, audio_dir, csv_pattern, task_type="yn"):
    """
    Prepares the Clotho entailment dataset for training and evaluation.
    """
    split_names = {
        "development": "train",
        "validation": "val",
        "evaluation": "test",
    }  # clotho -> aqa map

    audio_regex = os.path.join(input_dir, audio_dir, "*.wav")
    audio_list = glob.glob(audio_regex)
    for data_split in split_names:
        missing_audio = []
        question_csv_path = os.path.join(
            input_dir, csv_pattern.format(split=split_names[data_split])
        )
        qa_df = pd.read_csv(question_csv_path)  # file_name, QuestionText, answer

        n_processed = 0
        write_data_plit = f"{data_split}_aqa_{task_type}"
        os.makedirs(os.path.join(output_dir, write_data_plit), exist_ok=True)

        with open(
            os.path.join(output_dir, write_data_plit, "wav.scp"), "w", encoding="utf-8"
        ) as wav_scp_f, open(
            os.path.join(output_dir, write_data_plit, "utt2spk"), "w", encoding="utf-8"
        ) as utt2spk_f, open(
            os.path.join(output_dir, write_data_plit, "text"), "w", encoding="utf-8"
        ) as text_f, open(
            os.path.join(output_dir, write_data_plit, "question.txt"),
            "w",
            encoding="utf-8",
        ) as ques_f:
            idx = 0
            for _, row in qa_df.iterrows():
                file_name = row["file_name"].strip()
                audio_path = os.path.join(input_dir, audio_dir, file_name)
                if audio_path not in audio_list:
                    missing_audio.append(audio_path)
                    continue
                question = row["QuestionText"].strip().lower()
                label = row["answer"].strip().lower()

                if task_type == "yn":
                    # only prepare yes/no set
                    if label != "yes" and label != "no":
                        continue
                elif task_type == "open":
                    # only prepare open set
                    if label == "yes" or label == "no":
                        continue
                else:
                    raise ValueError(f"Invalid task_type {task_type}")

                uttid = f"{write_data_plit}_clotho_{idx}"
                print(f"{uttid} dummy", file=utt2spk_f)
                print(f"{uttid} {audio_path}", file=wav_scp_f)
                print(f"{uttid} {label}", file=text_f)
                print(f"{uttid} {question}", file=ques_f)
                idx += 1

                n_processed += 1
                if idx % 10000 == 0:
                    print(f"Processed {idx} examples.")

        print(
            f"For split {data_split}: Processed {n_processed} examples. {len(missing_audio)} audio files were missing."
        )


if __name__ == "__main__":
    INPUT_DIR = sys.argv[1]
    OUTPUT_DIR = sys.argv[2]
    task_type = sys.argv[3]  # yn or open
    AUDIO_DIR = "audio_files"
    CSV_PATTERN = "clotho_aqa_{split}.csv"

    prepare_data(INPUT_DIR, OUTPUT_DIR, AUDIO_DIR, CSV_PATTERN, task_type)
