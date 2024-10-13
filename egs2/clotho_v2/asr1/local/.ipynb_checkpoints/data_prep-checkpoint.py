"""Prepares data for Clotho_v2 dataset."""

import glob
import os
import sys

import pandas as pd

if __name__ == "__main__":
    ROOT_DATA_DIR = sys.argv[1]
    AUDIO_DIR = "clotho_audio_files"
    CSV_DIR = "clotho_csv_files"

    SPLIT_NAMES = ["development", "validation", "evaluation"]

    for data_split in SPLIT_NAMES:
        missing_audio = []
        audio_regex = os.path.join(ROOT_DATA_DIR, AUDIO_DIR, data_split, "*.wav")
        all_audio_list = glob.glob(audio_regex)
        all_captions_path = os.path.join(
            ROOT_DATA_DIR, CSV_DIR, f"clotho_captions_{data_split}.csv"
        )
        captions_df = pd.read_csv(all_captions_path)
        N_PROCESSED = 0
        with open(
            os.path.join("data", data_split, "text"), "w", encoding="utf-8"
        ) as text_f, open(
            os.path.join("data", data_split, "wav.scp"), "w", encoding="utf-8"
        ) as wav_scp_f, open(
            os.path.join("data", data_split, "utt2spk"), "w", encoding="utf-8"
        ) as utt2spk_f:
            for uttid, row in captions_df.iterrows():
                uttid = f"{data_split}_uttid"
                text = row["caption_1"].strip()
                file_name = row["file_name"]
                audio_path = os.path.join(
                    ROOT_DATA_DIR, AUDIO_DIR, data_split, f"{file_name}"
                )
                if audio_path not in all_audio_list:
                    # Missing audio file
                    missing_audio += [audio_path]
                    continue
                print(f"{uttid} dummy", file=utt2spk_f)
                print(f"{uttid} {audio_path}", file=wav_scp_f)
                print(f"{uttid} {text}", file=text_f)
                N_PROCESSED += 1
        print(
            f"For split {data_split}: Processed {N_PROCESSED} audio and their captions. {len(missing_audio)} audio files were missing."
        )
