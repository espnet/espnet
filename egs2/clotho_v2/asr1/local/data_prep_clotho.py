"""Prepares data for Clotho_v2 dataset."""

import glob
import os
import sys

import pandas as pd

if __name__ == "__main__":
    ROOT_DATA_DIR = sys.argv[1]
    N_REF = int(sys.argv[2])
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
        if data_split == "development":
            dir_write_name = "development_clotho"
        else:
            dir_write_name = data_split  # for validation and evaluation
        with open(
            os.path.join("data", dir_write_name, "wav.scp"), "w", encoding="utf-8"
        ) as wav_scp_f, open(
            os.path.join("data", dir_write_name, "utt2spk"), "w", encoding="utf-8"
        ) as utt2spk_f:
            text_f = []
            try:
                if data_split == "evaluation":
                    text_f = [
                        open(
                            os.path.join("data", dir_write_name, f"text_spk{n_ref}"),
                            "w",
                            encoding="utf-8",
                        )
                        for n_ref in range(1, N_REF + 1)
                    ]
                else:
                    text_f = [
                        open(
                            os.path.join("data", dir_write_name, "text"),
                            "w",
                            encoding="utf-8",
                        )
                    ]
                for uttid, row in captions_df.iterrows():
                    uttid = f"{data_split}_clotho_{uttid}"
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
                    if data_split == "evaluation":
                        for i in range(1, N_REF + 1):
                            text_i = row[f"caption_{i}"].strip()
                            print(f"{uttid} {text_i}", file=text_f[i - 1])
                    else:
                        text = row["caption_1"].strip()
                        print(f"{uttid} {text}", file=text_f[0])
                    N_PROCESSED += 1
                    if N_PROCESSED % 1000 == 0:
                        print(f"Processed {N_PROCESSED} audio files.")
                        continue
            finally:
                for f in text_f:
                    f.close()
        print(
            f"For split {data_split}: Processed {N_PROCESSED} audio and their captions. {len(missing_audio)} audio files were missing."
        )
