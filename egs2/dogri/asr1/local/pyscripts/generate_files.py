import pandas as pd
from tqdm import tqdm
import os
import re
import argparse

def generate_files(
    wav_root, 
    data_root, 
    df_path, 
    drop_long_audios=False, 
    max_audio_duration=15, 
    drop_short_audios=False, 
    min_audio_duration=0.5
):
    df = pd.read_csv(df_path, header=None, sep="\t")
    df[0] = wav_root + df[0]

    if drop_long_audios:
        df_old_len = len(df)
        df = df[df[2] <= max_audio_duration].reset_index(drop=True)
        print(f"Dropping files of duration greater than {max_audio_duration}s. Dropped {df_old_len - len(df)} files.")

    if drop_short_audios:
        df = df[df[2] >= min_audio_duration].reset_index(drop=True)
        print(f"Dropping files of duration lesser than {min_audio_duration}s. Dropped {df_old_len - len(df)} files totally.")
    print(df)

    wav_scp_path = "/wav.scp"
    text_path = "/text"
    utt2spk_path = "/utt2spk"

    wav_scp_file = open(data_root + wav_scp_path, "w")
    text_file = open(data_root + text_path, "w")
    utt2spk_file = open(data_root + utt2spk_path, "w")

    for i in tqdm(range(len(df))):
        utt_id = df.iloc[i, -1]
        print(utt_id, df.iloc[i, 0], file=wav_scp_file)
        print(utt_id, df.iloc[i, 1], file=text_file)
        spk_id = utt_id
        print(utt_id, spk_id, file=utt2spk_file)

    wav_scp_file.close()
    text_file.close()
    utt2spk_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate files from a TSV DataFrame.")
    parser.add_argument("--wav-root", required=True, help="Path to the root directory containing the WAV files.")
    parser.add_argument("--data-root", required=True, help="Path to the root directory where files will be generated.")
    parser.add_argument("--df-path", required=True, help="Path to the TSV DataFrame file.")
    args = parser.parse_args()

    generate_files(
        wav_root=args.wav_root,
        data_root=args.data_root,
        df_path=args.df_path,
           )
    print("donee!!")

