"""Prepares data for AudioCaps dataset."""

import json
import os
import sys
from string import punctuation

from scipy.io import wavfile

strip_punct_table = str.maketrans("", "", punctuation)

if __name__ == "__main__":
    ROOT_DATA_DIR = sys.argv[1]
    # this is a json with key "audiocap_ids" containing list of folder_ids
    TRAIN_FILES = "train_split.json"
    # this has folders each for a folder_id, containing audio.wav and
    # metadata.json files
    DATA_DIR = "data"

    LOCAL_DATA_DIR = "data/development_audiocaps"

    all_train_ids = None
    with open(os.path.join(ROOT_DATA_DIR, TRAIN_FILES)) as f:
        train_meta = json.load(f)
        all_train_ids = train_meta["audiocap_ids"]
    assert all_train_ids is not None, f"Could not read {ROOT_DATA_DIR/TRAIN_FILES}"

    output_wav_scp_path = os.path.join(LOCAL_DATA_DIR, "wav.scp")
    output_utt2spk_path = os.path.join(LOCAL_DATA_DIR, "utt2spk")
    output_txt_path = os.path.join(LOCAL_DATA_DIR, "text")
    missing_audio = []
    N_PROCESSED = 0
    N_ZERO_LENGTH_CAPTIONS = 0
    # We also skip examples with less than 6 words
    # according to Shih-Lun's paper Section 3.1:
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10447215
    N_SMALL_LENGTH_CAPTIONS = 0
    N_ZERO_LENGTH_AUDIO = 0
    N_VERY_LONG_AUDIO = 0
    AUDIO_LENGTH_THRESHOLD = 30  # seconds
    with open(output_txt_path, "w") as text_f, open(
        output_wav_scp_path, "w"
    ) as wav_scp_f, open(output_utt2spk_path, "w") as utt2spk_f:
        for train_id in all_train_ids:
            uttid = f"audiocaps_train_{train_id}"
            audio_path = os.path.join(ROOT_DATA_DIR, DATA_DIR, train_id, "audio.wav")
            samplerate, data = wavfile.read(audio_path)
            audio_length = len(data) / samplerate
            if audio_length == 0:
                N_ZERO_LENGTH_AUDIO += 1
                continue
            if audio_length > AUDIO_LENGTH_THRESHOLD:
                N_VERY_LONG_AUDIO += 1
                continue
            with open(
                os.path.join(ROOT_DATA_DIR, DATA_DIR, train_id, "metadata.json")
            ) as f:
                meta = json.load(f)
                text = meta["caption"].strip().lower().translate(strip_punct_table)
                if len(text) == 0:
                    N_ZERO_LENGTH_CAPTIONS += 1
                    continue
                if len(text.split()) < 6:
                    N_SMALL_LENGTH_CAPTIONS += 1
                    continue
                print(f"{uttid} {audio_path}", file=wav_scp_f)
                print(f"{uttid} {text}", file=text_f)
                print(f"{uttid} dummy", file=utt2spk_f)
                N_PROCESSED += 1
                if N_PROCESSED % 1000 == 0:
                    print(f"Processed {N_PROCESSED} audio files.")
    print(
        f"Processed {N_PROCESSED} audio files. Skipped {N_VERY_LONG_AUDIO}"
        f"long audio, {N_ZERO_LENGTH_AUDIO} empty audio and "
        f"{N_ZERO_LENGTH_CAPTIONS} empty caption items. {N_SMALL_LENGTH_CAPTIONS} "
        "were skipped because they had less than 6 words."
    )
