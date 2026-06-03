import os
from pathlib import Path

import soundfile as sf
from datasets import load_dataset


def prepare_data():
    print("Loading dataset from Hugging Face...")
    # Load dataset - it will use 'audio_path' and 'text'
    ds = load_dataset("Professor/kinyarwanda-tts-dataset-kin")

    split_map = {"train": "train", "test": "test", "validation": "dev"}

    for hf_split, espnet_split in split_map.items():
        print(f"Processing {hf_split} -> data/{espnet_split}")
        data_dir = Path(f"data/{espnet_split}")
        wav_dir = Path(f"downloads/wavs/{espnet_split}")
        data_dir.mkdir(parents=True, exist_ok=True)
        wav_dir.mkdir(parents=True, exist_ok=True)

        text_file = data_dir / "text"
        wav_file = data_dir / "wav.scp"
        u2s_file = data_dir / "utt2spk"

        with (
            open(text_file, "w", encoding="utf-8") as f_text,
            open(wav_file, "w", encoding="utf-8") as f_wav,
            open(u2s_file, "w", encoding="utf-8") as f_u2s,
        ):

            for i, example in enumerate(ds[hf_split]):
                utt_id = f"rw_{espnet_split}_{i:05d}"

                # 1. Use the correct column names
                transcript = example["text"]
                audio_data = example["audio_path"]

                try:
                    # Write the WAV file to disk
                    wav_path = wav_dir / f"{utt_id}.wav"
                    sr = audio_data["sampling_rate"]
                    arr = audio_data["array"]

                    sf.write(str(wav_path), arr, sr)

                    # 2. Write Kaldi files
                    f_text.write(f"{utt_id} {transcript}\n")
                    f_wav.write(f"{utt_id} {wav_path.absolute()}\n")
                    f_u2s.write(f"{utt_id} speaker1\n")

                except Exception as e:
                    if i < 5:  # Print first few errors to debug
                        print(f"Error {i}: {e}. Type: {type(audio_data)}")
                    continue

        # Generate spk2utt using a split command string to pass linting
        u2s_path = f"data/{espnet_split}/utt2spk"
        s2u_path = f"data/{espnet_split}/spk2utt"
        cmd = f"utils/utt2spk_to_spk2utt.pl {u2s_path} > {s2u_path}"
        os.system(cmd)


if __name__ == "__main__":
    prepare_data()
