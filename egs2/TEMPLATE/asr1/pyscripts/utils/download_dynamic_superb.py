import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--allow_multi_utt", action="store_true", default=False)
    return parser


def is_audio(example):
    """
    Check if the given example is an audio.

    Args:
        example: The example to check.

    Returns:
        True if the example is an audio, False otherwise.
    """
    if isinstance(example, dict):
        if "path" in example and "array" in example and "sampling_rate" in example:
            return True
    return False


def merge_audio(wavs: List[np.array], sampling_rate: int):
    """
    Merge a list of audio files into a single audio file by concatenating
    the files and inserting 0.5s of silence in between.

    Args:
        wavs (List[np.array]): A list of audio contents as numpy arrays.
        sampling_rate (int): The common sampling rate.

    Returns:
        np.ndarray: The merged audio as a numpy array.
    """

    # Concatenate all wavs and insert silence (0.5s) in between
    new_wavs = []
    for wav in wavs:
        new_wavs.append(wav)
        new_wavs.append(np.zeros(int(0.5 * sampling_rate)))

    # Remove the last silence
    wav = np.concatenate(new_wavs[:-1], axis=0)
    return wav


def main():
    parser = get_parser()
    args = parser.parse_args()
    metadata = json.load(args.json_path.open(mode="r"))

    hf_path = metadata["path"]
    version = metadata["version"]

    dataset = load_dataset(hf_path, revision=version, split="test")

    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    wav_dir = save_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    metadata = defaultdict(dict)

    for idx, example in tqdm(enumerate(dataset)):
        audio_inputs = []
        metadata[f"{idx:06d}"]["audio"] = []
        sampling_rate = None
        if "audio" in example:
            audio = example["audio"]["array"]
            sampling_rate = example["audio"]["sampling_rate"]
            audio_inputs.append(audio)
            # sf.write(save_path, audio, sampling_rate)

        # Handle multiple utterance as inputs (e.g., Speaker Verification)
        audio_idx = 2
        audio_key = f"audio{audio_idx}"
        while audio_key in example:
            audio = example[audio_key]["array"]
            curr_sr = example[audio_key]["sampling_rate"]
            if sampling_rate is None:
                sampling_rate = curr_sr
            assert (
                sampling_rate == curr_sr
            ), f"Sampling rate mismatch: {sampling_rate} != {curr_sr}"
            audio_inputs.append(audio)
            audio_idx = audio_idx + 1
            audio_key = f"audio{audio_idx}"

        if len(audio_inputs) >= 1:
            if args.allow_multi_utt:
                for audio_idx, audio in enumerate(audio_inputs):
                    save_path = wav_dir / f"{idx:06d}_pair{audio_idx}.wav"
                    save_path = save_path.absolute().as_posix()
                    sf.write(save_path, audio, sampling_rate)
                    metadata[f"{idx:06d}"]["audio"].append(save_path)
            else:
                audio = merge_audio(audio_inputs, sampling_rate)
                save_path = wav_dir / f"{idx:06d}.wav"
                save_path = save_path.absolute().as_posix()
                sf.write(save_path, audio, sampling_rate)
                metadata[f"{idx:06d}"]["audio"].append(save_path)

        assert "instruction" in example
        metadata[f"{idx:06d}"]["instruction"] = example["instruction"]

        assert "label" in example
        metadata[f"{idx:06d}"]["label"] = example["label"]

        if "text" in example:
            metadata[f"{idx:06d}"]["text"] = example["text"]

    scp_path = save_dir / "wav.scp"
    with scp_path.open(mode="w") as f:
        for idx, example in metadata.items():
            audio_paths = example["audio"]
            f.write(f"{idx} {' '.join(audio_paths)}\n")

    text_input_path = save_dir / "text.input"
    with text_input_path.open(mode="w") as f:
        for idx, example in metadata.items():
            text_input = example["instruction"]
            f.write(f"{idx} {text_input}\n")

    text_output_path = save_dir / "text.output"
    with text_output_path.open(mode="w") as f:
        for idx, example in metadata.items():
            text_output = example["label"]
            f.write(f"{idx} {text_output}\n")

    # Dummy utt2spk
    utt2spk_path = save_dir / "utt2spk"
    with utt2spk_path.open(mode="w") as f:
        for idx, example in metadata.items():
            f.write(f"{idx} {idx}\n")


if __name__ == "__main__":
    main()
