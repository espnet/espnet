#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from typing import List

import kaldiio
import numpy as np
import torch
import torchaudio

from espnet.utils.cli_utils import get_commandline_args


def main():
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    parser = argparse.ArgumentParser(
        description="Generate speech by text and speaker prompt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Input text file",
    )
    parser.add_argument(
        "--speaker_prompt",
        type=str,
        help="Input speaker prompt file",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        help="number of examples to generate for each input string",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="generated sample rate",
    )
    parser.add_argument(
        "--tts_choice",
        type=str,
        choices=["chat_tts"],
        help="choice of tts model",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="output directory",
    )

    args = parser.parse_args()

    # (1) load data
    text_dict = dict()
    for line in open(args.text):
        example_id, content = line.strip().split(maxsplit=1)
        text_dict[example_id] = content

    prompt_dict = kaldiio.load_scp(args.speaker_prompt)

    # (2) init model
    model = TTSModel(
        tts_choice=args.tts_choice,
        fs=args.sample_rate,
    )

    # (3) inference loop
    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = open(args.output_dir / "wav.scp", "w")
    for key, text in text_dict.items():
        logging.info(f"inference on {key}: {text}")
        texts = [text for _ in range(args.n_examples)]
        speaker_prompt = [prompt_dict[key] for _ in range(args.n_examples)]

        wavs = model.infer(texts, speaker_prompt)

        for idx, wav in enumerate(wavs):
            output_name = "tts_" + key + f"_sample{idx}"
            output_path = str(args.output_dir / output_name) + ".wav"
            torchaudio.save(
                output_path,
                wav,
                sample_rate=args.sample_rate,
                bits_per_sample=16,
                encoding="PCM_S",
            )
            writer.write(f"{output_name} {output_path}\n")
            logging.info(f"save audio {output_path}")


class TTSModel:
    def __init__(
        self,
        tts_choice: str = "chat_tts",
        fs: int = 16000,
    ):
        self.tts_choice = tts_choice
        self.fs = fs

        if self.tts_choice == "chat_tts":
            import ChatTTS

            self.model = ChatTTS.Chat()
            self.model.load(compile=False)  # Set to True for better performance

            self.resampler = torchaudio.transforms.Resample(
                orig_freq=24000, new_freq=fs
            )

            logging.info("Load ChatTTS. Speaker Prompt is not in use")

        else:
            raise NotImplementedError

    def infer(
        self,
        texts: List[str],
        speaker_prompt: List[np.ndarray],
    ):
        if self.tts_choice == "chat_tts":
            wavs = self.model.infer(texts)
            wavs = [torch.from_numpy(wav) for wav in wavs]
            print(wavs[0], wavs[0].size(), flush=True)
            wavs = [self.resampler(wav) for wav in wavs]
            print(wavs[0], wavs[0].size(), flush=True)

        return wavs


if __name__ == "__main__":
    main()
