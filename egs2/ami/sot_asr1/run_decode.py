#!/usr/bin/env python3
"""Simple SOT inference script — bypasses ESPnet DataLoader."""

import logging
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    from espnet2.bin.sot_inference import SOTSpeech2Text

    config = "exp/sot_converted_turbo/config.yaml"
    model_file = "exp/sot_converted_turbo/valid.loss.best.pth"
    wav_scp = "data/test/wav.scp"
    output_dir = Path("exp/sot_converted_turbo/decode_test/1best_recog")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("Loading model...")
    t0 = time.time()
    speech2text = SOTSpeech2Text(
        asr_train_config=config,
        asr_model_file=model_file,
        device="cuda",
        beam_size=5,
        separator_token="????",
    )
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    # Read wav.scp
    utts = []
    with open(wav_scp) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utts.append((parts[0], parts[1]))
    logger.info(f"Total utterances: {len(utts)}")

    # Open output files
    f_text = open(output_dir / "text", "w")
    f_token = open(output_dir / "token", "w")
    f_token_int = open(output_dir / "token_int", "w")
    f_score = open(output_dir / "score", "w")

    t_start = time.time()
    for i, (key, wav_path) in enumerate(utts):
        try:
            audio, sr = sf.read(wav_path)
            results = speech2text(audio)
            text, token, token_int, hyp = results[0]

            f_text.write(f"{key} {text}\n")
            f_token.write(f"{key} {' '.join(token)}\n")
            f_token_int.write(f"{key} {' '.join(map(str, token_int))}\n")
            f_score.write(f"{key} {hyp.score}\n")

            # Flush every 10 utterances
            if (i + 1) % 10 == 0:
                f_text.flush()
                f_token.flush()
                f_token_int.flush()
                f_score.flush()

        except Exception as e:
            logger.warning(f"Failed {key}: {e}")
            f_text.write(f"{key} \n")
            f_token.write(f"{key} <space>\n")
            f_token_int.write(f"{key} 2\n")
            f_score.write(f"{key} 0.0\n")

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(utts) - i - 1) / rate
            logger.info(
                f"[{i+1}/{len(utts)}] " f"{rate:.1f} utt/s, " f"ETA: {eta/3600:.1f}h"
            )

    f_text.close()
    f_token.close()
    f_token_int.close()
    f_score.close()

    elapsed = time.time() - t_start
    logger.info(f"Done! {len(utts)} utterances in {elapsed/3600:.1f}h")


if __name__ == "__main__":
    main()
