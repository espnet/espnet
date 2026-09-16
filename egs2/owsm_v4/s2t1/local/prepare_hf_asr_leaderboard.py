#!/usr/bin/env python3
"""Sample a small Open ASR Leaderboard test set into a Kaldi-style data dir.

The Hugging Face Open ASR Leaderboard (hf-audio/open-asr-leaderboard) stores
each test set sorted by length, longest first, so taking "the first N rows"
would give the N longest utterances. This script instead streams the split,
takes a seeded shuffle over a buffer of rows and keeps the first N that are
at most --max_dur seconds long (OWSM pads or trims every input to 30 s).
References that are empty or equal to "ignore time segment in scoring" are
dropped, as in the leaderboard's own data loader.

The output directory contains wav.scp, text, utt2dur, utt2spk, spk2utt and an
info.json that records how the sample was drawn, so that the set can be used
both by local/eval_hf_asr_leaderboard.py and by the regular recipe stages.

Example:
    python local/prepare_hf_asr_leaderboard.py --dataset librispeech \
        --split test.clean --n 100 --out data/lb_librispeech_test_clean
"""

import argparse
import io
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

TEXT_KEYS = ("text", "sentence", "normalized_text", "transcript", "transcription")
IGNORE_REF = "ignore time segment in scoring"
# commit of hf-audio/open-asr-leaderboard the README numbers were sampled from; the
# default branch of a dataset can change, which would change the sample despite the seed
DEFAULT_REVISION = "b6bdcd0beb34f8975dc659796176d88f43aff502"
TARGET_SR = 16000


def get_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="leaderboard config, e.g. librispeech, ami_cleaned, earnings22",
    )
    parser.add_argument("--split", default="test", help="e.g. test, test.clean")
    parser.add_argument("--n", type=int, default=100, help="utterances to keep")
    parser.add_argument("--seed", type=int, default=0, help="shuffle seed")
    parser.add_argument(
        "--buffer", type=int, default=3000, help="shuffle buffer size in rows"
    )
    parser.add_argument(
        "--max_dur",
        type=float,
        default=30.0,
        help="skip utterances longer than this many seconds",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="dataset commit hash or branch to stream from (default: the commit "
        "the recipe README numbers were sampled from)",
    )
    parser.add_argument("--out", required=True, help="output data directory")
    return parser


def get_text(sample: dict) -> str:
    """Return the reference transcript of a leaderboard row."""
    for key in TEXT_KEYS:
        if key in sample:
            return sample[key]
    raise KeyError(f"no transcript column among {sorted(sample)}")


def decode_audio(audio: dict) -> np.ndarray:
    """Decode raw audio bytes to a mono 16 kHz float32 waveform."""
    wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    return wav


def main() -> None:
    """Sample the split and write the data directory."""
    args = get_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for name in ("httpx", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)

    from datasets import Audio, load_dataset

    out = Path(args.out)
    wav_dir = out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "hf-audio/open-asr-leaderboard",
        name=args.dataset,
        split=args.split,
        revision=args.revision,
        streaming=True,
    )
    dataset = dataset.cast_column("audio", Audio(decode=False))
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.buffer)

    prefix = re.sub(r"[^A-Za-z0-9]+", "_", f"{args.dataset}_{args.split}")
    rows = []
    seen = skipped_long = skipped_empty = 0
    start = time.time()
    for sample in dataset:
        seen += 1
        ref = " ".join(get_text(sample).split())
        if ref == "" or ref == IGNORE_REF:
            skipped_empty += 1
            continue
        wav = decode_audio(sample["audio"])
        duration = len(wav) / TARGET_SR
        if duration > args.max_dur:
            skipped_long += 1
            continue
        orig_id = str(sample.get("id") or sample.get("audio_id") or len(rows))
        orig_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", orig_id)[:60]
        utt_id = f"{prefix}_{len(rows):04d}_{orig_id}"
        wav_path = wav_dir / f"{utt_id}.wav"
        sf.write(str(wav_path), wav, TARGET_SR, subtype="PCM_16")
        rows.append((utt_id, wav_path.resolve(), duration, ref))
        if len(rows) % 20 == 0:
            logging.info(f"{len(rows)}/{args.n} kept ({time.time() - start:.0f}s)")
        if len(rows) >= args.n:
            break

    if not rows:
        logging.error("no utterance kept; check the dataset and split names")
        sys.exit(1)

    files = {
        "wav.scp": [f"{u} {p}" for u, p, _, _ in rows],
        "text": [f"{u} {ref}" for u, _, _, ref in rows],
        "utt2dur": [f"{u} {d:.3f}" for u, _, d, _ in rows],
        "utt2spk": [f"{u} {u}" for u, _, _, _ in rows],
        "spk2utt": [f"{u} {u}" for u, _, _, _ in rows],
    }
    for name, lines in files.items():
        (out / name).write_text("\n".join(lines) + "\n")

    durations = np.array([r[2] for r in rows])
    info = {
        "dataset": args.dataset,
        "split": args.split,
        "revision": args.revision,
        "n": len(rows),
        "seed": args.seed,
        "buffer": args.buffer,
        "max_dur": args.max_dur,
        "rows_seen": seen,
        "skipped_too_long": skipped_long,
        "skipped_empty_ref": skipped_empty,
        "audio_min": round(float(durations.sum()) / 60, 2),
        "dur_mean": round(float(durations.mean()), 2),
        "dur_median": round(float(np.median(durations)), 2),
        "dur_max": round(float(durations.max()), 2),
    }
    (out / "info.json").write_text(json.dumps(info, indent=1) + "\n")
    logging.info(json.dumps(info))

    # Leaving the streaming iterator early can hang inside the Hub client's
    # retry loop while the generator is torn down, so exit without waiting
    # for it; everything has been written and flushed above.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
