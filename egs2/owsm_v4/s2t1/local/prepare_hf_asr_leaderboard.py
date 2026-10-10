#!/usr/bin/env python3
"""Prepare an Open ASR Leaderboard test set into a Kaldi-style data dir.

The Hugging Face Open ASR Leaderboard (hf-audio/open-asr-leaderboard) stores
each test set sorted by length, longest first. For full evaluations (--n 0 or
--all), this script streams the complete split without sampling or duration
skipping, preserving all utterances for leaderboard-comparable evaluation.

For quick smoke tests (--n > 0), this script streams the split, takes a seeded
shuffle over a buffer of rows, and keeps the first N utterances that are at most
--max_dur seconds long (OWSM pads or trims single windows to 30 s).

References that are empty or equal to "ignore time segment in scoring" are
dropped, matching the leaderboard's own data loader.

The output directory contains wav.scp, text, utt2dur, utt2spk, spk2utt and an
info.json that records metadata and statistics (including utterances > 30 s),
so that the set can be used both by local/eval_hf_asr_leaderboard.py and by the
regular recipe stages.

Supported configs of hf-audio/open-asr-leaderboard:
    ami, ami_cleaned, common_voice, earnings22, gigaspeech, gigaspeech_cleaned,
    librispeech (splits: test.clean, test.other), spgispeech, tedlium, voxpopuli,
    voxpopuli_cleaned_aa, urgent2024, urgent2024_clean

Examples:
    # Full split evaluation (all utterances, no sampling)
    python local/prepare_hf_asr_leaderboard.py --dataset librispeech \
        --split test.clean --all --out data/lb_librispeech_test_clean

    # Quick 100-utterance sample smoke test
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

LEADERBOARD_CONFIGS = {
    "ami": ["test"],
    "ami_cleaned": ["test"],
    "common_voice": ["test"],
    "earnings22": ["test"],
    "gigaspeech": ["test"],
    "gigaspeech_cleaned": ["test"],
    "librispeech": ["test.clean", "test.other"],
    "spgispeech": ["test"],
    "tedlium": ["test"],
    "voxpopuli": ["test"],
    "voxpopuli_cleaned_aa": ["test"],
    "urgent2024": ["test"],
    "urgent2024_clean": ["test"],
}


def get_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="leaderboard config, e.g. " + ", ".join(sorted(LEADERBOARD_CONFIGS.keys())),
    )
    parser.add_argument("--split", default="test", help="e.g. test, test.clean, test.other")
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="utterances to keep; 0 means keep the whole split with no sampling",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="keep the whole split with no sampling (equivalent to --n 0)",
    )
    parser.add_argument("--seed", type=int, default=0, help="shuffle seed")
    parser.add_argument(
        "--buffer", type=int, default=3000, help="shuffle buffer size in rows"
    )
    parser.add_argument(
        "--max_dur",
        type=float,
        default=None,
        help="skip utterances longer than this many seconds (default: 30.0 for sampled "
        "runs; 0.0 / no limit for full runs so utterances over 30s are preserved)",
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

    if args.all or args.n <= 0:
        args.n = 0
        if args.max_dur is None:
            args.max_dur = 0.0
    elif args.max_dur is None:
        args.max_dur = 30.0

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
    if args.n > 0:
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
        if args.max_dur > 0 and duration > args.max_dur:
            skipped_long += 1
            continue
        orig_id = str(sample.get("id") or sample.get("audio_id") or len(rows))
        orig_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", orig_id)[:60]
        utt_id = f"{prefix}_{len(rows):06d}_{orig_id}"
        wav_path = wav_dir / f"{utt_id}.wav"
        sf.write(str(wav_path), wav, TARGET_SR, subtype="PCM_16")
        rows.append((utt_id, wav_path.resolve(), duration, ref))
        if args.n > 0:
            if len(rows) % 20 == 0:
                logging.info(f"{len(rows)}/{args.n} kept ({time.time() - start:.0f}s)")
            if len(rows) >= args.n:
                break
        else:
            if len(rows) % 100 == 0:
                logging.info(
                    f"{len(rows)} kept (seen {seen}, {time.time() - start:.0f}s)"
                )

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
    n_over_30s = int(np.sum(durations > 30.0))
    info = {
        "dataset": args.dataset,
        "split": args.split,
        "revision": args.revision,
        "n_requested": args.n,
        "n": len(rows),
        "seed": args.seed if args.n > 0 else None,
        "buffer": args.buffer if args.n > 0 else None,
        "max_dur": args.max_dur,
        "rows_seen": seen,
        "n_over_30s": n_over_30s,
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
