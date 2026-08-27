#!/usr/bin/env python3
"""Resample wav.scp to target SR, writing actual wav files.
Handles plain paths and pipe commands (sox ... |).
"""
import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf
import numpy as np


def _resample_one(utt, src, wav_dir, target_sr):
    out_path = Path(wav_dir) / f"{utt}.wav"
    if out_path.exists():
        return utt, str(out_path), None

    try:
        # pipe command or plain path — use sox for both
        if src.endswith("|"):
            # e.g. "sox /path/to/file.wav -r 24000 -t wav - |"
            cmd = src.rstrip(" |").split()
            # replace or add -r target_sr
            result = subprocess.run(
                cmd + ["-r", str(target_sr), "-c", "1", str(out_path)],
                capture_output=True, check=True
            )
        else:
            subprocess.run(
                ["sox", src, "-r", str(target_sr), "-c", "1", str(out_path)],
                capture_output=True, check=True
            )
        return utt, str(out_path), None
    except Exception as e:
        return utt, None, str(e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_scp",  required=True)
    p.add_argument("--output_scp", required=True)
    p.add_argument("--wav_dir",    required=True)
    p.add_argument("--target_sr",  type=int, default=16000)
    p.add_argument("--nj",         type=int, default=8)
    args = p.parse_args()

    Path(args.wav_dir).mkdir(parents=True, exist_ok=True)

    entries = []
    with open(args.input_scp) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                entries.append((parts[0], parts[1]))

    print(f"Resampling {len(entries)} utterances to {args.target_sr}Hz ...")

    results = {}
    skipped = 0
    with ProcessPoolExecutor(max_workers=args.nj) as ex:
        futures = {
            ex.submit(_resample_one, utt, src, args.wav_dir, args.target_sr): utt
            for utt, src in entries
        }
        for i, fut in enumerate(as_completed(futures), 1):
            utt, out_path, err = fut.result()
            if err:
                print(f"  SKIP {utt}: {err}")
                skipped += 1
            else:
                results[utt] = out_path
            if i % 5000 == 0:
                print(f"  {i}/{len(entries)} done")

    # Write output scp in original order
    with open(args.output_scp, "w") as f:
        for utt, _ in entries:
            if utt in results:
                f.write(f"{utt} {results[utt]}\n")

    print(f"Done. {len(results)} written, {skipped} skipped → {args.output_scp}")


if __name__ == "__main__":
    main()