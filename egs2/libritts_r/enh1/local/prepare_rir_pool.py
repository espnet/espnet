#!/usr/bin/env python3
"""Pre-generate Room Impulse Responses (RIRs) for on-the-fly degradation.

Saves RIRs as 16kHz mono WAV files to out_dir/rir_XXXXXX.wav.
At training time, collate_fn randomly samples from this pool (CPU-only,
no pyroomacoustics overhead per step).

Usage:
    python local/prepare_rir_pool.py --out_dir data/rir_pool --n_rirs 50000 --nj 16
"""

import argparse
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import soundfile as sf

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

SR = 16000  # RIRs always at 16kHz; collate_fn resamples if needed


def _gen_one(idx: int, out_dir: str) -> str:
    """Generate one RIR and save to disk. Returns saved path."""
    try:
        import pyroomacoustics as pra
    except ImportError:
        raise ImportError("pyroomacoustics required: pip install pyroomacoustics")

    rng = random.Random(idx)
    np.random.seed(idx % (2**31))

    rt60 = rng.uniform(0.1, 2.0)
    room_dim = [rng.uniform(2.0, 20.0) for _ in range(3)]

    try:
        e_abs, max_order = pra.inverse_sabine(rt60, room_dim)
        e_abs = float(np.clip(e_abs, 1e-4, 0.9999))
        room = pra.ShoeBox(
            room_dim,
            fs=SR,
            materials=pra.Material(e_abs),
            max_order=max_order,
        )
        src_pos = [d * rng.uniform(0.1, 0.9) for d in room_dim]
        mic_pos = [d * rng.uniform(0.1, 0.9) for d in room_dim]
        room.add_source(src_pos)
        room.add_microphone(mic_pos)
        room.simulate()
        rir = room.rir[0][0].astype(np.float32)

        # Normalize
        peak = np.abs(rir).max()
        if peak > 1e-8:
            rir = rir / peak

        out_path = os.path.join(out_dir, f"rir_{idx:06d}.wav")
        sf.write(out_path, rir, samplerate=SR)
        return out_path
    except Exception as e:
        logger.debug("RIR %d failed (%s), generating white noise fallback", idx, e)
        # Fallback: Gaussian impulse (approx dry signal, no reverb effect)
        rir = np.zeros(SR // 10, dtype=np.float32)
        rir[0] = 1.0
        out_path = os.path.join(out_dir, f"rir_{idx:06d}.wav")
        sf.write(out_path, rir, samplerate=SR)
        return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_rirs", type=int, default=50000)
    p.add_argument("--nj", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Check how many already exist
    existing = {f for f in os.listdir(args.out_dir) if f.endswith(".wav")}
    existing_count = len(existing)
    if existing_count >= args.n_rirs:
        logger.info(
            "Already have %d RIRs (≥ %d requested). Nothing to do.",
            existing_count,
            args.n_rirs,
        )
        return

    to_generate = [i for i in range(args.n_rirs) if f"rir_{i:06d}.wav" not in existing]
    logger.info(
        "Generating %d RIRs with %d workers → %s",
        len(to_generate),
        args.nj,
        args.out_dir,
    )

    done = 0
    with ProcessPoolExecutor(max_workers=args.nj) as ex:
        futures = {ex.submit(_gen_one, i, args.out_dir): i for i in to_generate}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logger.warning("Worker failed: %s", e)
            done += 1
            if done % 5000 == 0:
                logger.info("  %d / %d done", done, len(to_generate))

    total = len([f for f in os.listdir(args.out_dir) if f.endswith(".wav")])
    logger.info("Done. %d RIRs in %s", total, args.out_dir)


if __name__ == "__main__":
    main()
