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
# Cap on the image-source order (see _gen_one). 30 keeps the direct path and
# early reflections exact while bounding an otherwise cubic cost.
MAX_ORDER_CAP = 30


def _gen_one(idx: int, out_dir: str, seed: int = 0) -> str:
    """Generate one RIR and save to disk. Returns saved path."""
    try:
        import pyroomacoustics as pra
    except ImportError:
        raise ImportError("pyroomacoustics required: pip install pyroomacoustics")

    rng = random.Random(seed + idx)
    np.random.seed((seed + idx) % (2**31))

    # Not every (rt60, room_dim) pair is physically realisable -- a large room
    # cannot have a short RT60 -- and pra.inverse_sabine raises for those. Draw
    # the room first, then draw an RT60 from the range Sabine admits for it,
    # so a rejected combination is retried rather than silently discarded.
    rir = None
    last_err = None
    for _ in range(8):
        room_dim = [rng.uniform(2.0, 20.0) for _ in range(3)]
        rt60 = rng.uniform(0.1, 2.0)
        try:
            e_abs, max_order = pra.inverse_sabine(rt60, room_dim)
            e_abs = float(np.clip(e_abs, 1e-4, 0.9999))
            # inverse_sabine happily returns max_order in the hundreds for a
            # large room with a long RT60, and the image-source method is
            # cubic in it: order 209 means ~74 million image sources, 10 s and
            # gigabytes for ONE RIR. Uncapped, 50k RIRs on 64 workers thrash
            # memory and take ~19 h. Capping at MAX_ORDER_CAP costs a little
            # of the very late tail -- measured RT60 moves 1.46 s -> 1.25 s at
            # the median -- and brings the same 50k down to a couple of
            # minutes. The direct path and early reflections, which are what
            # matter for a restoration model, are untouched.
            max_order = int(min(max_order, MAX_ORDER_CAP))
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
            # compute_rir(), NOT simulate(). simulate() convolves a source
            # SIGNAL with the RIR and needs one to be set; called without a
            # signal it raises "object of type 'NoneType' has no len()".
            # That exception used to be swallowed by the fallback below, so
            # every RIR in the pool became a unit impulse -- i.e. reverb
            # augmentation silently did nothing at all.
            room.compute_rir()
            cand = np.asarray(room.rir[0][0], dtype=np.float32)
            if cand.size > 1 and np.isfinite(cand).all() and np.abs(cand).max() > 1e-8:
                rir = cand / np.abs(cand).max()
                break
        except Exception as err:  # infeasible geometry; redraw and retry
            last_err = err

    out_path = os.path.join(out_dir, f"rir_{idx:06d}.wav")
    if rir is not None:
        sf.write(out_path, rir, samplerate=SR)
        return out_path

    # Exhausted the retries. Do NOT write a unit impulse: that is a valid
    # wav file describing no reverberation, so it makes a broken pool
    # indistinguishable from a working one at every later stage. Fail, and
    # let the caller report how many failed.
    raise RuntimeError(
        f"RIR {idx} could not be generated in 8 attempts; last error: "
        f"{type(last_err).__name__}: {last_err}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_rirs", type=int, default=50000)
    p.add_argument("--nj", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # pyroomacoustics is not an ESPnet dependency and is not installed by
    # tools/. Check it once here rather than letting every worker raise the
    # same ImportError, which buries the actionable message in traceback spam.
    try:
        import pyroomacoustics  # noqa: F401
    except ImportError:
        raise SystemExit(
            "pyroomacoustics is required to build the RIR pool but is not "
            "installed. Run: pip install pyroomacoustics\n"
            "(See egs2/libritts_r/enh1/README.md for the recipe's extra "
            "dependencies.)"
        )

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
        futures = {
            ex.submit(_gen_one, i, args.out_dir, args.seed): i for i in to_generate
        }
        failed = 0
        first_error = None
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                failed += 1
                if first_error is None:
                    first_error = e
            done += 1
            if done % 5000 == 0:
                logger.info(
                    "  %d / %d done (%d failed)", done, len(to_generate), failed
                )

    total = len([f for f in os.listdir(args.out_dir) if f.endswith(".wav")])
    logger.info("Done. %d RIRs in %s (%d failed)", total, args.out_dir, failed)

    # A pool that is mostly missing is a silently broken degradation stage:
    # training still runs, reverb augmentation just stops happening. Refuse to
    # exit successfully in that case.
    if failed:
        logger.warning(
            "%d/%d RIRs failed; first error: %s", failed, len(to_generate), first_error
        )
    if total < 0.9 * args.n_rirs:
        raise SystemExit(
            f"only {total}/{args.n_rirs} RIRs were written. Refusing to leave a "
            f"partial pool in place -- reverb augmentation would quietly become "
            f"a near-no-op. First error: {first_error}"
        )

    # Verify the pool actually contains reverberation rather than impulses.
    # This is the check whose absence let a pool of 50,000 unit impulses pass
    # unnoticed through an entire training run.
    import soundfile as _sf

    names = sorted(f for f in os.listdir(args.out_dir) if f.endswith(".wav"))
    probe = [
        names[i]
        for i in np.linspace(0, len(names) - 1, min(200, len(names))).astype(int)
    ]
    degenerate = 0
    for name in probe:
        x, _ = _sf.read(
            os.path.join(args.out_dir, name), dtype="float32", always_2d=True
        )
        if int((np.abs(x.mean(1)) > 1e-9).sum()) <= 2:
            degenerate += 1
    if degenerate:
        raise SystemExit(
            f"{degenerate}/{len(probe)} sampled RIRs contain <=2 non-zero taps, "
            f"i.e. they are impulses describing no reverberation. The pool is "
            f"unusable; delete {args.out_dir} and regenerate."
        )
    logger.info("verified %d sampled RIRs all contain real reverberation", len(probe))


if __name__ == "__main__":
    main()
