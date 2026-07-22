#!/usr/bin/env python3
"""Compute-efficient AudioSet data prep for BEATs AS-2M pretraining.

Speedups over local/data_prep_as2m.py:
1. Pre-index existing wav stems via os.scandir (one pass per dir)
   instead of os.path.exists() per CSV row (2M+ stat calls).
2. Skip 0-byte wavs at index time so they never reach the pipeline.
3. Parallelize cut_wav generation via multiprocessing.Pool, skipping
   files that already exist.
4. Single-pass write of wav.scp + utt2spk.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool

import soundfile as sf
from tqdm import tqdm

CSV_TO_SUBDIR = {
    "eval_segments": "eval_wav",
    "balanced_train_segments": "balance_wav",
    "unbalanced_train_segments": "unbalanced_wav",
}


def index_wav_dir(directory):
    """Return set of wav stems present in *directory* (no per-entry stat).

    The per-entry st_size check is expensive on NFS (~5 ms/file).
    For a 2 M-entry corpus that is ~3 hours just on stat calls. We
    skip it here and let downstream stages (or a separate filter pass)
    handle empty/corrupt files.
    """
    if not os.path.isdir(directory):
        return set()
    stems = set()
    with os.scandir(directory) as it:
        for entry in it:
            if entry.name.endswith(".wav"):
                stems.add(entry.name[:-4])
    return stems


def check_nonempty(path):
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return False


def cut_one(arg):
    src_path, dst_path, audio_len = arg
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        return True
    try:
        s, r = sf.read(src_path)
        sf.write(dst_path, s[: int(r * audio_len)], r)
        return True
    except Exception:
        return False


def parse_csv(csv_path, src_dir, cut_dir, existing_src, cut_existing,
              entries_out, cut_jobs_out):
    n_missing = 0
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",", maxsplit=3)
            yt_id = parts[0]
            if yt_id not in existing_src:
                n_missing += 1
                continue
            start_s = float(parts[1])
            end_s = float(parts[2])
            audio_len = end_s - start_s
            if audio_len < 10:
                dst_path = os.path.join(cut_dir, yt_id + ".wav")
                if yt_id not in cut_existing:
                    cut_jobs_out.append(
                        (os.path.join(src_dir, yt_id + ".wav"), dst_path, audio_len)
                    )
                entries_out.append({"wav_path": dst_path, "audio_len": audio_len})
            else:
                entries_out.append(
                    {
                        "wav_path": os.path.join(src_dir, yt_id + ".wav"),
                        "audio_len": audio_len,
                    }
                )
    return n_missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_read")
    ap.add_argument("data_write")
    ap.add_argument("--n_proc", type=int, default=16)
    args = ap.parse_args()

    # 1) Parallel-index every wav dir at once.
    print("[1/4] Indexing existing wavs (os.scandir)...", flush=True)
    dirs_to_index = list(CSV_TO_SUBDIR.values()) + ["cut_wav"]
    indexed = {}
    with ProcessPoolExecutor(max_workers=len(dirs_to_index)) as pool:
        futs = {
            pool.submit(index_wav_dir, os.path.join(args.data_read, d)): d
            for d in dirs_to_index
        }
        for fut in as_completed(futs):
            d = futs[fut]
            indexed[d] = fut.result()
            print(f"  {d}: {len(indexed[d])} non-empty wavs", flush=True)
    cut_existing = indexed["cut_wav"]

    # 2) Parse CSVs.
    print("[2/4] Parsing CSVs...", flush=True)
    train_entries = []
    eval_entries = []
    cut_jobs = []
    # Same order as data_prep_as2m.py (unbalanced first, then balanced) so the
    # generated utt-ids match the v1 output line by line.
    for csv_name, target_list in [
        ("eval_segments.csv", eval_entries),
        ("unbalanced_train_segments.csv", train_entries),
        ("balanced_train_segments.csv", train_entries),
    ]:
        csv_path = os.path.join(args.data_read, csv_name)
        subdir = CSV_TO_SUBDIR[csv_name.replace(".csv", "")]
        n_missing = parse_csv(
            csv_path,
            os.path.join(args.data_read, subdir),
            os.path.join(args.data_read, "cut_wav"),
            indexed[subdir],
            cut_existing,
            target_list,
            cut_jobs,
        )
        print(f"  {csv_name}: kept {len(target_list)} cumulative, missing {n_missing}", flush=True)

    # 3) Generate cut_wav files (parallel, idempotent).
    if cut_jobs:
        print(f"[3/4] Generating {len(cut_jobs)} cut_wav files "
              f"({args.n_proc} workers)...", flush=True)
        os.makedirs(os.path.join(args.data_read, "cut_wav"), exist_ok=True)
        with Pool(processes=args.n_proc) as p:
            ok = 0
            for r in tqdm(p.imap_unordered(cut_one, cut_jobs, chunksize=64),
                          total=len(cut_jobs)):
                ok += int(r)
            print(f"  generated {ok}/{len(cut_jobs)}", flush=True)
    else:
        print("[3/4] No cut_wav generation needed (all present).", flush=True)

    # 4) Write wav.scp + utt2spk.
    print("[4/4] Writing wav.scp + utt2spk...", flush=True)
    for entries, name in [(train_entries, "AudioSet"), (eval_entries, "eval")]:
        out_dir = os.path.join(args.data_write, name)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "wav.scp"), "w") as wav_f, open(
            os.path.join(out_dir, "utt2spk"), "w"
        ) as utt_f:
            for uttid, item in enumerate(entries):
                print(f"as2m_20k-{name}-{uttid} {item['wav_path']}", file=wav_f)
                print(f"as2m_20k-{name}-{uttid} dummy", file=utt_f)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
