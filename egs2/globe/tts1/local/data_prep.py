#!/usr/bin/env python3
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map
from datasets import load_dataset, Audio
import pathlib
import soundfile as sf
import os
import sys
import argparse, os, time
from datasets import load_dataset, Audio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from huggingface_hub import login

def write_kaldi_dir(items, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    spk2utt = defaultdict(list)

    # open all Kaldi files
    with (outdir / "wav.scp").open("w")  as wscp, \
         (outdir / "text").open("w")    as txtf, \
         (outdir / "utt2spk").open("w") as u2s, \
         (outdir / "utt2dur").open("w") as u2d:

        for utt, spk, wav_path, text, dur in items:
            wscp.write(f"{utt} {wav_path}\n")
            txtf.write(f"{utt} {text}\n")
            u2s.write(f"{utt} {spk}\n")
            u2d.write(f"{utt} {dur:.3f}\n")
            spk2utt[spk].append(utt)

    # write spk2utt
    with (outdir / "spk2utt").open("w") as s2u:
        for spk, utts in spk2utt.items():
            s2u.write(f"{spk} {' '.join(utts)}\n")

def process_and_dump(ex, split, dump_root, fs_out):
    """
    Download & decode one example, write out a WAV,
    return the Kaldi-tuple for it.
    """
    # make sure audio is decoded
    try:
        arr, sr = ex["audio"]["array"], ex["audio"]["sampling_rate"]
        spk      = ex["speaker_id"]
        text     = ex["transcript"].strip()
        dur      = float(ex.get("duration", len(arr)/sr))

        stem     = Path(ex["audio"]["path"]).stem  # original HF cache name
        utt      = f"{spk}_{stem}"
        out_dir  = dump_root / split
        out_dir.mkdir(parents=True, exist_ok=True)

        out_wav  = out_dir / f"{utt}.wav"
        # write only once
        if not out_wav.exists():
            sf.write(out_wav, arr, fs_out)

        return (
            utt,
            spk,
            str(out_wav),
            text,
            dur
        )
    except Exception as e:
        print(f"⚠️  Failed to process {ex.get('audio', {}).get('path','?')}: {e}",
              file=sys.stderr)
        return None

def main():
    p = argparse.ArgumentParser(
        description="GLOBE-V2 → Kaldi data prep (on-the-fly sox pipeline)"
    )
    p.add_argument("--hf_repo",   required=True,
                   help="HuggingFace dataset, e.g. MushanW/GLOBE_V2")
    p.add_argument("--train_set", required=True)
    p.add_argument("--dev_set",   required=True)
    p.add_argument("--test_set",  required=True)
    p.add_argument("--dest_path", required=True,
                   help="output Kaldi data root (train/, val/, test/)")
    p.add_argument("--fs_out",    type=int, default=44100,
                   help="sox will resample to this rate")
    p.add_argument("--jobs",      type=int, default=8,
                   help="number of parallel threads")
    args = p.parse_args()
    if "HF_TOKEN" in os.environ:
        print('🔑 Using HuggingFace token from environment variable HF_TOKEN')
        login(os.environ["HF_TOKEN"]) 
    dump_root = Path("downloads/audio")
    for split in (args.train_set, args.dev_set, args.test_set):
        print(f"\n⏳ Preparing split “{split}” → data/{split}")
        ds = ( load_dataset(args.hf_repo, split=split,
                            cache_dir="downloads/cache",
                            verification_mode="no_checks",
                            streaming=False)
               .cast_column("audio", Audio(decode=True)) )
        items = []
        with ThreadPoolExecutor(args.jobs) as exe:
            # 1️⃣ seed the pool
            in_iter = iter(ds)
            futures = {exe.submit(process_and_dump, next(in_iter),
                                split, dump_root, args.fs_out)
                    for _ in range(args.jobs)}

            with tqdm(desc=f"Dumping {split}") as pbar:
                while futures:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)

                    # 2️⃣ collect results & top-up the pool
                    for fut in done:
                        result = fut.result()          # raise exception here if any
                        if result is not None:
                            items.append(result)
                        pbar.update()

                        try:
                            fut_new = exe.submit(process_and_dump, next(in_iter),
                                                split, dump_root, args.fs_out)
                            futures.add(fut_new)       # keep pool full
                        except StopIteration:
                            pass
        items = [x for x in items if x is not None]
        write_kaldi_dir(items, Path(args.dest_path) / split)

    print(f"\n✅ All done; WAVs in downloads/audio/* and Kaldi data under {args.dest_path}")


if __name__=="__main__":
    main()
