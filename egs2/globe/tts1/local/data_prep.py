#!/usr/bin/env python3
import argparse, os
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset, Audio

def write_kaldi_dir(items, outdir, fs_out):
    os.makedirs(outdir, exist_ok=True)
    spk2utt = defaultdict(list)
    with open(f"{outdir}/wav.scp","w") as wscp, \
         open(f"{outdir}/text","w")   as txtf, \
         open(f"{outdir}/utt2spk","w") as u2s, \
         open(f"{outdir}/utt2dur","w") as u2d:
        for utt, spk, wav, text, dur in items:
            wscp.write(f"{utt} sox -t flac {wav} -r {fs_out} -t wav - |\n")
            txtf.write(f"{utt} {text}\n")
            u2s.write(f"{utt} {spk}\n")
            u2d.write(f"{utt} {dur:.3f}\n")
            spk2utt[spk].append(utt)
    with open(f"{outdir}/spk2utt","w") as s2u:
        for spk, utts in spk2utt.items():
            s2u.write(f"{spk} {' '.join(utts)}\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_repo",   required=True,
                  help="huggingface dataset repo, e.g. MushanW/GLOBE_V2")
    p.add_argument("--train_set", required=True)
    p.add_argument("--dev_set",   required=True)
    p.add_argument("--test_set",  required=True)
    p.add_argument("--dest_path", required=True)
    p.add_argument("--fs_out",    type=int, default=24000)
    args = p.parse_args()

    for split in (args.train_set, args.dev_set, args.test_set):
        print(f">>> Preparing {split} → data/{split}")
        ds = load_dataset(
            args.hf_repo,
            split=split,            # must be exactly "train"/"dev"/"test"
            cache_dir="downloads/cache/",
            streaming=False,
        )
        ds = ds.cast_column("audio", Audio(decode=False))

        items = []
        for ex in tqdm(ds, desc=split):
            wav   = ex["audio"]["path"]
            spk   = ex["speaker_id"]
            text  = ex["transcript"].strip()
            dur   = float(ex["duration"])
            utt   = f"{spk}_{os.path.basename(wav).split('.')[0]}"
            items.append((utt, spk, wav, text, dur))

        write_kaldi_dir(
            items,
            outdir=os.path.join(args.dest_path, split),
            fs_out=args.fs_out,
        )

    print("✅  Data prep done.")

if __name__=="__main__":
    main()
