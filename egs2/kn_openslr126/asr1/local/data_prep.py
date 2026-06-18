#!/usr/bin/env python3
import argparse, glob, os, random


def speaker_of(utt):
    # MILE_03_SP_2496_UTT_0116 -> MILE_03_SP_2496
    return utt.split("_UTT_")[0]


def collect(split_dir):
    """Return [(utt_id, wav_path, transcript), ...] for a split."""
    items = []
    audio_dir = os.path.join(split_dir, "audio_files")
    trans_dir = os.path.join(split_dir, "trans_files")
    for wav in sorted(glob.glob(os.path.join(audio_dir, "*.wav"))):
        utt = os.path.splitext(os.path.basename(wav))[0]
        txt = os.path.join(trans_dir, utt + ".txt")
        if not os.path.isfile(txt):
            continue
        with open(txt, encoding="utf-8") as f:
            transcript = f.read().strip()
        if not transcript:
            continue
        items.append((utt, wav, transcript))
    return items


def write_dir(outdir, items):
    os.makedirs(outdir, exist_ok=True)
    wav_scp, text, utt2spk, spk2utt = [], [], [], {}
    for utt, wav, tr in items:
        spk = speaker_of(utt)
        wav_scp.append(f"{utt} {wav}")
        text.append(f"{utt} {tr}")
        utt2spk.append(f"{utt} {spk}")
        spk2utt.setdefault(spk, []).append(utt)

    def dump(path, lines):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for line in sorted(lines):
                f.write(line + "\n")

    dump(os.path.join(outdir, "wav.scp"), wav_scp)
    dump(os.path.join(outdir, "text"), text)
    dump(os.path.join(outdir, "utt2spk"), utt2spk)
    with open(os.path.join(outdir, "spk2utt"), "w", encoding="utf-8", newline="\n") as f:
        for spk in sorted(spk2utt):
            f.write(f"{spk} {' '.join(sorted(spk2utt[spk]))}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--datadir", required=True, help="MILE Kannada root")
    ap.add_argument("--dev-spk-percent", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # test: provided by the corpus, use as-is
    test_items = collect(os.path.join(args.datadir, "test"))
    write_dir("data/test", test_items)

    # train: carve a speaker-disjoint dev set
    train_items = collect(os.path.join(args.datadir, "train"))
    spks = sorted({speaker_of(u) for u, _, _ in train_items})
    rng = random.Random(args.seed)
    rng.shuffle(spks)
    n_dev = max(1, int(len(spks) * args.dev_spk_percent / 100.0))
    dev_spks = set(spks[:n_dev])

    dev_items = [it for it in train_items if speaker_of(it[0]) in dev_spks]
    tr_items = [it for it in train_items if speaker_of(it[0]) not in dev_spks]
    write_dir("data/dev", dev_items)
    write_dir("data/train", tr_items)

    print(f"train: {len(tr_items)} utts / {len(spks) - n_dev} spk")
    print(f"dev:   {len(dev_items)} utts / {n_dev} spk")
    print(f"test:  {len(test_items)} utts")


if __name__ == "__main__":
    main()
