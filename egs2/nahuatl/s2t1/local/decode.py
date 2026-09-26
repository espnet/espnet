#!/usr/bin/env python3
"""Prompt-conditioned decoding + CER scoring for the Nahuatl OWSM recipe.

s2t.sh's decode path (stage 12) feeds only the speech to inference, so it cannot
supply the dialect prompt this recipe conditions on. This driver decodes each
test set with Speech2Text, passing each utterance its own ``text.prev``
("nahuatl <region>") as the decoder prompt while keeping a single ``<na>``
language symbol, then scores character CER the same way local/score.sh does
(strip every ``<...>`` token from both sides, tokenize to characters, run
sclite). It prints a per-set CER and a combined CER over all test sets.

Usage:
    python local/decode.py --exp_dir exp/s2t_train_owsm_v4_nahuatl_raw_bpe50000 \
        --test_sets "nahuatl_hidalgo_test nahuatl_orizaba_zongolica_test \
                     nahuatl_zacatlan_tepetzintla_test"
"""

import argparse
import io
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

_SPECIAL = re.compile(r"<[^>]*>")


def load_audio(entry: str):
    """Read one wav.scp value: a Kaldi pipe ('... |') or a plain audio path."""
    entry = entry.strip()
    if entry.endswith("|"):
        raw = subprocess.check_output(entry[:-1], shell=True)
        data, sr = sf.read(io.BytesIO(raw))
    else:
        data, sr = sf.read(entry)
    if data.ndim > 1:
        data = data[:, 0]
    return data.astype(np.float32), sr


def read_kv(path: Path) -> dict:
    d = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        k, _, v = line.partition(" ")
        d[k] = v
    return d


def char_tokenize(lines, bpemodel=None):
    """Char-tokenize a list of 'text' lines via espnet2.bin.tokenize_text,
    matching local/score.sh so the CER is computed identically."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "espnet2.bin.tokenize_text",
            "-f",
            "2-",
            "--input",
            "-",
            "--output",
            "-",
            "--cleaner",
            "none",
            "--token_type",
            "char",
        ],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit("tokenize_text failed")
    return proc.stdout.splitlines()


def score_dir(score_dir: Path, hyp: dict, ref: dict, utt2spk: dict) -> str:
    """Write hyp.trn/ref.trn (char-tokenized, symmetric) and run sclite.
    Returns the sclite Sum/Avg summary line."""
    score_dir.mkdir(parents=True, exist_ok=True)
    uids = [u for u in hyp if u in ref]

    def trn(texts):
        toks = char_tokenize([f"x {t}" for t in texts])
        return toks

    hyp_toks = trn([hyp[u] for u in uids])
    ref_toks = trn([_SPECIAL.sub("", ref[u]) for u in uids])
    with open(score_dir / "hyp.trn", "w") as fh, open(score_dir / "ref.trn", "w") as fr:
        for u, ht, rt in zip(uids, hyp_toks, ref_toks):
            tag = f"({utt2spk.get(u, u)}-{u})"
            fh.write(f"{ht} {tag}\n")
            fr.write(f"{rt} {tag}\n")
    with open(score_dir / "result.txt", "w") as out:
        subprocess.run(
            [
                "sclite",
                "-r",
                str(score_dir / "ref.trn"),
                "trn",
                "-h",
                str(score_dir / "hyp.trn"),
                "trn",
                "-i",
                "rm",
                "-o",
                "all",
                "stdout",
            ],
            stdout=out,
            check=True,
        )
    for line in (score_dir / "result.txt").read_text().splitlines():
        if "Sum/Avg" in line:
            return line.strip()
    return "(no Sum/Avg line)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--model_file", default="valid.acc.ave.pth")
    ap.add_argument("--decode_config", default="conf/decode.yaml")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--test_sets", required=True, help="space-separated data dirs")
    ap.add_argument("--lang_sym", default="<na>")
    ap.add_argument("--task_sym", default="<asr>")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default=None, help="default: <exp_dir>/decode_prompt")
    args = ap.parse_args()

    from espnet2.bin.s2t_inference import Speech2Text

    cfg = yaml.safe_load(open(args.decode_config)) or {}
    exp = Path(args.exp_dir)
    out_root = Path(args.out_dir) if args.out_dir else exp / "decode_prompt"

    s2t = Speech2Text(
        s2t_train_config=str(exp / "config.yaml"),
        s2t_model_file=str(exp / args.model_file),
        device=args.device,
        beam_size=int(cfg.get("beam_size", 5)),
        ctc_weight=float(cfg.get("ctc_weight", 0.3)),
        lang_sym=args.lang_sym,
        task_sym=args.task_sym,
        nbest=1,
    )

    all_hyp: dict = {}
    all_ref: dict = {}
    all_u2s: dict = {}
    summaries = []
    for dset in args.test_sets.split():
        ddir = Path(args.data_root) / dset
        wav = read_kv(ddir / "wav.scp")
        text = read_kv(ddir / "text")
        prev = read_kv(ddir / "text.prev")
        u2s = read_kv(ddir / "utt2spk")
        hyp = {}
        for uid, entry in wav.items():
            speech, _ = load_audio(entry)
            results = s2t(speech, text_prev=prev.get(uid, ""))
            hyp[uid] = results[0][3]  # text_nospecial
        summary = score_dir(out_root / dset / "score_cer", hyp, text, u2s)
        print(f"[{dset}] {summary}", flush=True)
        summaries.append((dset, summary))
        all_hyp.update(hyp)
        all_ref.update(text)
        all_u2s.update(u2s)

    combined = score_dir(out_root / "combined" / "score_cer", all_hyp, all_ref, all_u2s)
    print("=" * 70)
    for dset, summary in summaries:
        print(f"[{dset}] {summary}")
    print(f"[combined] {combined}")
    print(f"Full results under {out_root}")


if __name__ == "__main__":
    main()
