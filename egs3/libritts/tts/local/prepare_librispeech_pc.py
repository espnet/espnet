"""Build the LibriSpeech-PC cross-sentence eval manifest for the F5 recipe.

Consumes the F5-TTS repo's ``librispeech_pc_test_clean_cross_sentence.lst``
(6 tab-separated columns: ref_utt, ref_dur, ref_txt, gen_utt, gen_dur,
gen_txt; same-speaker prompt/target pairs) and the read-only LibriSpeech
``test-clean`` tree, and writes the recipe-side manifest:

    gen_utt \t gen_text \t ref_utt \t ref_wav_path \t ref_text

``--gt_wav_dir`` additionally builds a flat directory of ``<gen_utt>.wav``
symlinks to the ground-truth flacs, which lets the official
``eval_librispeech_test_clean.py`` score the real audio (its GT mode is a
code edit we avoid; audio loaders sniff the container, so a .wav-named
symlink to flac is read correctly).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _utt_to_flac(root: Path, utt: str) -> Path:
    spk, chap, _ = utt.split("-")
    path = root / spk / chap / f"{utt}.flac"
    if not path.exists():
        raise FileNotFoundError(f"Missing LibriSpeech audio: {path}")
    return path


def _read_lst(lst_path: Path):
    rows = []
    with Path(lst_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            ref_utt, _ref_dur, ref_txt, gen_utt, _gen_dur, gen_txt = line.split("\t")
            rows.append((ref_utt, ref_txt, gen_utt, gen_txt))
    return rows


def build_manifest(lst_path, test_clean_root, out_tsv) -> int:
    root = Path(test_clean_root).resolve()
    out = Path(out_tsv)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = _read_lst(lst_path)
    with out.open("w", encoding="utf-8") as f:
        for ref_utt, ref_txt, gen_utt, gen_txt in rows:
            ref_wav = _utt_to_flac(root, ref_utt)
            _utt_to_flac(root, gen_utt)  # fail fast if the target audio is absent
            f.write(f"{gen_utt}\t{gen_txt}\t{ref_utt}\t{ref_wav}\t{ref_txt}\n")
    return len(rows)


def build_gt_wav_dir(lst_path, test_clean_root, out_dir) -> int:
    root = Path(test_clean_root).resolve()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = _read_lst(lst_path)
    for _ref_utt, _ref_txt, gen_utt, _gen_txt in rows:
        target = _utt_to_flac(root, gen_utt)
        link = out / f"{gen_utt}.wav"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lst", required=True)
    ap.add_argument("--test_clean_root", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--gt_wav_dir", default=None)
    args = ap.parse_args()
    n = build_manifest(args.lst, args.test_clean_root, args.out_tsv)
    print(f"Wrote {n} manifest rows to {args.out_tsv}")
    if args.gt_wav_dir:
        n = build_gt_wav_dir(args.lst, args.test_clean_root, args.gt_wav_dir)
        print(f"Linked {n} GT wavs into {args.gt_wav_dir}")


if __name__ == "__main__":
    main()
