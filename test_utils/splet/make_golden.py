#!/usr/bin/env python3
"""Regenerate the sclite golden files that SPLET's parity tests assert against.

Run this only when the fixtures change or SCTK is upgraded, and commit the
result::

    python3 test_utils/splet/make_golden.py

The goldens exist so the parity tests still mean something on a machine without
SCTK built. They are sclite's own output, so regenerating them with a different
SCTK version is a deliberate act that should be visible in a diff.
"""

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent

# name -> (ref fixture, hyp fixture, extra sclite flags, tokenization)
#
# "word" joins tokens with a space, as tokenize_text does for --token_type word.
# "char" is the egs2 CER convention: one token per character, with a literal
# space becoming the <space> token (espnet2/text/char_tokenizer.py).
CASES = {
    "wer_default": ("ref.scp", "hyp_good.scp", [], "word"),
    "wer_poor": ("ref.scp", "hyp_poor.scp", [], "word"),
    "wer_case_sensitive": ("ref.scp", "hyp_good.scp", ["-s"], "word"),
    "cer_default": ("ref.scp", "hyp_good.scp", [], "char"),
    "cer_poor": ("ref.scp", "hyp_poor.scp", [], "char"),
    "cs_noascii": (
        "ref_cs.scp",
        "hyp_cs.scp",
        ["-e", "utf-8", "-c", "NOASCII"],
        "word",
    ),
}

SPACE_SYMBOL = "<space>"


def read_scp(path):
    """Read a kaldi-style ``uttid text`` file, keeping order and empty values."""
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        pairs.append((fields[0], fields[1] if len(fields) > 1 else ""))
    return pairs


def tokenize(text, mode):
    if mode == "word":
        return text.split()
    if mode == "char":
        return [SPACE_SYMBOL if ch == " " else ch for ch in text]
    raise ValueError(f"unknown tokenization {mode!r}")


def write_trn(pairs, mode, path):
    with path.open("w", encoding="utf-8") as handle:
        for key, text in pairs:
            # sclite's -i rm reads the speaker as the id text up to the first
            # "-" or "_", so a bare utterance id needs a speaker prefix.
            handle.write(f"{' '.join(tokenize(text, mode))} (spk-{key})\n")


def parse_sum_avg(report):
    """Pull the counts out of sclite's `-o dtl` report.

    The percentage table rounds, so the parity tests compare counts; the dtl
    report is the only place sclite prints them unrounded.
    """
    wanted = {
        "Percent Total Error": "errors",
        "Percent Correct": "hits",
        "Percent Substitution": "substitutions",
        "Percent Deletions": "deletions",
        "Percent Insertions": "insertions",
    }
    out = {}
    for line in report.splitlines():
        for label, key in wanted.items():
            if line.startswith(label):
                match = re.search(r"\(\s*(\d+)\)", line)
                if match:
                    out[key] = int(match.group(1))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sclite",
        default=shutil.which("sclite"),
        help="path to the sclite binary (default: the one on PATH)",
    )
    args = parser.parse_args()
    if not args.sclite:
        parser.error("sclite not found on PATH; build it with tools/Makefile")

    golden = {}
    for name, (ref_file, hyp_file, flags, mode) in CASES.items():
        ref = read_scp(HERE / ref_file)
        hyp = read_scp(HERE / hyp_file)
        if [k for k, _ in ref] != [k for k, _ in hyp]:
            raise SystemExit(f"{name}: fixture keys differ between ref and hyp")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            write_trn(ref, mode, tmp / "ref.trn")
            write_trn(hyp, mode, tmp / "hyp.trn")
            subprocess.run(
                [
                    args.sclite,
                    *flags,
                    "-r",
                    str(tmp / "ref.trn"),
                    "trn",
                    "-h",
                    str(tmp / "hyp.trn"),
                    "trn",
                    "-i",
                    "rm",
                    "-o",
                    "dtl",
                    "-O",
                    str(tmp),
                ],
                check=True,
                capture_output=True,
            )
            counts = parse_sum_avg((tmp / "hyp.trn.dtl").read_text(encoding="utf-8"))

        # Derived from sclite's own counts rather than by re-tokenizing here:
        # under -c NOASCII sclite splits non-ASCII words into characters, so a
        # length counted on this side would not be the one it scored against.
        counts["ref_len"] = (
            counts["hits"] + counts["substitutions"] + counts["deletions"]
        )
        counts["hyp_len"] = (
            counts["hits"] + counts["substitutions"] + counts["insertions"]
        )
        golden[name] = {
            "ref": ref_file,
            "hyp": hyp_file,
            "sclite_flags": flags,
            "tokenization": mode,
            "counts": counts,
        }

    # sclite prints its usage banner, version included, on stderr.
    banner = subprocess.run([args.sclite], capture_output=True, text=True)
    golden["_sctk_version"] = next(
        (
            line.strip()
            for line in (banner.stderr + banner.stdout).splitlines()
            if "SCTK Version" in line
        ),
        "unknown",
    )

    out = HERE / "sclite_golden.json"
    out.write_text(
        json.dumps(golden, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
