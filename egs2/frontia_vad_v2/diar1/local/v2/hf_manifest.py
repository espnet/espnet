#!/usr/bin/env python3
"""Write a download manifest (local/v2/downloads/<corpus>.tsv) from a Hugging Face dataset.

    python3 local/v2/hf_manifest.py <corpus> <repo id> [--include REGEX] \
        [--licences cc-by-4.0,cc0-1.0,...] [--out local/v2/downloads]

One line per file of the dataset at its current commit (the URLs are pinned to
that commit): corpus, ARCHIVES/<path in the repo>, the resolve URL, the size and
the sha256 that the Hub lists for LFS files. The dataset must be ungated and
public, and its card licence one of --licences; otherwise nothing is written and
the exit status is 1.
"""

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ACCEPTED = "cc-by-4.0,cc-by-3.0,cc-by-2.0,cc0-1.0,mit,apache-2.0,odc-by,cdla-permissive-1.0,cdla-permissive-2.0"


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("corpus")
    ap.add_argument("repo", help="dataset repo id, e.g. owner/name")
    ap.add_argument("--include", default="", help="keep only paths matching this regex")
    ap.add_argument("--licences", default=ACCEPTED, help="accepted card licences")
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "downloads"
    )
    a = ap.parse_args()

    info = get(f"https://huggingface.co/api/datasets/{a.repo}")
    licence = (info.get("cardData") or {}).get("license", "")
    licence = licence[0] if isinstance(licence, list) else licence
    sha = info["sha"]
    print(
        f"{a.corpus}: {a.repo} @ {sha[:10]}, licence {licence!r}, gated {info.get('gated')}, private {info.get('private')}"
    )
    if info.get("gated") or info.get("private"):
        print(f"{a.corpus}: gated or private; no manifest written", file=sys.stderr)
        sys.exit(1)
    if str(licence).lower() not in a.licences.split(","):
        print(
            f"{a.corpus}: licence {licence!r} not accepted; no manifest written",
            file=sys.stderr,
        )
        sys.exit(1)
    tree = get(
        f"https://huggingface.co/api/datasets/{a.repo}/tree/{sha}?recursive=true"
    )
    rows, total = [], 0
    for e in tree:
        if e.get("type") != "file" or (
            a.include and not re.search(a.include, e["path"])
        ):
            continue
        lfs = e.get("lfs") or {}
        size = lfs.get("size", e.get("size", ""))
        check = f"sha256:{lfs['oid']}" if lfs.get("oid") else ""
        url = f"https://huggingface.co/datasets/{a.repo}/resolve/{sha}/{e['path']}"
        rows.append(f"{a.corpus}\tARCHIVES/{e['path']}\t{url}\t{size}\t{check}\n")
        total += int(size or 0)
    if not rows:
        print(f"{a.corpus}: no file matches {a.include!r}", file=sys.stderr)
        sys.exit(1)
    out = a.out / f"{a.corpus}.tsv"
    with out.open("w") as o:
        o.write("# corpus_dir\trelative_path\turl\tsize_bytes\tchecksum\tunpack\n")
        o.write(f"# Hugging Face dataset {a.repo} @ {sha} ({licence})\n")
        o.writelines(rows)
    print(f"{a.corpus}: {len(rows)} files, {total / 1e9:.2f} GB -> {out}")


if __name__ == "__main__":
    main()
