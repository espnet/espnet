#!/usr/bin/env python3
"""Write a download manifest (local/v2/downloads/<corpus>.tsv) from a Zenodo record.

    python3 local/v2/zenodo_manifest.py <corpus> <record id> [--include REGEX] \
        [--licences cc-by-4.0,cc0-1.0] [--out local/v2/downloads]

One line per file of the record: corpus, ARCHIVES/<file name>, the record's
download link, the size and the md5 that Zenodo lists. Files whose name does not
match --include are left out. The record's licence (metadata.license.id) must be
one of --licences, otherwise nothing is written and the exit status is 1.
"""

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ACCEPTED = (
    "cc-by-4.0,cc-by-3.0,cc-by-2.0,cc-by-1.0,cc0-1.0,cc-zero,mit,apache-2.0,odc-by-1.0"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("corpus")
    ap.add_argument("record", help="Zenodo record id")
    ap.add_argument("--include", default="", help="keep only files matching this regex")
    ap.add_argument("--licences", default=ACCEPTED, help="accepted licence ids")
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "downloads"
    )
    a = ap.parse_args()

    url = f"https://zenodo.org/api/records/{a.record}"
    with urllib.request.urlopen(url, timeout=60) as r:
        rec = json.load(r)
    licence = (rec.get("metadata", {}).get("license") or {}).get("id", "")
    print(
        f"{a.corpus}: record {a.record} {rec['metadata'].get('title', '')!r}, licence {licence!r}"
    )
    if licence.lower() not in a.licences.split(","):
        print(
            f"{a.corpus}: licence {licence!r} not accepted; no manifest written",
            file=sys.stderr,
        )
        sys.exit(1)
    rows = []
    for f in rec["files"]:
        name = f["key"]
        if a.include and not re.search(a.include, name):
            continue
        check = f.get("checksum", "")  # "md5:<hex>"
        rows.append(
            f"{a.corpus}\tARCHIVES/{name}\t{f['links']['self']}\t{f['size']}\t{check}\n"
        )
    if not rows:
        print(f"{a.corpus}: no file matches {a.include!r}", file=sys.stderr)
        sys.exit(1)
    out = a.out / f"{a.corpus}.tsv"
    with out.open("w") as o:
        o.write("# corpus_dir\trelative_path\turl\tsize_bytes\tchecksum\tunpack\n")
        o.write(f"# Zenodo record {a.record} ({licence}): {url}\n")
        o.writelines(rows)
    total = sum(
        f["size"]
        for f in rec["files"]
        if not a.include or re.search(a.include, f["key"])
    )
    print(f"{a.corpus}: {len(rows)} files, {total / 1e9:.1f} GB -> {out}")


if __name__ == "__main__":
    main()
