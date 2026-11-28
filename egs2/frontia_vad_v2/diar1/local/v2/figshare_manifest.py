#!/usr/bin/env python3
"""Write a download manifest (local/v2/downloads/<corpus>.tsv) from a figshare article.

    python3 local/v2/figshare_manifest.py <corpus> <article id> [--include REGEX] \
        [--licences "CC BY 4.0,CC0"] [--out local/v2/downloads]

One line per file of the article: corpus, ARCHIVES/<file name>, the figshare
download link, the size and the md5 that figshare computed. Files whose name does
not match --include, and link-only files, are left out. The article's licence
(license.name) must be one of --licences, otherwise nothing is written and the
exit status is 1.
"""

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ACCEPTED = "CC BY 4.0,CC BY 3.0,CC0,Apache 2.0,MIT"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("corpus")
    ap.add_argument("article", help="figshare article id")
    ap.add_argument("--include", default="", help="keep only files matching this regex")
    ap.add_argument("--licences", default=ACCEPTED, help="accepted licence names")
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "downloads"
    )
    a = ap.parse_args()

    url = f"https://api.figshare.com/v2/articles/{a.article}"
    with urllib.request.urlopen(url, timeout=60) as r:
        art = json.load(r)
    licence = (art.get("license") or {}).get("name", "")
    print(
        f"{a.corpus}: article {a.article} {art.get('title', '')!r}, licence {licence!r}"
    )
    if licence.lower() not in [x.strip().lower() for x in a.licences.split(",")]:
        print(
            f"{a.corpus}: licence {licence!r} not accepted; no manifest written",
            file=sys.stderr,
        )
        sys.exit(1)
    rows, total = [], 0
    for f in art.get("files") or []:
        if f.get("is_link_only") or (a.include and not re.search(a.include, f["name"])):
            continue
        check = f"md5:{f['computed_md5']}" if f.get("computed_md5") else ""
        rows.append(
            f"{a.corpus}\tARCHIVES/{f['name']}\t{f['download_url']}\t{f['size']}"
            f"\t{check}\n"
        )
        total += int(f["size"])
    if not rows:
        print(f"{a.corpus}: no file matches {a.include!r}", file=sys.stderr)
        sys.exit(1)
    out = a.out / f"{a.corpus}.tsv"
    with out.open("w") as o:
        o.write("# corpus_dir\trelative_path\turl\tsize_bytes\tchecksum\tunpack\n")
        o.write(f"# figshare article {a.article} ({licence}): {url}\n")
        o.writelines(rows)
    print(f"{a.corpus}: {len(rows)} files, {total / 1e9:.2f} GB -> {out}")


if __name__ == "__main__":
    main()
