#!/usr/bin/env python3
"""Write a download manifest (local/v2/downloads/<corpus>.tsv) from a list of URLs.

    python3 local/v2/head_manifest.py <corpus> <url> [<url> ...] \
        --licence CC-BY-4.0 --evidence TEXT [--md5-sidecar] \
        [--out local/v2/downloads]

One line per URL: corpus, ARCHIVES/<file name of the URL>, the URL, the size
from a HEAD request (Content-Length; empty when the server gives none), and with
--md5-sidecar the md5 read from <url>.md5 (first word of the file; empty when
there is none). --licence must be accepted (manifest_common.licence_accepted),
otherwise nothing is written and the exit status is 1. The licence and
--evidence (where the licence is stated) go into the manifest's comment lines.

Example (Kielipankki SKN, CC BY 4.0 per README.txt):

    python3 local/v2/head_manifest.py kielipankki_skn \
        https://www.kielipankki.fi/download/SKN/README.txt \
        https://www.kielipankki.fi/download/SKN/skn_01-10.zip ... \
        --licence CC-BY-4.0 --md5-sidecar \
        --evidence "README.txt: Licensed under CC BY 4.0"
"""

import argparse
import re
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manifest_common import (  # noqa: E402
    DOWNLOADS,
    Fetcher,
    licence_accepted,
    log,
    write_manifest,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("corpus")
    ap.add_argument("urls", nargs="+")
    ap.add_argument("--licence", required=True, help="SPDX-like id, e.g. CC-BY-4.0")
    ap.add_argument("--evidence", required=True, help="where the licence is stated")
    ap.add_argument("--md5-sidecar", action="store_true", help="read <url>.md5")
    ap.add_argument("--out", type=Path, default=DOWNLOADS)
    a = ap.parse_args()
    if not licence_accepted(a.licence):
        log(f"{a.corpus}: licence {a.licence} not accepted; no manifest written")
        sys.exit(1)

    fetch, rows, total = Fetcher(), [], 0
    for url in a.urls:
        name = urllib.parse.unquote(urllib.parse.urlparse(url).path.rsplit("/", 1)[-1])
        size = fetch.size(url)
        check = ""
        if a.md5_sidecar and not url.endswith(".md5"):
            try:
                words = fetch.raw(url + ".md5").decode().split()
            except RuntimeError:
                words = []
            if words and re.fullmatch(r"[0-9a-f]{32}", words[0]):
                check = f"md5:{words[0]}"
        rows.append((f"ARCHIVES/{name}", url, size, check))
        total += int(size or 0)
        log(f"{a.corpus}: {name} {size or '?'} bytes {check}")
    out = write_manifest(
        a.out,
        a.corpus,
        rows,
        [f"licence {a.licence}: {a.evidence}", f"{len(rows)} files"],
    )
    print(f"{a.corpus}: {len(rows)} files, {total / 1e9:.2f} GB -> {out}")


if __name__ == "__main__":
    main()
