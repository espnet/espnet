#!/usr/bin/env python3
"""Write a download manifest and licence table from a public S3 bucket listing.

    python3 local/v2/s3_manifest.py <corpus> --prefix P [--prefix P ...] \
        --include REGEX [--exclude REGEX] --licence CC-BY-4.0 \
        [--bucket https://s3.amazonaws.com/download.opencontent.netflix.com/] \
        [--creator "Netflix, Inc."] [--source-page URL] [--out local/v2/downloads]

1. Lists every key under each --prefix with ListObjectsV2 (anonymous,
   continuation tokens followed).
2. Keeps the keys that match --include and do not match --exclude.
3. Looks in the top folder of each kept key for a licence file (a key whose name
   contains "licen") and uses its URL as the licence URL of the key; a top
   folder without one makes the script stop with exit status 1. --licence must
   be accepted (manifest_common.licence_accepted).

Rows: corpus, ARCHIVES/<key>, the bucket URL of the key, the size, and the md5
when the ETag is one (single-part uploads; multipart ETags contain "-"). The
licence table gives --licence, the licence file URL, --creator and --source-page.

Netflix Open Content (bucket download.opencontent.netflix.com, CC BY 4.0 per the
licence file in each title folder):

    python3 local/v2/s3_manifest.py netflix_meridian --prefix Meridian/ \
        --include '_EN_20_B\\.mxf$|/audio/atmos/.*\\.wav$|/subtitles/.*\\.stl$|licen'
    python3 local/v2/s3_manifest.py netflix_odas --prefix ODAS/ \
        --include '/Mixes/.*_ST_(LCR|MS)\\.wav$|^ODAS/LICENSE$'
    python3 local/v2/s3_manifest.py netflix_open_misc \
        --prefix Nocturne/ --prefix sparks-mxf-tracks/ \
        --include 'Nocture_ADM\\.wav$|LtRt_Mixdown\\.mxf$|final_mix.*\\.mxf$|licen'
"""

import argparse
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manifest_common import (  # noqa: E402
    DOWNLOADS,
    Fetcher,
    licence_accepted,
    log,
    write_licences,
    write_manifest,
)

NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
NETFLIX = "https://s3.amazonaws.com/download.opencontent.netflix.com/"


def list_keys(fetch, bucket, prefix):
    """(key, size, etag) of every object under prefix."""
    out, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        x = ET.fromstring(fetch.raw(bucket + "?" + urllib.parse.urlencode(q)))
        for c in x.findall("s:Contents", NS):
            out.append(
                (
                    c.findtext("s:Key", "", NS),
                    c.findtext("s:Size", "", NS),
                    c.findtext("s:ETag", "", NS).strip('"'),
                )
            )
        token = x.findtext("s:NextContinuationToken", None, NS)
        if not token:
            return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("corpus")
    ap.add_argument("--bucket", default=NETFLIX, help="bucket URL ending in /")
    ap.add_argument("--prefix", action="append", required=True)
    ap.add_argument("--include", required=True, help="keep keys matching this")
    ap.add_argument("--exclude", default="", help="drop keys matching this")
    ap.add_argument("--licence", default="CC-BY-4.0")
    ap.add_argument("--creator", default="Netflix, Inc.")
    ap.add_argument("--source-page", default="https://opencontent.netflix.com/")
    ap.add_argument("--out", type=Path, default=DOWNLOADS)
    a = ap.parse_args()
    if not licence_accepted(a.licence):
        log(f"{a.corpus}: licence {a.licence} not accepted; no manifest written")
        sys.exit(1)

    fetch = Fetcher(interval=0.5)
    keys = []
    for p in a.prefix:
        got = list_keys(fetch, a.bucket, p)
        log(f"{a.corpus}: {len(got)} keys under {p}")
        keys += got
    licence_files = defaultdict(list)
    for k, _, _ in keys:
        if "licen" in k.rsplit("/", 1)[-1].lower():
            licence_files[k.split("/", 1)[0]].append(k)
    rows, lic_rows, nbytes = [], [], 0
    for k, size, etag in keys:
        if k.endswith("/") or not re.search(a.include, k):
            continue
        if a.exclude and re.search(a.exclude, k):
            continue
        top = k.split("/", 1)[0]
        if not licence_files[top]:
            log(f"{a.corpus}: no licence file in {top}/; no manifest written")
            sys.exit(1)
        url = a.bucket + urllib.parse.quote(k)
        check = f"md5:{etag}" if re.fullmatch(r"[0-9a-f]{32}", etag) else ""
        rel = f"ARCHIVES/{k}"
        rows.append((rel, url, size, check))
        lic_url = a.bucket + urllib.parse.quote(licence_files[top][0])
        lic_rows.append((rel, a.licence, lic_url, a.creator, a.source_page))
        nbytes += int(size or 0)
        log(f"{a.corpus}: {int(size) / 1e9:8.3f} GB {k}")
    rows.sort()
    lic_rows.sort()
    out = write_manifest(
        a.out,
        a.corpus,
        rows,
        [
            f"S3 {a.bucket} prefixes {', '.join(a.prefix)}, keys ~ {a.include!r}; "
            f"licence {a.licence} per the licence file of each folder "
            f"({', '.join(sorted(v[0] for v in licence_files.values() if v))}); "
            f"per-file licence: {a.corpus}.licences.tsv",
            f"{len(rows)} files, {nbytes / 1e9:.2f} GB",
        ],
    )
    write_licences(a.out, a.corpus, lic_rows)
    print(f"{a.corpus}: {len(rows)} files, {nbytes / 1e9:.2f} GB -> {out}")


if __name__ == "__main__":
    main()
