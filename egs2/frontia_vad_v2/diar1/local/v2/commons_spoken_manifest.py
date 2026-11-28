#!/usr/bin/env python3
"""Write the spoken_wikisource download manifest and licence table (Wikimedia Commons).

    python3 local/v2/commons_spoken_manifest.py [--corpus spoken_wikisource] \
        [--category "Category:Spoken Wikisource - German" ...] [--max-depth 3] \
        [--cache DIR] [--out local/v2/downloads]

1. Collects the files of the categories (default: Spoken Wikisource - German and
   - Portuguese) and of their subcategories down to --max-depth levels, through
   the MediaWiki API (list=categorymembers). Unlike commons_manifest.py, no
   subcategory or file is dropped for being speech: these are read texts.
2. Drops files without an audio extension (commons_manifest.AUDIO_EXT) and reads
   the imageinfo (url, size, mime, extmetadata, metadata) of the rest, 50 titles
   per request.
3. Keeps a file when its extmetadata License is cc0, pd-* or cc-by-<version>
   (commons_manifest.licence_of) and its mime is audio/* or application/ogg.

Rows: spoken_wikisource, ARCHIVES/<file name>, the original file URL, the size;
no checksum. The licence table gives the licence, LicenseUrl (or the PD template
page), Artist and the file page. Hours come from the file metadata.

Requests are sequential POSTs with maxlag=5, at least one second apart, with the
User-Agent of manifest_common.py.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from commons_manifest import AUDIO_EXT, Api, duration, licence_of  # noqa: E402
from manifest_common import (  # noqa: E402
    DOWNLOADS,
    Fetcher,
    clean,
    log,
    write_licences,
    write_manifest,
)

CATEGORIES = [
    "Category:Spoken Wikisource - German",
    "Category:Spoken Wikisource - Portuguese",
]


def members(api, cat, depth, seen):
    """File titles of cat and, while depth > 0, of all its subcategories."""
    files, cont = [], {}
    seen.add(cat)
    while True:
        d = api.get(
            action="query",
            list="categorymembers",
            cmtitle=cat,
            cmlimit=500,
            cmtype="file|subcat",
            **cont,
        )
        for m in d["query"]["categorymembers"]:
            if m["ns"] == 6:
                files.append(m["title"])
            elif m["ns"] == 14 and depth > 0 and m["title"] not in seen:
                files += members(api, m["title"], depth - 1, seen)
        if "continue" not in d:
            return files
        cont = d["continue"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default="spoken_wikisource")
    ap.add_argument("--category", action="append", default=None)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--cache", type=Path, default=None, help="keep API replies here")
    ap.add_argument("--out", type=Path, default=DOWNLOADS)
    a = ap.parse_args()
    cats = a.category or CATEGORIES
    api = Api(a.cache)
    api.fetch = Fetcher()  # the User-Agent of manifest_common

    titles, seen = {}, set()
    for cat in cats:
        got = members(api, cat, a.max_depth, seen)
        for t in got:
            titles.setdefault(t, cat)
        log(f"{cat}: {len(got)} files, {len(titles)} unique so far")

    excluded, order = Counter(), []
    for t in sorted(titles):
        ext = t.rsplit(".", 1)[-1].lower()
        if ext in AUDIO_EXT:
            order.append(t)
        else:
            excluded[f"not audio (.{ext})"] += 1
    kept, rows, lic_rows, secs, nbytes = Counter(), [], [], 0.0, 0
    for i in range(0, len(order), 50):
        d = api.get(
            action="query",
            prop="imageinfo",
            titles="|".join(order[i : i + 50]),
            iiprop="url|size|mime|extmetadata|metadata",
            iiextmetadatafilter="License|LicenseUrl|LicenseShortName|Artist",
        )
        for p in d["query"]["pages"]:
            ii = (p.get("imageinfo") or [{}])[0]
            em = ii.get("extmetadata") or {}
            mime = ii.get("mime", "")
            lic = licence_of(em)
            if lic is None:
                tid = (em.get("License") or {}).get("value", "") or "none"
                excluded[f"licence {tid}"] += 1
            elif not (mime.startswith("audio/") or mime == "application/ogg"):
                excluded[f"not audio ({mime or 'no mime'})"] += 1
            else:
                rel = "ARCHIVES/" + p["title"].split(":", 1)[1].replace(" ", "_")
                rows.append((rel, ii["url"], ii.get("size", ""), ""))
                artist = clean((em.get("Artist") or {}).get("value", ""))
                lic_rows.append(
                    (rel, lic[0], lic[1], artist, ii.get("descriptionurl", ""))
                )
                kept[f"{titles[p['title']].split(' - ')[-1]} {lic[0]}"] += 1
                secs += duration(ii)
                nbytes += int(ii.get("size") or 0)
    rows.sort()
    lic_rows.sort()
    out = write_manifest(
        a.out,
        a.corpus,
        rows,
        [
            "Wikimedia Commons, " + "; ".join(cats) + f" (subcategories to depth "
            f"{a.max_depth}); licence cc0/pd/cc-by only; per-file licence: "
            f"{a.corpus}.licences.tsv",
            f"{len(rows)} files, {secs / 3600:.1f} h",
        ],
    )
    write_licences(a.out, a.corpus, lic_rows)
    print(
        f"{a.corpus}: {len(rows)} files, {nbytes / 1e9:.2f} GB, {secs / 3600:.1f} h; "
        f"kept {dict(kept)}; excluded {dict(excluded.most_common())} -> {out}"
    )


if __name__ == "__main__":
    main()
