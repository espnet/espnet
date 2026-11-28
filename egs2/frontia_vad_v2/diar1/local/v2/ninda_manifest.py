#!/usr/bin/env python3
"""Write the ninda_dialects download manifest and licence table (NINJAL NINDA).

    python3 local/v2/ninda_manifest.py [--corpus ninda_dialects] \
        [--out local/v2/downloads]

1. Lists the items of the NINDA Omeka S API (ninda.ninjal.ac.jp/api/items) whose
   dcterms:format contains "mp3": the recordings of the 方言談話資料 and
   方言録音資料 series (one item per recording side).
2. Reads each item's licence from dcterms:license, or dcterms:rights when the
   item has no dcterms:license, and keeps the item when it is CC BY or CC0
   (manifest_common.licence_accepted).
3. Reads the item's media (api/media/<id>) and keeps the audio/mpeg ones.

Rows: ninda_dialects, ARCHIVES/<item id>_<media source file name>, the media's
original_url, its size and sha256 as Omeka lists them. The licence table gives
the licence id, the licence URL, dcterms:creator and the item page (on the item's first
Omeka site).

The word-level vocabulary clips of NINDA (about 52k short audio/mpeg media of
dictionary items) are not taken: their items have no mp3 dcterms:format.
Requests are sequential, one per second.
"""

import argparse
import sys
import urllib.parse
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manifest_common import (  # noqa: E402
    DOWNLOADS,
    Fetcher,
    licence_accepted,
    licence_id,
    log,
    write_licences,
    write_manifest,
)

API = "https://ninda.ninjal.ac.jp/api"
LICENCE_URL = {
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
}


def values(item, key):
    """The literal values of the property key of an Omeka item."""
    return [v.get("@value") or v.get("@id") or "" for v in item.get(key, [])]


def item_page(item, slugs):
    """The item's page on its first Omeka site, else its API URL."""
    sites = [slugs.get(s["o:id"]) for s in item.get("o:site", [])]
    if sites and sites[0]:
        return f"https://ninda.ninjal.ac.jp/s/{sites[0]}/item/{item['o:id']}"
    return f"{API}/items/{item['o:id']}"


def mp3_items(fetch):
    """All items whose dcterms:format (property 9) contains mp3."""
    items, page = [], 1
    while True:
        q = urllib.parse.urlencode(
            {
                "property[0][property]": 9,
                "property[0][type]": "in",
                "property[0][text]": "mp3",
                "per_page": 100,
                "page": page,
                "sort_by": "id",
            }
        )
        got = fetch.json(f"{API}/items?{q}")
        items += got
        if len(got) < 100:
            return items
        page += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default="ninda_dialects")
    ap.add_argument("--out", type=Path, default=DOWNLOADS)
    a = ap.parse_args()
    fetch = Fetcher()

    slugs = {s["o:id"]: s["o:slug"] for s in fetch.json(f"{API}/sites")}
    items = mp3_items(fetch)
    log(f"{a.corpus}: {len(items)} items with an mp3 format")
    excluded, kept, rows, lic_rows, nbytes = Counter(), Counter(), [], [], 0
    for it in items:
        stated = values(it, "dcterms:license") or values(it, "dcterms:rights")
        lid = licence_id(" ".join(stated))
        if not licence_accepted(lid):
            excluded[f"licence {stated or 'none'}"] += 1
            continue
        audio = 0
        for m in it.get("o:media", []):
            media = fetch.json(f"{API}/media/{m['o:id']}")
            if media.get("o:media_type") != "audio/mpeg":
                continue
            src = urllib.parse.unquote(media.get("o:source") or "").rsplit("/", 1)[-1]
            name = f"{it['o:id']}_{src or media['o:id']}"
            sha = media.get("o:sha256") or ""
            size = media.get("o:size") or ""
            rows.append(
                (f"ARCHIVES/{name}", media["o:original_url"], size, f"sha256:{sha}")
            )
            lic_rows.append(
                (
                    f"ARCHIVES/{name}",
                    lid,
                    LICENCE_URL.get(lid, ""),
                    "; ".join(values(it, "dcterms:creator")) or "国立国語研究所",
                    item_page(it, slugs),
                )
            )
            nbytes += int(size or 0)
            audio += 1
        if audio:
            kept[lid] += 1
        else:
            excluded["no audio/mpeg media"] += 1
    rows.sort()
    lic_rows.sort()
    out = write_manifest(
        a.out,
        a.corpus,
        rows,
        [
            "NINJAL NINDA (ninda.ninjal.ac.jp) items with an mp3 dcterms:format "
            "(方言談話資料 and 方言録音資料 series), dcterms:license or "
            "dcterms:rights CC BY / CC0 only; per-file licence: "
            f"{a.corpus}.licences.tsv",
            f"{len(rows)} files from {sum(kept.values())} items",
        ],
    )
    write_licences(a.out, a.corpus, lic_rows)
    print(
        f"{a.corpus}: {len(rows)} files, {nbytes / 1e9:.2f} GB; kept items by "
        f"licence {dict(kept)}; excluded {dict(excluded)} -> {out}"
    )


if __name__ == "__main__":
    main()
