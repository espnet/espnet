#!/usr/bin/env python3
"""Write a download manifest and licence table from Internet Archive items.

    python3 local/v2/ia_manifest.py <corpus> <advancedsearch query> \
        [--licences CC-BY-*,CC0-1.0,PDM-1.0] [--exclude-title REGEX] \
        [--cap-hours N] [--site-regex REGEX] [--max-items N] [--cache DIR] \
        [--out local/v2/downloads]

1. Lists the items matching the query with the scrape API
   (archive.org/services/search/v1/scrape, cursor paging): identifier, title,
   date, creator, licenseurl.
2. Converts each item's licenseurl to an SPDX-like id (manifest_common.licence_id;
   the Public Domain Mark is PDM-1.0) and keeps the item when the id matches one
   of the --licences patterns (fnmatch) and has no NC, SA or ND. Items whose title
   matches --exclude-title are left out.
3. Orders the kept items: with --site-regex (its first group, applied to the
   title, names the site) round-robin over the sites, each site's items in a
   spread order over their dates (bit-reversed index of the date-sorted list);
   without it, by the sha1 of the identifier. Reads /metadata/<id> in that order
   until --cap-hours of audio or --max-items items are reached. With
   --spread-files, reads every item and applies the same site round-robin and
   date spread to the files (date of the item, then file name) before the cap.
4. Takes the item's original audio files (source == original, an audio format or
   audio file extension). Rows: corpus, ARCHIVES/<id>/<file>, the archive.org
   download URL, the size and md5 that the item lists. The licence table gives the
   item's licence id, licenseurl, creator and details page per file.

Requests are sequential, one per second, with a User-Agent that names the project.
--cache keeps the API responses as JSON files, so a rerun reads them instead.
"""

import argparse
import fnmatch
import hashlib
import json
import re
import sys
import urllib.parse
from collections import Counter, defaultdict
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

SCRAPE = "https://archive.org/services/search/v1/scrape"
AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".aif", ".aiff", ".m4a")
AUDIO_FMT = re.compile(r"wave|flac|mp3|ogg|opus|aiff|apple lossless|audio", re.I)


class CachedApi:
    """JSON GET, kept as files in `cache` when it is given."""

    def __init__(self, cache):
        self.fetch, self.cache = Fetcher(), cache
        if cache:
            cache.mkdir(parents=True, exist_ok=True)

    def get(self, url):
        f = None
        if self.cache:
            f = self.cache / (hashlib.sha1(url.encode()).hexdigest() + ".json")
            if f.exists():
                return json.loads(f.read_text())
        d = self.fetch.json(url)
        if f:
            f.write_text(json.dumps(d))
        return d


def search(api, query):
    """All items of the scrape API for query."""
    items, cursor = [], None
    while True:
        p = {
            "q": query,
            "fields": "identifier,title,date,creator,licenseurl",
            "count": 10000,
        }
        if cursor:
            p["cursor"] = cursor
        d = api.get(f"{SCRAPE}?{urllib.parse.urlencode(p)}")
        items += d.get("items", [])
        cursor = d.get("cursor")
        log(f"search: {len(items)} of {d.get('total')}")
        if not cursor:
            return items


def text(v) -> str:
    return "; ".join(map(str, v)) if isinstance(v, list) else str(v or "")


def seconds(length) -> float:
    """Seconds of an IA file length ("447.78" or "07:27"); 0 when unknown."""
    try:
        s = 0.0
        for part in str(length or "").split(":"):
            s = s * 60 + float(part)
        return s
    except ValueError:
        return 0.0


def bit_reversed(n: int):
    """Indices 0..n-1 in bit-reversed order (a spread over the range)."""
    bits = max(1, (n - 1).bit_length())
    order = sorted(range(1 << bits), key=lambda i: int(f"{i:0{bits}b}"[::-1], 2))
    return [i for i in order if i < n]


def spread(records, site_of, key_of):
    """Records round-robin over their sites, each site spread over its key order."""
    sites = defaultdict(list)
    for r in records:
        sites[site_of(r)].append(r)
    queues = []
    for s in sorted(sites):
        v = sorted(sites[s], key=key_of)
        queues.append([v[i] for i in bit_reversed(len(v))])
    out = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def site_of_title(site_regex):
    def site(it):
        m = re.search(site_regex, text(it.get("title")))
        return m.group(1) if m else ""

    return site


def item_files(api, it, lic, excluded):
    """Records of the original audio files of item it ([] when it has none)."""
    ident = it["identifier"]
    try:
        md = api.get(f"https://archive.org/metadata/{urllib.parse.quote(ident)}")
    except RuntimeError as e:
        log(f"{ident}: {e}")
        excluded["metadata unavailable"] += 1
        return []
    lid, url = lic
    meta = md.get("metadata") or {}
    if licence_id(text(meta.get("licenseurl"))) != lid:
        excluded["item metadata licence differs from search"] += 1
        return []
    orig = [
        f
        for f in md.get("files") or []
        if f.get("source") == "original"
        and (
            AUDIO_FMT.search(f.get("format", ""))
            or f.get("name", "").lower().endswith(AUDIO_EXT)
        )
        and not f.get("name", "").lower().endswith((".png", ".xml", ".jpg"))
    ]
    if not orig:
        excluded["no original audio file"] += 1
    return [
        {
            "item": it,
            "name": f["name"],
            "rel": f"ARCHIVES/{ident}/{f['name']}",
            "url": f"https://archive.org/download/{urllib.parse.quote(ident)}/"
            f"{urllib.parse.quote(f['name'])}",
            "size": f.get("size", ""),
            "check": f"md5:{f['md5']}" if f.get("md5") else "",
            "lic": (lid, url, text(meta.get("creator"))),
            "secs": seconds(f.get("length")),
        }
        for f in orig
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("corpus")
    ap.add_argument("query", help="advancedsearch query, e.g. collection:xyz")
    ap.add_argument("--licences", default="CC-BY-*,CC0-1.0,PDM-1.0")
    ap.add_argument("--exclude-title", default="")
    ap.add_argument("--cap-hours", type=float, default=0, help="0: no cap")
    ap.add_argument("--site-regex", default="", help="group 1 of it names the site")
    ap.add_argument(
        "--spread-files",
        action="store_true",
        help="read every item, then spread the cap over files instead of items",
    )
    ap.add_argument("--max-items", type=int, default=0, help="0: all items")
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=DOWNLOADS)
    a = ap.parse_args()
    api = CachedApi(a.cache)
    pats = a.licences.split(",")

    items = search(api, a.query)
    excluded, kept, lic_of = Counter(), [], {}
    for it in items:
        url = text(it.get("licenseurl"))
        lid = licence_id(url) or ("none" if not url else url)
        ok = licence_accepted(lid) and any(fnmatch.fnmatch(lid, p) for p in pats)
        if not ok:
            excluded[f"licence {lid}"] += 1
        elif a.exclude_title and re.search(
            a.exclude_title, text(it.get("title")), re.I
        ):
            excluded["title looks speech-heavy"] += 1
        else:
            kept.append(it)
            lic_of[it["identifier"]] = (lid, url)
    print(f"{a.corpus}: {len(items)} items, {len(kept)} kept by licence and title")
    kept_lic = Counter(lid for lid, _ in lic_of.values())
    print(f"{a.corpus}: kept items by licence {dict(kept_lic)}")
    if a.site_regex:
        site = site_of_title(a.site_regex)
        order = spread(kept, site, lambda it: (text(it.get("date")), it["identifier"]))
    else:
        order = sorted(
            kept, key=lambda it: hashlib.sha1(it["identifier"].encode()).hexdigest()
        )
    if a.max_items:
        order = order[: a.max_items]

    files, secs, n_items = [], 0.0, 0
    for i, it in enumerate(order):
        if a.cap_hours and not a.spread_files and secs >= a.cap_hours * 3600:
            break
        got = item_files(api, it, lic_of[it["identifier"]], excluded)
        n_items += bool(got)
        files += got
        secs += sum(f["secs"] for f in got)
        if i % 100 == 0:
            log(f"{i}/{len(order)} items read, {len(files)} files, {secs / 3600:.0f} h")
    if a.spread_files:
        files = spread(
            files,
            lambda f: site(f["item"]) if a.site_regex else "",
            lambda f: (text(f["item"].get("date")), f["name"]),
        )
    chosen, secs = [], 0.0
    for f in files:
        if a.cap_hours and secs >= a.cap_hours * 3600:
            break
        chosen.append(f)
        secs += f["secs"]
    if not chosen:
        print(f"{a.corpus}: no file left", file=sys.stderr)
        sys.exit(1)
    chosen.sort(key=lambda f: f["rel"])
    rows = [(f["rel"], f["url"], f["size"], f["check"]) for f in chosen]
    lic_rows = [
        (f["rel"], *f["lic"], f"https://archive.org/details/{f['item']['identifier']}")
        for f in chosen
    ]
    nbytes = sum(int(f["size"] or 0) for f in chosen)
    files_lic = Counter(f["lic"][0] for f in chosen)
    n_chosen = len({f["item"]["identifier"] for f in chosen})
    limit = []
    if a.cap_hours:
        unit = "files" if a.spread_files else "items"
        limit.append(f"cap {a.cap_hours:g} h, {unit} in spread order over sites/dates")
    if a.max_items:
        limit.append(f"first {a.max_items} items of the spread order")
    out = write_manifest(
        a.out,
        a.corpus,
        rows,
        [
            f"Internet Archive query {a.query!r}, item licenseurl in {a.licences}"
            + (f", titles not matching {a.exclude_title!r}" if a.exclude_title else "")
            + (f" ({'; '.join(limit)})" if limit else "")
            + f"; per-file licence: {a.corpus}.licences.tsv",
            f"{n_chosen} items, {len(rows)} original audio files, {secs / 3600:.1f} h",
        ],
    )
    write_licences(a.out, a.corpus, lic_rows)
    print(
        f"{a.corpus}: {n_items} items with audio read, {n_chosen} used, "
        f"{len(rows)} files, {nbytes / 1e9:.1f} GB, {secs / 3600:.1f} h; "
        f"files by licence {dict(files_lic)}; "
        f"excluded items {dict(excluded.most_common())} -> {out}"
    )


if __name__ == "__main__":
    main()
