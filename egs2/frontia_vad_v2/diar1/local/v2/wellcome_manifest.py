#!/usr/bin/env python3
"""Write the wellcome_av download manifest and licence table (Wellcome Collection).

    python3 local/v2/wellcome_manifest.py [--corpus wellcome_av] \
        [--out local/v2/downloads]

1. Lists the catalogue works of type Audio (i), Videos (g) and Music (c) whose
   digital location has the licence pdm, cc-by or cc-0
   (api.wellcomecollection.org/catalogue/v2/works); the licence counts of all
   works of these types are printed for the excluded tally.
2. Reads each work's IIIF Presentation v3 manifest. The manifest's rights URL
   must give the same licence as the catalogue location, otherwise the work is
   left out.
3. Leaves out media bodies that carry an IIIF auth service (a clickthrough or
   login: accepting terms is not done here). Per canvas, takes the audio body
   (Sound) when there is one, else the smallest video body: the alternatives of a
   Choice are compared by the Content-Length of a HEAD request (most responses
   give none, and then the first alternative, MP4, is taken).

Rows: wellcome_av, ARCHIVES/<b number>/<media id>.<ext>, the media URL, the size
(Content-Length when it was requested, else empty). The licence table gives
PDM-1.0, CC-BY-4.0 or CC0-1.0, the licence URL, the contributors and the work
page. Hours are the canvas durations. Requests are sequential, one per second,
with a User-Agent that names the project.
"""

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manifest_common import (  # noqa: E402
    DOWNLOADS,
    UA,
    Fetcher,
    licence_id,
    log,
    write_licences,
    write_manifest,
)

WORKS = "https://api.wellcomecollection.org/catalogue/v2/works"
TYPES = "i,g,c"
LICENCES = {"pdm": "PDM-1.0", "cc-by": "CC-BY-4.0", "cc-0": "CC0-1.0"}


def head_size(fetch: Fetcher, url: str) -> int:
    """Content-Length of a HEAD request (0 when absent or on error)."""
    wait = fetch.interval - (time.time() - fetch.last)
    if wait > 0:
        time.sleep(wait)
    fetch.last = time.time()
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return int(r.headers.get("Content-Length") or 0)
    except Exception as e:  # noqa: BLE001 - an unknown size is allowed
        log(f"HEAD {url}: {e}")
        return 0


def gated(body) -> bool:
    """True when the media body carries an IIIF auth service (clickthrough/login)."""
    return any("auth" in json.dumps(s).lower() for s in body.get("service") or [])


def bodies(manifest):
    """(canvas duration, [media bodies]) per canvas of a IIIF v3 manifest."""
    out = []
    for canvas in manifest.get("items") or []:
        media = []
        for page in canvas.get("items") or []:
            for anno in page.get("items") or []:
                b = anno.get("body") or {}
                if b.get("type") == "Choice":
                    media += b.get("items") or []
                else:
                    media.append(b)
        out.append((canvas.get("duration") or 0, [m for m in media if m.get("id")]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default="wellcome_av")
    ap.add_argument("--out", type=Path, default=DOWNLOADS)
    a = ap.parse_args()
    fetch = Fetcher()

    agg = fetch.json(
        f"{WORKS}?pageSize=1&workType={TYPES}&aggregations=items.locations.license"
    )
    all_lic = {
        b["data"]["id"]: b["count"]
        for b in agg["aggregations"]["items.locations.license"]["buckets"]
    }
    print(f"{a.corpus}: works of types {TYPES} by location licence: {all_lic}")

    q = urllib.parse.urlencode(
        {
            "pageSize": 100,
            "workType": TYPES,
            "items.locations.license": ",".join(LICENCES),
            "include": "items,contributors",
        }
    )
    works, url = [], f"{WORKS}?{q}"
    while url:
        d = fetch.json(url)
        works += d["results"]
        url = d.get("nextPage")
    print(f"{a.corpus}: {len(works)} works with licence {','.join(LICENCES)}")

    excluded, kinds, rows, lic_rows, secs, nbytes = Counter(), Counter(), [], [], 0.0, 0
    for w in works:
        locs = [
            loc
            for it in w.get("items") or []
            for loc in it.get("locations") or []
            if loc.get("locationType", {}).get("id") == "iiif-presentation"
        ]
        if not locs or (locs[0].get("license") or {}).get("id") not in LICENCES:
            excluded["no IIIF location with an accepted licence"] += 1
            continue
        loc = locs[0]
        lid = LICENCES[loc["license"]["id"]]
        man = fetch.json(loc["url"].replace("/presentation/v2/", "/presentation/v3/"))
        if licence_id(man.get("rights", "")) != lid:
            excluded[f"IIIF rights {man.get('rights')!r} differ from {lid}"] += 1
            continue
        bnum = man["id"].rstrip("/").rsplit("/", 1)[-1]
        creators = "; ".join(c["agent"]["label"] for c in w.get("contributors") or [])
        n_before, auth = len(rows), 0
        for dur, media in bodies(man):
            if any(gated(m) for m in media):
                auth += 1
                continue
            sound = [m for m in media if m.get("type") == "Sound"]
            video = [m for m in media if m.get("type") == "Video"]
            if sound:
                pick, size = sound[0], ""
            elif video:
                sized = [(head_size(fetch, m["id"]), m) for m in video]
                known = [s for s in sized if s[0]] or sized
                size_i, pick = min(known, key=lambda s: s[0])
                size = str(size_i) if size_i else ""
            else:
                continue
            path = urllib.parse.urlparse(pick["id"]).path
            media_id = re.sub(
                r"\.[a-z0-9]+$", "", path.split("/av/", 1)[-1].split("/")[0]
            )
            rel = f"ARCHIVES/{bnum}/{media_id}{Path(path).suffix}"
            rows.append((rel, pick["id"], size, ""))
            lic_rows.append(
                (
                    rel,
                    lid,
                    loc["license"]["url"],
                    creators,
                    f"https://wellcomecollection.org/works/{w['id']}",
                )
            )
            secs += float(dur or pick.get("duration") or 0)
            nbytes += int(size or 0)
            kinds[f"{pick.get('type')} {pick.get('format')}"] += 1
        if auth:
            excluded["media behind an IIIF auth (clickthrough) service"] += 1
        elif len(rows) == n_before:
            excluded["no audio or video body"] += 1
    rows.sort()
    lic_rows.sort()
    out = write_manifest(
        a.out,
        a.corpus,
        rows,
        [
            f"Wellcome Collection works of type Audio/Videos/Music with licence "
            f"pdm, cc-by or cc-0; per-file licence: {a.corpus}.licences.tsv",
            f"{len(rows)} files, {secs / 3600:.1f} h",
        ],
    )
    write_licences(a.out, a.corpus, lic_rows)
    print(
        f"{a.corpus}: {len(rows)} files, {nbytes / 1e9:.2f} GB (videos sized), "
        f"{secs / 3600:.1f} h; {dict(kinds)}; by licence "
        f"{dict(Counter(r[1] for r in lic_rows))}; excluded works {dict(excluded)}"
        f" -> {out}"
    )


if __name__ == "__main__":
    main()
