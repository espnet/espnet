#!/usr/bin/env python3
"""Write the ccc_talks download manifest and licence table from media.ccc.de.

    python3 local/v2/ccc_manifest.py [--corpus ccc_talks] [--max-talks 3000] \
        [--cache DIR] [--out local/v2/downloads]

1. Reads https://api.media.ccc.de/public/conferences and each conference's
   event list (one request per conference).
2. Keeps the talks whose own description states the licence line "Licensed to the
   public under <url>" with exactly one URL whose id is CC-BY-4.0. Talks with a
   BY-SA (or any other) licence and talks with no statement are counted and left
   out.
3. When more than --max-talks talks are kept, takes a subset of conferences: the
   conferences sorted by release date, every k-th one, k the smallest step that
   brings the kept talks under --max-talks. The totals of all kept talks (count,
   hours from the listed durations) are printed as the projection.
4. Reads each kept talk's /public/events/<guid> (one request per talk) and picks
   one recording: an audio recording (audio/mpeg first, then audio/opus, then any
   audio) in the talk's original language, else the smallest video recording in
   that language, else the smallest recording. The API lists recording sizes in
   MB, so the manifest's size column stays empty (download.py takes the served
   Content-Length).

Rows: ccc_talks, ARCHIVES/<conference acronym>/<recording file name>, recording
URL. The licence table gives CC-BY-4.0, the licence URL, the speakers and the
talk page per file. Requests are sequential, one per second, with a User-Agent
that names the project. --cache keeps the API responses as JSON files, so a rerun
reads them instead of the API.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manifest_common import (  # noqa: E402
    DOWNLOADS,
    Fetcher,
    licence_id,
    log,
    write_licences,
    write_manifest,
)

API = "https://api.media.ccc.de/public"
LINE = re.compile(r"Licensed to the public under\s+(\S+)", re.I)
WANT = "CC-BY-4.0"


class CachedApi:
    """JSON GET of the API, kept as files in `cache` when it is given."""

    def __init__(self, cache):
        self.fetch, self.cache = Fetcher(), cache
        if cache:
            cache.mkdir(parents=True, exist_ok=True)

    def get(self, url):
        if self.cache:
            f = self.cache / (hashlib.sha1(url.encode()).hexdigest() + ".json")
            if f.exists():
                return json.loads(f.read_text())
        d = self.fetch.json(url)
        if self.cache:
            f.write_text(json.dumps(d))
        return d


def untag(description) -> str:
    """description without HTML tags (a licence URL may sit in an <a> element)."""
    return re.sub(r"<[^>]+>", " ", description or "")


def talk_licence(description: str) -> str:
    """The licence id of the talk's "Licensed to the public under" line(s)."""
    urls = LINE.findall(untag(description))
    if not urls:
        return "no statement"
    ids = {licence_id(u.rstrip(".,)")) or u for u in urls}
    return ids.pop() if len(ids) == 1 else "several: " + ",".join(sorted(ids))


def pick_recording(recs, lang):
    """The recording to download, in lang when there is one.

    Audio first (mp3, opus, other audio), else the smallest video; subtitles and
    other non-media recordings are never chosen.
    """
    media = [
        r
        for r in recs
        if r.get("recording_url")
        and r.get("mime_type", "").startswith(("audio/", "video/"))
    ]
    same = [r for r in media if r.get("language") == lang] or media

    def rank(r):
        mime = r.get("mime_type", "")
        audio = {"audio/mpeg": 0, "audio/opus": 1}.get(
            mime, 2 if "audio" in mime else 3
        )
        return (audio, r.get("size") or 0)

    return min(same, key=rank) if same else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default="ccc_talks")
    ap.add_argument("--max-talks", type=int, default=3000)
    ap.add_argument("--cache", type=Path, default=None, help="keep API responses here")
    ap.add_argument("--out", type=Path, default=DOWNLOADS)
    a = ap.parse_args()
    api = CachedApi(a.cache)

    confs = api.get(f"{API}/conferences")["conferences"]
    log(f"{len(confs)} conferences")
    excluded, kept = Counter(), {}  # kept: acronym -> [event]
    for i, c in enumerate(confs):
        evs = api.get(c["url"]).get("events", [])
        for e in evs:
            lic = talk_licence(e.get("description"))
            if lic == WANT:
                kept.setdefault(c["acronym"], []).append(e)
            else:
                excluded[lic] += 1
        if i % 50 == 0:
            n = sum(map(len, kept.values()))
            log(f"{i}/{len(confs)} conferences read, {n} kept")
    n_all = sum(map(len, kept.values()))
    h_all = sum(e.get("duration") or 0 for v in kept.values() for e in v) / 3600
    print(
        f"{a.corpus}: {n_all} CC BY 4.0 talks in {len(kept)} conferences, {h_all:.0f} h"
    )
    print(f"{a.corpus}: excluded talks by licence: {dict(excluded.most_common())}")

    released = {c["acronym"]: c.get("event_last_released_at") or "" for c in confs}
    order = sorted(kept, key=lambda k: (released[k], k))
    step = 1
    while sum(len(kept[k]) for k in order[::step]) > a.max_talks:
        step += 1
    chosen = order[::step]
    if step > 1:
        n = sum(len(kept[k]) for k in chosen)
        print(
            f"{a.corpus}: subset: every {step}th conference by release date, "
            f"{len(chosen)} of {len(order)} conferences, {n} of {n_all} talks"
        )

    rows, lic_rows, sizes_mb, secs, kinds = [], [], 0, 0, Counter()
    for k in chosen:
        for e in kept[k]:
            ev = api.get(e["url"])
            rec = pick_recording(
                ev.get("recordings") or [], ev.get("original_language")
            )
            if rec is None:
                excluded["no audio or video recording"] += 1
                continue
            rel = f"ARCHIVES/{k}/{rec['filename']}"
            rows.append((rel, rec["recording_url"], "", ""))
            lic_url = LINE.search(untag(ev.get("description"))).group(1)
            lic_rows.append(
                (
                    rel,
                    WANT,
                    lic_url,
                    "; ".join(ev.get("persons") or []),
                    ev["frontend_link"],
                )
            )
            sizes_mb += rec.get("size") or 0
            secs += ev.get("duration") or 0
            kinds[rec.get("mime_type", "")] += 1
    rows.sort()
    lic_rows.sort()
    note = f"subset every {step}th conference" if step > 1 else "all conferences"
    out = write_manifest(
        a.out,
        a.corpus,
        rows,
        [
            f"media.ccc.de talks whose description states 'Licensed to the public "
            f"under' CC BY 4.0 ({note}); per-file licence: {a.corpus}.licences.tsv",
            f"{len(rows)} recordings, {secs / 3600:.1f} h",
        ],
    )
    write_licences(a.out, a.corpus, lic_rows)
    print(
        f"{a.corpus}: {len(rows)} recordings, ~{sizes_mb / 1e3:.1f} GB (API MB), "
        f"{secs / 3600:.1f} h, formats {dict(kinds)}; excluded {dict(excluded)}"
        f" -> {out}"
    )


if __name__ == "__main__":
    main()
