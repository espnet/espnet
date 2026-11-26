#!/usr/bin/env python3
"""Write the commons_nonspeech download manifest and licence table (Wikimedia Commons).

    python3 local/v2/commons_manifest.py [--corpus commons_nonspeech] \
        [--max-depth 5] [--cache DIR] [--out local/v2/downloads]

1. Collects the files of the whitelisted categories (CATEGORIES) through the
   MediaWiki API (list=categorymembers). Categories marked deep are walked into
   their subcategories up to --max-depth levels; a subcategory whose title
   matches SPEECH_CAT (spoken word, pronunciation, Lingua Libre, voice, choir,
   people, ...) is not entered.
2. Drops files without an audio extension (AUDIO_EXT; scans and PDFs) and MIDI.
   Reads each remaining file's imageinfo (url, size, mime, extmetadata,
   metadata) and categories, 50 titles per request.
3. Keeps a file when its extmetadata License (the licence template id) is cc0,
   pd-* (or pd) or cc-by-<version>[-<port>]; -sa, -nc, -nd and every other licence
   are dropped. A kept file is still dropped when it is not audio (mime audio/* or
   application/ogg), when it is MIDI, when its title or categories match SPEECH
   (pronunciation, LL-Q, spoken, interview, voice, choir, hymn, anthem, cantata,
   speech, lecture, sermon, narration, recitation, poem), or when its description
   or source mentions YouTube.

Rows: commons_nonspeech, ARCHIVES/<file name>, the original file URL (imageinfo
url), the size; no checksum (Commons lists sha1 only). The licence table gives the
licence (CC0-1.0, PD, CC-BY-x.y[-port]), LicenseUrl (or the PD template page),
Artist and the file page. Hours come from the file metadata (length or
playtime_seconds) where present.

Requests are sequential POSTs (50 long titles exceed the URL limit), at least one
second apart, with maxlag=5; Retry-After is honoured on HTTP errors, and maxlag
replies are retried after 5 s. --cache keeps the API replies as JSON files, so a
rerun reads them instead.
"""

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manifest_common import (  # noqa: E402
    DOWNLOADS,
    Fetcher,
    clean,
    log,
    write_licences,
    write_manifest,
)

API = "https://commons.wikimedia.org/w/api.php"
UA = "ESPnet-FRONTIA-VAD-dataprep/0.1 (https://github.com/espnet/espnet)"
CATEGORIES = [
    ("Category:Audio files of bells", True),
    ("Category:Organ music", True),
    ("Category:Audio files of sound effects", True),
    ("Category:Soundscape", True),
    ("Category:Audio files of animals", True),
    ("Category:Sounds of rain", False),
    ("Category:Audio files of thunder", False),
    ("Category:Sounds of trains", False),
]
SPEECH_CAT = re.compile(
    r"spoken|pronunciation|lingua libre|\bLL-|interview|voice|speech|choir|choral|"
    r"hymn|anthem|cantata|lecture|podcast|poetry|\bpeople\b|\bhumans?\b",
    re.I,
)
SPEECH = re.compile(
    r"pronunciation|\bLL-Q|spoken|interview|voice|choir|hymn|anthem|cantata|"
    r"speech|lecture|sermon|narrat|recitation|\bpoem",
    re.I,
)
AUDIO_EXT = ("ogg", "oga", "opus", "wav", "flac", "mp3", "webm", "m4a", "aac")
KEEP_LICENCE = re.compile(r"cc0|cc-zero|pd(-.*)?|cc-by-\d\.\d(-[a-z]{2,})?", re.I)


class Api:
    """POST to the MediaWiki API with maxlag=5, waiting on maxlag replies.

    With `cache`, every reply is kept as a JSON file named by the sha1 of the
    request, and a rerun reads it instead of the API.
    """

    def __init__(self, cache=None):
        self.fetch, self.cache = Fetcher(ua=UA, interval=1.0), cache
        if cache:
            cache.mkdir(parents=True, exist_ok=True)

    def get(self, **params):
        params.update(format="json", formatversion=2, maxlag=5)
        body = urllib.parse.urlencode(params).encode()
        f = None
        if self.cache:
            f = self.cache / (hashlib.sha1(body).hexdigest() + ".json")
            if f.exists():
                return json.loads(f.read_text())
        for _ in range(10):
            d = self.fetch.json(API, body)
            if (d.get("error") or {}).get("code") != "maxlag":
                if f:
                    f.write_text(json.dumps(d))
                return d
            log("maxlag; waiting 5 s")
            time.sleep(5)
        raise RuntimeError(f"{API} {params.get('action')}: maxlag persists")


def members(api, cat, depth, seen_cats, skipped_cats):
    """File titles of cat and, while depth > 0, of its non-speech subcategories."""
    files, cont = [], {}
    seen_cats.add(cat)
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
            elif m["ns"] == 14 and depth > 0 and m["title"] not in seen_cats:
                if SPEECH_CAT.search(m["title"]):
                    skipped_cats.add(m["title"])
                    continue
                files += members(api, m["title"], depth - 1, seen_cats, skipped_cats)
        if "continue" not in d:
            return files
        cont = d["continue"]


def licence_of(em):
    """(licence id, licence url) of extmetadata, or None when it is not kept."""
    tid = (em.get("License") or {}).get("value", "").strip().lower()
    url = (em.get("LicenseUrl") or {}).get("value", "")
    if not KEEP_LICENCE.fullmatch(tid):
        return None
    if tid in ("cc0", "cc-zero"):
        return "CC0-1.0", url or "https://creativecommons.org/publicdomain/zero/1.0/"
    if tid.startswith("pd"):
        return "PD", url or f"https://commons.wikimedia.org/wiki/Template:{tid}"
    m = re.fullmatch(r"cc-by-(\d\.\d)(?:-([a-z]{2,}))?", tid)
    port = f"-{m.group(2).upper()}" if m.group(2) else ""
    return f"CC-BY-{m.group(1)}{port}", url


def duration(ii) -> float:
    """Seconds from the imageinfo metadata (length or playtime_seconds), else 0."""
    for m in ii.get("metadata") or []:
        if m.get("name") in ("length", "playtime_seconds"):
            try:
                return float(m["value"])
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default="commons_nonspeech")
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--cache", type=Path, default=None, help="keep API replies here")
    ap.add_argument("--out", type=Path, default=DOWNLOADS)
    a = ap.parse_args()
    api = Api(a.cache)

    titles, seen_cats, skipped_cats = {}, set(), set()
    for cat, deep in CATEGORIES:
        got = members(api, cat, a.max_depth if deep else 0, seen_cats, skipped_cats)
        for t in got:
            titles.setdefault(t, cat)
        log(f"{cat}: {len(got)} files, {len(titles)} unique so far")
    print(
        f"{a.corpus}: {len(titles)} files in {len(seen_cats)} categories; "
        f"{len(skipped_cats)} speech-like subcategories not entered"
    )

    excluded, order = Counter(), []
    for t in sorted(titles):
        ext = t.rsplit(".", 1)[-1].lower()
        if ext in ("mid", "midi"):
            excluded["MIDI"] += 1
        elif ext not in AUDIO_EXT:
            excluded[f"not audio (.{ext})"] += 1
        else:
            order.append(t)
    kept_lic, rows, lic_rows, secs, nbytes = Counter(), [], [], 0.0, 0
    for i in range(0, len(order), 50):
        d = api.get(
            action="query",
            prop="imageinfo|categories",
            titles="|".join(order[i : i + 50]),
            iiprop="url|size|mime|extmetadata|metadata",
            iiextmetadatafilter="License|LicenseUrl|LicenseShortName|Artist|"
            "ImageDescription|Credit",
            cllimit="max",
        )
        for p in d["query"]["pages"]:
            t = p["title"]
            ii = (p.get("imageinfo") or [{}])[0]
            em = ii.get("extmetadata") or {}
            mime = ii.get("mime", "")
            cats = " ".join(c["title"] for c in p.get("categories") or [])
            desc = " ".join(
                (em.get(k) or {}).get("value", "")
                for k in ("ImageDescription", "Credit")
            )
            lic = licence_of(em)
            if lic is None:
                tid = (em.get("License") or {}).get("value", "") or "none"
                excluded[f"licence {tid}"] += 1
            elif "midi" in mime or t.lower().endswith((".mid", ".midi")):
                excluded["MIDI"] += 1
            elif not (mime.startswith("audio/") or mime == "application/ogg"):
                excluded[f"not audio ({mime or 'no mime'})"] += 1
            elif SPEECH.search(t) or SPEECH.search(cats):
                excluded["speech-like title or category"] += 1
            elif re.search(r"youtu\.?be", desc, re.I):
                excluded["YouTube import"] += 1
            else:
                name = t.split(":", 1)[1].replace(" ", "_")
                rel = f"ARCHIVES/{name}"
                rows.append((rel, ii["url"], ii.get("size", ""), ""))
                artist = clean((em.get("Artist") or {}).get("value", ""))
                lic_rows.append(
                    (rel, lic[0], lic[1], artist, ii.get("descriptionurl", ""))
                )
                kept_lic[lic[0]] += 1
                secs += duration(ii)
                nbytes += int(ii.get("size") or 0)
        log(f"{min(i + 50, len(order))}/{len(order)} files read, {len(rows)} kept")
    rows.sort()
    lic_rows.sort()
    out = write_manifest(
        a.out,
        a.corpus,
        rows,
        [
            "Wikimedia Commons, categories "
            + "; ".join(f"{c}{' (deep)' if d else ''}" for c, d in CATEGORIES)
            + f"; licence cc0/pd/cc-by only; per-file licence: {a.corpus}.licences.tsv",
            f"{len(rows)} files, {secs / 3600:.1f} h",
        ],
    )
    write_licences(a.out, a.corpus, lic_rows)
    print(
        f"{a.corpus}: {len(rows)} files, {nbytes / 1e9:.2f} GB, {secs / 3600:.1f} h; "
        f"kept by licence {dict(kept_lic)}; excluded {dict(excluded.most_common())}"
        f" -> {out}"
    )
    print(f"{a.corpus}: subcategories not entered: {sorted(skipped_cats)}")


if __name__ == "__main__":
    main()
