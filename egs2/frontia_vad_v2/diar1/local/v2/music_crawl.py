#!/usr/bin/env python3
"""List the items of the crawled music sources with their own licence (metadata only).

    python3 local/v2/music_crawl.py ccmixter <out.tsv>
    python3 local/v2/music_crawl.py commons <out.tsv>
    python3 local/v2/music_crawl.py europeana <out.tsv> --key_file <file with a wskey>
    python3 local/v2/music_crawl.py loc <out.tsv> [--max_year 1925]

One row per item (columns HEAD): source, item_id, item_url, media_url, retrieved (UTC
date of the listing), licence (SPDX id or URI), licence_url, credit (the attribution
line), may_have_vocals, accepted, reason (why an item is excluded), extra (source-
specific evidence: tags, PD template, provider, rights advisory, date, duration). The
media are fetched afterwards by local/v2/packed_fetch.py, which adds the sha256 of each
file.

- ccmixter (ccmixter.org/api/query): uploads tagged "instrumental" with lic=by (CC BY,
  any version) or lic=zero (CC0); credit "<title> by <artist> (ccMixter), <licence>";
  may_have_vocals = 1 when a tag names vocals, a cappella, voice or spoken word. -
  commons (commons.wikimedia.org API): files of Category:Audio files of instrumental
  music and Category:Musopen (subcategories included, one level); licence = the file's
  LicenseShortName; accepted: CC0, Public Domain / PD marks, CC BY; dropped: BY-SA and
  everything else. For "Public domain" files the PD template is kept in extra, and
  pd_note=composition_only flags templates that speak only of the work (e.g. PD-old for
  the composer) without a recording-level PD or free licence. - europeana
  (api.europeana.eu search): music (query "music", reusability open) TYPE:SOUND items
  whose RIGHTS is the Public Domain Mark, CC0 or CC BY (any version); dropped: BY-SA and
  the rest; provider and media URL (edmIsShownBy) per item. - loc (www.loc.gov National
  Jukebox, fo=json): items with subject "instrumental" dated up to --max_year; the
  rights advisory and date per item and the WAV master URL when there is one. accepted
  stays 0 with reason "rights decision pending": the advisory names the rights holder's
  courtesy, not a licence (the US public-domain status of pre-1926 recordings is a legal
  judgement for Shinji).

Requests are polite: at most one per second per source, retried with back-off, with a
User-Agent that names no person.
"""

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters_nonspeech import licence_class  # noqa: E402

UA = "frontia-vad-data-collector/1.0 (research dataset licence audit)"
HEAD = [
    "source",
    "item_id",
    "item_url",
    "media_url",
    "retrieved",
    "licence",
    "licence_url",
    "credit",
    "may_have_vocals",
    "accepted",
    "reason",
    "extra",
]
VOCAL = re.compile(
    r"vocal|a ?cappella|acapella|voice|spoken|singer|singing|choir|rap\b", re.I
)
TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")
_last = [0.0]


def get(url, tries=5, as_json=True):
    for k in range(tries):
        wait = 1.0 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            return json.loads(data) if as_json else data
        except Exception as e:  # noqa: BLE001 - retried, then raised
            if k == tries - 1:
                raise RuntimeError(f"{url}: {e}") from e
            time.sleep(2 ** (k + 1))


def spdx(cls_url: str) -> str:
    """SPDX id of a Creative Commons licence or mark URL, else the URL."""
    u = cls_url.lower()
    m = re.search(r"licenses/(by(?:-[a-z-]+)?)/(\d\.\d)", u)
    if m:
        return f"CC-{m.group(1).upper()}-{m.group(2)}"
    if "publicdomain/zero" in u:
        return "CC0-1.0"
    if "publicdomain/mark" in u:
        return "LicenseRef-PDM-1.0"
    return cls_url


def ccmixter(out):
    for lic in ("by", "zero"):
        offset = 0
        while True:
            q = urllib.parse.urlencode(
                {
                    "tags": "instrumental",
                    "lic": lic,
                    "f": "json",
                    "limit": 100,
                    "offset": offset,
                }
            )
            batch = get(f"http://ccmixter.org/api/query?{q}")
            if not batch:
                break
            for r in batch:
                url = r.get("license_url", "")
                cls = licence_class(f"{r.get('license_name', '')} {url}")
                tags = r.get("upload_tags", "")
                files = [
                    f
                    for f in (r.get("files") or [])
                    if str(f.get("file_name", ""))
                    .lower()
                    .endswith((".mp3", ".flac", ".wav", ".ogg"))
                ]
                media = files[0].get("download_url", "") if files else ""
                ok = cls in ("BY", "PD/CC0") and bool(media)
                reason = (
                    "" if ok else ("no audio file" if not media else f"licence {cls}")
                )
                artist = r.get("user_real_name") or r.get("user_name", "")
                out.writerow(
                    [
                        "ccmixter",
                        r.get("upload_id"),
                        r.get("file_page_url", ""),
                        media,
                        TODAY,
                        spdx(url),
                        url,
                        f"\"{r.get('upload_name', '')}\" by {artist} (ccMixter), "
                        f"{r.get('license_name', '')}",
                        int(bool(VOCAL.search(tags))),
                        int(ok),
                        reason,
                        json.dumps(
                            {
                                "tags": tags.strip(","),
                                "date": r.get("upload_date", ""),
                                "file": files[0].get("file_name", "") if files else "",
                            }
                        ),
                    ]
                )
            offset += len(batch)
            if len(batch) < 100:
                break


COMMONS = "https://commons.wikimedia.org/w/api.php"
PD_RECORDING = re.compile(
    r"pd-(self|author|user|ineligible|usgov|us-gov|release)|cc0|"
    r"public domain mark|pdm",
    re.I,
)


def commons_members(cat, depth=1):
    seen, cont = [], {}
    while True:
        q = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": cat,
            "cmlimit": 500,
            "cmtype": "file|subcat",
            "format": "json",
            **cont,
        }
        d = get(f"{COMMONS}?{urllib.parse.urlencode(q)}")
        for m in d["query"]["categorymembers"]:
            if m["ns"] == 14 and depth > 0:
                seen += commons_members(m["title"], depth - 1)
            elif m["ns"] == 6:
                seen.append(m["title"])
        if "continue" not in d:
            return seen
        cont = d["continue"]


def commons(out):
    titles = []
    for cat in ("Category:Audio files of instrumental music", "Category:Musopen"):
        titles += [(t, cat) for t in commons_members(cat)]
    done = set()
    for i in range(0, len(titles), 50):
        chunk = [t for t, _ in titles[i : i + 50] if t not in done]
        cats = dict(titles[i : i + 50])
        if not chunk:
            continue
        q = {
            "action": "query",
            "prop": "imageinfo|categories",
            "titles": "|".join(chunk),
            "iiprop": "url|extmetadata|sha1|size|mime",
            "cllimit": 500,
            "format": "json",
        }
        d = get(f"{COMMONS}?{urllib.parse.urlencode(q)}")
        for p in d["query"]["pages"].values():
            t = p["title"]
            done.add(t)
            ii = (p.get("imageinfo") or [{}])[0]
            em = ii.get("extmetadata") or {}
            short = (em.get("LicenseShortName") or {}).get("value", "")
            url = (em.get("LicenseUrl") or {}).get("value", "")
            cls = licence_class(f"{short} {url}")
            artist = re.sub(r"<[^>]+>", "", (em.get("Artist") or {}).get("value", ""))
            pcats = " ".join(c["title"] for c in p.get("categories") or [])
            tmpl = re.findall(
                r"Category:(PD[-_ ][^|]*|CC-Zero|Public domain[^|]*)", pcats
            )
            pd_note = ""
            if (
                cls == "PD/CC0"
                and "cc0" not in short.lower()
                and not PD_RECORDING.search(f"{short} {' '.join(tmpl)}")
            ):
                pd_note = "composition_only"
            ok = cls in ("BY", "PD/CC0")
            reason = "" if ok else f"licence {cls}: {short}"
            vocal = int(bool(VOCAL.search(f"{t} {pcats}")))
            out.writerow(
                [
                    "commons",
                    t,
                    ii.get("descriptionurl", ""),
                    ii.get("url", ""),
                    TODAY,
                    spdx(url) if url else short,
                    url,
                    f"{t[5:]} by {artist or 'unknown'} (Wikimedia Commons), {short}",
                    vocal,
                    int(ok),
                    reason,
                    json.dumps(
                        {
                            "category": cats.get(t, ""),
                            "pd_templates": tmpl,
                            "pd_note": pd_note,
                            "commons_sha1": ii.get("sha1", ""),
                            "size": ii.get("size"),
                            "mime": ii.get("mime", ""),
                        }
                    ),
                ]
            )


def europeana(out, key):
    cursor = "*"
    while cursor:
        q = urllib.parse.urlencode(
            {
                "wskey": key,
                "query": "music",
                "qf": "TYPE:SOUND",
                "reusability": "open",
                "rows": 100,
                "cursor": cursor,
                "profile": "rich",
            }
        )
        d = get(f"https://api.europeana.eu/record/v2/search.json?{q}")
        for it in d.get("items", []):
            rights = (it.get("rights") or [""])[0]
            cls = licence_class(rights)
            ok = cls in ("BY", "PD/CC0")
            media = (it.get("edmIsShownBy") or [""])[0]
            provider = (it.get("dataProvider") or [""])[0]
            title = (it.get("title") or [""])[0]
            creator = (it.get("dcCreator") or [""])[0]
            text = f"{title} {' '.join(it.get('dcSubject') or [])}"
            out.writerow(
                [
                    "europeana",
                    it.get("id", ""),
                    it.get("guid", ""),
                    media,
                    TODAY,
                    spdx(rights),
                    rights,
                    f"{title} by {creator or 'unknown'}, {provider} (Europeana), "
                    f"{spdx(rights)}",
                    int(bool(VOCAL.search(text))),
                    int(ok and bool(media)),
                    (
                        ""
                        if ok and media
                        else ("no media URL" if ok else f"rights {cls}: {rights}")
                    ),
                    json.dumps(
                        {
                            "provider": provider,
                            "type": it.get("type", ""),
                            "year": it.get("year", ""),
                        }
                    ),
                ]
            )
        cursor = d.get("nextCursor")


def loc(out, max_year):
    page = 1
    while True:
        q = urllib.parse.urlencode(
            {
                "fo": "json",
                "c": 100,
                "sp": page,
                "fa": "subject:instrumental",
                "dates": f"1890/{max_year}",
            }
        )
        d = get(f"https://www.loc.gov/collections/national-jukebox/?{q}")
        for r in d.get("results", []):
            date = str(r.get("date", ""))
            year = int(date[:4]) if date[:4].isdigit() else 9999
            item = get(f"{r['id'].replace('http://', 'https://')}?fo=json")
            it = item.get("item", {})
            wav = ""
            for res in item.get("resources", []):
                for fs in res.get("files", []):
                    for f in fs:
                        if f.get("mimetype") == "audio/wav" and f.get("url"):
                            wav = wav or f["url"]
            advisory = str(it.get("rights_advisory") or "")
            ok_date = year <= max_year
            out.writerow(
                [
                    "loc_jukebox",
                    r["id"],
                    r["id"],
                    wav,
                    TODAY,
                    "LicenseRef-LoC-rights-advisory",
                    r["id"],
                    f"{r.get('title', '')} ({date}), Library of Congress National "
                    f"Jukebox",
                    int(bool(VOCAL.search(" ".join(r.get("subject") or [])))),
                    0,
                    (
                        "rights decision pending"
                        if ok_date
                        else f"date {date} after {max_year}"
                    ),
                    json.dumps(
                        {
                            "date": date,
                            "rights_advisory": advisory,
                            "subject": r.get("subject"),
                            "wav": bool(wav),
                        }
                    ),
                ]
            )
        if not d.get("pagination", {}).get("next"):
            break
        page += 1


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source", choices=("ccmixter", "commons", "europeana", "loc"))
    p.add_argument("out", type=Path)
    p.add_argument("--key_file", type=Path)
    p.add_argument("--max_year", type=int, default=1925)
    a = p.parse_args()
    with a.out.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(HEAD)
        if a.source == "ccmixter":
            ccmixter(w)
        elif a.source == "commons":
            commons(w)
        elif a.source == "europeana":
            europeana(w, a.key_file.read_text().strip())
        else:
            loc(w, a.max_year)
    rows = list(csv.DictReader(a.out.open(), delimiter="\t"))
    acc = sum(1 for r in rows if r["accepted"] == "1")
    reasons = {}
    for r in rows:
        k = r["reason"].split(":")[0] if r["reason"] else "accepted"
        reasons[k] = reasons.get(k, 0) + 1
    print(f"{a.source}: {len(rows)} items, {acc} accepted; {reasons}")


if __name__ == "__main__":
    main()
