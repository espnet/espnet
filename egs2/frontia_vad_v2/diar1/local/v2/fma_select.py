#!/usr/bin/env python3
"""Select the FMA tracks whose licence allows training, from fma_metadata, per track.

    python3 local/v2/fma_select.py <fma_metadata.zip> <out.tsv> [--full_zip_url URL]

tracks.csv (fma_metadata/tracks.csv, three header rows) gives each track's licence
('track', 'license'), genre_top, tags and duration; raw_tracks.csv (one header row),
joined by track_id, gives what tracks.csv leaves out: the copyright notes
(track_copyright_c / track_copyright_p), the licence URL, the uploader's instrumental
flag (track_instrumental) and the credit fields (artist_name, track_title, track_url).
A track is accepted when its licence is CC BY (1.0-4.0), CC0 or the Public Domain Mark
(adapters_nonspeech.licence_class: "BY" or "PD/CC0"), the licence is not FMA's "Public
Domain / Sound Recording Common Law Protection" (the composition is public domain, the
recording is not), and none of its copyright notes names a more restrictive licence (NC,
ND, SA, Sampling+): e.g. WFMU tracks labelled CC BY 4.0 whose copyright reads "by-nc-sa
3.0 US" are dropped. NC, ND, SA, Sampling+, the Free Art License and unknown licences are
dropped. Singing is speech: may_have_vocals = 0 only for genre_top "Instrumental", a
track tag naming instrumental music, or the uploader's instrumental flag.

With --full_zip_url, the size and CRC32 of each accepted track in fma_full.zip are read
from its central directory (HTTP range requests; nothing else is downloaded).

Writes one row per track: track_id, member (path in fma_full.zip), accepted, reason,
licence, licence_class, licence_url, copyright_c, copyright_p, genre_top,
may_have_vocals, duration_s, size_bytes, crc32, artist, title, url.
"""

import argparse
import ast
import csv
import io
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters_nonspeech import licence_class  # noqa: E402
from http_range import HttpRangeFile  # noqa: E402

RESTRICT = {"NC", "ND", "SA", "Sampling+"}
INSTRUMENTAL = re.compile(r"instrumental", re.I)


def tracks(meta_zip: Path):
    """Rows of tracks.csv as dicts keyed "<group>.<field>" (two header rows)."""
    with zipfile.ZipFile(meta_zip) as z:
        # fma_metadata/tracks.csv, not raw_tracks.csv (one header row, other columns),
        # which also ends in "tracks.csv" and comes first in the archive
        name = next(n for n in z.namelist() if Path(n).name == "tracks.csv")
        text = io.TextIOWrapper(z.open(name), encoding="utf-8", newline="")
        r = csv.reader(text)
        top, sub, _ = next(r), next(r), next(r)
        keys = ["track_id"] + [f"{a}.{b}" for a, b in zip(top[1:], sub[1:])]
        for row in r:
            yield dict(zip(keys, row))


def raw_tracks(meta_zip: Path):
    """{track_id: row of fma_metadata/raw_tracks.csv} (one header row)."""
    with zipfile.ZipFile(meta_zip) as z:
        name = next(n for n in z.namelist() if Path(n).name == "raw_tracks.csv")
        text = io.TextIOWrapper(z.open(name), encoding="utf-8", newline="")
        return {r["track_id"]: r for r in csv.DictReader(text)}


COMMON_LAW = re.compile(r"common law", re.I)


def decide(t, raw):
    lic = t.get("track.license", "")
    cls = licence_class(lic)
    if cls not in ("BY", "PD/CC0"):
        return 0, f"licence {cls}", cls
    if COMMON_LAW.search(lic):
        return 0, "licence: sound recording common-law protection", cls
    for k in ("track_copyright_c", "track_copyright_p"):
        c = licence_class(raw.get(k, "") or "")
        if c in RESTRICT:
            return 0, f"{k.split('_', 1)[1]} names {c}", cls
    return 1, "", cls


def vocals(t, raw):
    if t.get("track.genre_top", "") == "Instrumental":
        return 0
    if str(raw.get("track_instrumental", "")).strip() == "1":
        return 0
    try:
        tags = ast.literal_eval(t.get("track.tags", "") or "[]")
    except (ValueError, SyntaxError):
        tags = []
    return 0 if any(INSTRUMENTAL.search(str(x)) for x in tags) else 1


def member(track_id: str) -> str:
    tid = f"{int(track_id):06d}"
    return f"fma_full/{tid[:3]}/{tid}.mp3"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("meta_zip", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--full_zip_url")
    a = p.parse_args()
    info = {}
    if a.full_zip_url:
        with zipfile.ZipFile(HttpRangeFile(a.full_zip_url)) as z:
            info = {i.filename: (i.file_size, i.CRC) for i in z.infolist()}
        print(f"fma_full.zip: {len(info)} members in its central directory")
    raws = raw_tracks(a.meta_zip)
    n = acc = 0
    hours = acc_h = acc_bytes = 0.0
    reasons = {}
    with a.out.open("w") as f:
        f.write(
            "track_id\tmember\taccepted\treason\tlicence\tlicence_class\tlicence_url\t"
            "copyright_c\tcopyright_p\tgenre_top\tmay_have_vocals\tduration_s\tsize_bytes\t"
            "crc32\tartist\ttitle\turl\n"
        )
        for t in tracks(a.meta_zip):
            raw = raws.get(t["track_id"], {})
            ok, why, cls = decide(t, raw)
            m = member(t["track_id"])
            size, crc = info.get(m, ("", ""))
            if info and ok and m not in info:
                ok, why = 0, "not in fma_full.zip"
            d = float(t.get("track.duration") or 0)
            n += 1
            hours += d / 3600
            reasons[why or "accepted"] = reasons.get(why or "accepted", 0) + 1
            if ok:
                acc += 1
                acc_h += d / 3600
                acc_bytes += float(size or 0)
            vals = [
                t["track_id"],
                m,
                ok,
                why,
                t.get("track.license", ""),
                cls,
                raw.get("license_url", ""),
                raw.get("track_copyright_c", ""),
                raw.get("track_copyright_p", ""),
                t.get("track.genre_top", ""),
                vocals(t, raw),
                f"{d:.1f}",
                size,
                f"{crc:08x}" if crc != "" else "",
                raw.get("artist_name", ""),
                raw.get("track_title", ""),
                raw.get("track_url", ""),
            ]
            f.write(
                "\t".join(str(v).replace("\t", " ").replace("\n", " ") for v in vals)
                + "\n"
            )
    print(
        f"tracks: {n} ({hours:.1f} h); accepted {acc} ({acc_h:.1f} h, "
        f"{acc_bytes / 1e9:.1f} GB in fma_full.zip)"
    )
    for k, v in sorted(reasons.items(), key=lambda z: -z[1]):
        print(f"  {v:7d}  {k}")
    if not n or not hours:
        raise SystemExit("no track with a duration: tracks.csv was not read correctly")
    joined = sum(1 for k in raws)
    print(f"raw_tracks.csv: {joined} rows joined by track_id")
    if not joined:
        raise SystemExit("raw_tracks.csv gave no rows: copyright notes cannot be checked")


if __name__ == "__main__":
    main()
