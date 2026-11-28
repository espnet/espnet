#!/usr/bin/env python3
"""Write a download manifest (local/v2/downloads/<corpus>.tsv) from a podcast RSS feed.

    python3 local/v2/rss_manifest.py <corpus> <feed url> [--licence cc-by-4.0] \
        [--exclude-title REGEX] [--out local/v2/downloads]

The feed's own licence fields (channel <copyright>, <podcast:license>,
<creativeCommons:license>, <dc:rights>) are printed and converted to SPDX-like ids
(manifest_common.licence_id). At least one field must state a licence, every
stated licence must be the same id, and that id must equal --licence (case
ignored) and be accepted (CC0, PD or CC BY without NC/SA/ND); otherwise nothing is
written and the exit status is 1.

One line per episode with an audio enclosure: corpus, ARCHIVES/<name>, the
enclosure URL, no size (enclosure lengths in feeds often differ from the served
file, so download.py takes the served Content-Length), no
checksum. <name> is the enclosure file name when those are unique in the feed,
otherwise the episode guid with characters other than [A-Za-z0-9._-] replaced by
"_" plus the enclosure's extension. Episodes are left out when their title
matches --exclude-title, when they have no audio enclosure, or when the episode
carries its own licence field with a different id. Total hours are summed from
<itunes:duration> where it is present.
"""

import argparse
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manifest_common import (  # noqa: E402
    DOWNLOADS,
    Fetcher,
    licence_accepted,
    licence_id,
    write_manifest,
)

NS = {
    "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
    "podcast": "https://podcastindex.org/namespace/1.0",
    "cc": "http://backend.userland.com/creativeCommonsRssModule",
    "dc": "http://purl.org/dc/elements/1.1/",
}
LICENCE_TAGS = (
    "copyright",
    "{https://podcastindex.org/namespace/1.0}license",
    "{http://backend.userland.com/creativeCommonsRssModule}license",
    "{http://purl.org/dc/elements/1.1/}rights",
)
AUDIO_EXT = (".mp3", ".m4a", ".ogg", ".oga", ".opus", ".aac", ".flac", ".wav")


def licence_fields(el):
    """(tag, text) of the licence fields that are direct children of el."""
    found = []
    for tag in LICENCE_TAGS:
        for f in el.findall(tag):
            text = " ".join(x for x in ((f.text or "").strip(), f.get("url", "")) if x)
            found.append((tag.split("}")[-1], text))
    return found


def seconds(duration: str):
    """Seconds of an itunes:duration ("HH:MM:SS", "MM:SS" or seconds).

    None when the duration is absent, unparsable or zero.
    """
    d = (duration or "").strip()
    if not d:
        return None
    try:
        s = 0.0
        for part in d.split(":"):
            s = s * 60 + float(part)
        return s or None
    except ValueError:
        return None


def is_audio(url: str, mime: str) -> bool:
    path = urllib.parse.urlparse(url).path.lower()
    return mime.lower().startswith("audio/") or path.endswith(AUDIO_EXT)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("corpus")
    ap.add_argument("feed", help="RSS feed URL")
    ap.add_argument("--licence", default="cc-by-4.0", help="licence the feed states")
    ap.add_argument("--exclude-title", default="", help="drop episodes matching this")
    ap.add_argument("--out", type=Path, default=DOWNLOADS)
    a = ap.parse_args()

    channel = ET.fromstring(Fetcher().raw(a.feed)).find("channel")
    fields = licence_fields(channel)
    for tag, text in fields:
        print(f"{a.corpus}: feed {tag}: {text!r} -> {licence_id(text) or '-'}")
    ids = {licence_id(text) for _, text in fields} - {""}
    want = a.licence.upper()
    if len(ids) != 1 or want not in ids or not licence_accepted(want):
        print(
            f"{a.corpus}: feed states {sorted(ids) or 'no licence'}, expected {want};"
            " no manifest written",
            file=sys.stderr,
        )
        sys.exit(1)

    dropped, episodes = Counter(), []
    for item in channel.findall("item"):
        title = (item.findtext("title") or "").strip()
        enc = item.find("enclosure")
        url = enc.get("url", "") if enc is not None else ""
        own = {licence_id(t) for _, t in licence_fields(item)} - {""}
        if a.exclude_title and re.search(a.exclude_title, title):
            dropped["title excluded"] += 1
        elif not url or not is_audio(url, enc.get("type", "")):
            dropped["no audio enclosure"] += 1
        elif own and own != {want}:
            dropped[f"episode licence {sorted(own)}"] += 1
        else:
            dur = seconds(item.findtext("itunes:duration", "", NS))
            guid = (item.findtext("guid") or url).strip()
            episodes.append((url, "", dur, guid))
    names = [urllib.parse.urlparse(u).path.rsplit("/", 1)[-1] for u, *_ in episodes]
    if len(set(names)) < len(names):
        names = [
            re.sub(r"[^A-Za-z0-9._-]", "_", g)[-120:]
            + Path(urllib.parse.urlparse(u).path).suffix
            for u, _, _, g in episodes
        ]
    rows = [
        (f"ARCHIVES/{n}", u, size, "") for n, (u, size, _, _) in zip(names, episodes)
    ]
    if not rows:
        print(f"{a.corpus}: no episode left", file=sys.stderr)
        sys.exit(1)
    durs = [d for *_, d, _ in episodes if d is not None]
    gb = sum(int(s) for _, _, s, _ in rows if s) / 1e9
    sized = sum(1 for _, _, s, _ in rows if s)
    hours = sum(durs) / 3600
    title = (channel.findtext("title") or "").strip()
    out = write_manifest(
        a.out,
        a.corpus,
        rows,
        [
            f"RSS feed {a.feed} ({title}), feed licence {want}: "
            + "; ".join(f"{t}={x}" for t, x in fields),
            f"{len(rows)} episodes, {hours:.1f} h ({len(durs)} with itunes:duration)",
        ],
    )
    print(
        f"{a.corpus}: {len(rows)} episodes, {gb:.2f} GB ({sized} sized), "
        f"{hours:.1f} h ({len(durs)} with duration); dropped {dict(dropped)} -> {out}"
    )


if __name__ == "__main__":
    main()
