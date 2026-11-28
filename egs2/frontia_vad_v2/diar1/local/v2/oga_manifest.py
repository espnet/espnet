#!/usr/bin/env python3
"""Write the opengameart_voices download manifest and licence table (OpenGameArt).

    python3 local/v2/oga_manifest.py [--corpus opengameart_voices] \
        [--tag voice --tag vocal] [--out local/v2/downloads]

1. Collects the entries of type Sound (art type 13) carrying each --tag (the
   advanced search, 144 per page, pages read until one adds nothing new) plus
   the packs in SEEDS.
2. Reads each entry page: the License(s) field (licence names and links), the
   Copyright/Attribution Notice, the author and the File(s) field (URL and the
   length in the link's type attribute).
3. Keeps the entry when one of its licences is CC0 or CC BY
   (manifest_common.licence_accepted); an entry offered under several licences
   (e.g. CC-BY 3.0 and GPL 3.0) is taken under the accepted one, CC0 first.
   OGA-BY, CC BY-SA, GPL and LGPL alone are dropped.
4. Takes the entry's audio files and archives (AUDIO_EXT, ARCHIVE_EXT); images,
   text and project files are left out.

Rows: opengameart_voices, ARCHIVES/<entry slug>/<file name>, the file URL, the
length; no checksum. The licence table gives the licence id, its link on the
page, the author with the attribution notice, and the entry page.
Requests are sequential, one per second.
"""

import argparse
import html
import re
import sys
import urllib.parse
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manifest_common import (  # noqa: E402
    DOWNLOADS,
    Fetcher,
    clean,
    licence_accepted,
    licence_id,
    log,
    write_licences,
    write_manifest,
)

SITE = "https://opengameart.org"
SEEDS = [
    "voiceover-pack-40-lines",  # Kenney Voiceover Pack
    "voiceover-pack-fighter-40-taunts",  # Kenney Voiceover Pack: Fighter
    "female-rpg-voice-starter-pack",
    "voice-clip-packs-for-visual-novels-and-rpgs",
    "dialog-vocal-samples",
]
AUDIO_EXT = (".wav", ".ogg", ".oga", ".mp3", ".flac", ".opus", ".aif", ".aiff")
ARCHIVE_EXT = (".zip", ".7z", ".rar", ".tar.gz", ".tgz", ".tar.bz2", ".tar")


def field(page, name):
    """The HTML of the field-name-<name> block of an entry page, or ""."""
    m = re.search(
        rf'field-name-{name}\b.*?(?=<div class="field field-name-|<ul class="links)',
        page,
        re.S,
    )
    return m.group(0) if m else ""


def search(fetch, tag):
    """Slugs of the Sound entries carrying tag."""
    slugs, page = [], 0
    while True:
        q = urllib.parse.urlencode(
            {
                "field_art_tags_tid": tag,
                "field_art_type_tid[]": 13,
                "items_per_page": 144,
                "page": page,
            }
        )
        got = fetch.raw(f"{SITE}/art-search-advanced?{q}").decode("utf-8", "replace")
        new = [
            s
            for s in dict.fromkeys(re.findall(r'<a href="/content/([^"/]+)"', got))
            if s not in slugs and s != "faq"
        ]
        if not new:
            return slugs
        slugs += new
        page += 1


def entry(fetch, slug):
    """(licences [(name, url)], notice, author, files [(url, length)])."""
    page = fetch.raw(f"{SITE}/content/{slug}").decode("utf-8", "replace")
    lic = field(page, "field-art-licenses")
    licences = re.findall(
        r"<a href='([^']*)'[^>]*>.*?license-name'>([^<]+)<", lic, re.S
    )
    notice = clean(html.unescape(field(page, "field-copyright-notice")))
    notice = re.sub(r"^Copyright/Attribution Notice:\s*", "", notice)
    author = re.findall(
        r"<a href=\"/users/[^\"]+\">([^<]+)</a>", field(page, "author-submitter")
    )
    files = re.findall(
        r'<a href="(https://opengameart.org/sites/default/files/[^"]+)" '
        r'type="[^";]*; length=(\d+)"',
        field(page, "field-art-files"),
    )
    return [(n.strip(), u) for u, n in licences], notice, author, files


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default="opengameart_voices")
    ap.add_argument("--tag", action="append", default=None)
    ap.add_argument("--out", type=Path, default=DOWNLOADS)
    a = ap.parse_args()
    fetch = Fetcher()

    slugs = list(SEEDS)
    for tag in a.tag or ["voice", "vocal"]:
        got = search(fetch, tag)
        slugs += [s for s in got if s not in slugs]
        log(f"{a.corpus}: tag {tag}: {len(got)} entries, {len(slugs)} in all")

    rows, lic_rows, kept, excluded, packs = [], [], Counter(), Counter(), []
    for slug in slugs:
        licences, notice, author, files = entry(fetch, slug)
        ids = [(licence_id(n) or licence_id(u), u) for n, u in licences]
        ok = sorted(
            (x for x in ids if licence_accepted(x[0])), key=lambda x: x[0] != "CC0-1.0"
        )
        if not ok:
            excluded[f"licence {'/'.join(n for n, _ in licences) or 'none'}"] += 1
            continue
        lid, lic_url = ok[0]
        taken = 0
        for url, length in files:
            name = urllib.parse.unquote(url.rsplit("/", 1)[-1])
            if not name.lower().endswith(AUDIO_EXT + ARCHIVE_EXT):
                excluded[f"file type .{name.rsplit('.', 1)[-1].lower()}"] += 1
                continue
            rel = f"ARCHIVES/{slug}/{name}"
            rows.append((rel, url, length, ""))
            creator = "; ".join(author) + (f" ({notice})" if notice else "")
            lic_rows.append((rel, lid, lic_url, creator, f"{SITE}/content/{slug}"))
            taken += 1
        if taken:
            kept[lid] += 1
            packs.append(f"{slug}: {taken} files, {lid}")
        else:
            excluded["entry without audio or archive files"] += 1
    rows.sort()
    lic_rows.sort()
    nbytes = sum(int(r[2] or 0) for r in rows)
    out = write_manifest(
        a.out,
        a.corpus,
        rows,
        [
            "OpenGameArt Sound entries tagged "
            + ", ".join(a.tag or ["voice", "vocal"])
            + " plus "
            + ", ".join(SEEDS)
            + f"; licence CC0 / CC BY only; per-file licence: {a.corpus}.licences.tsv",
            f"{len(rows)} files from {sum(kept.values())} entries",
        ],
    )
    write_licences(a.out, a.corpus, lic_rows)
    for p in packs:
        log(f"{a.corpus}: {p}")
    print(
        f"{a.corpus}: {len(rows)} files, {nbytes / 1e9:.3f} GB; entries kept by "
        f"licence {dict(kept)}; excluded {dict(excluded.most_common())} -> {out}"
    )


if __name__ == "__main__":
    main()
