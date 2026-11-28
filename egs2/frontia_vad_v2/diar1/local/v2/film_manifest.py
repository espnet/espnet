#!/usr/bin/env python3
"""Write the blender_films download manifest and licence table (Blender open movies).

    python3 local/v2/film_manifest.py [--corpus blender_films] \
        [--out local/v2/downloads]

download.blender.org answers every automated request with a Cloudflare
challenge, so the files come from:

- the dotsrc.org mirror of download.blender.org/demo (MIRROR): Tears of Steel
  (TOS_DVDSTEREOMIX.aif, TOS_MUSIC+FX_NO_DIALOGUE.aif, copyright.txt and the
  subtitles/*.srt of the directory listing) and Sintel.2010.1080p.mkv with the
  md5 of its .md5 file;
- the Internet Archive item ElephantsDream: the stereo film mix
  ED-CM-St-16bit.flac with the md5 the item lists;
- video.blender.org (PeerTube API /api/v1/videos/<id>) for the Blender Studio
  films in PEERTUBE (object-storage fileUrl): the audio-only file (resolution 0)
  when the video has one, otherwise the smallest non-fragmented MP4. PeerTube
  re-encodes the audio of every resolution the same way, so the smallest file
  has the same audio as the largest. A video whose downloads are disabled
  stops the script.

Licences (licence table: relative_path, licence, licence_url, creator,
source_page):
- Tears of Steel film files: CC-BY-3.0. TOS_MUSIC+FX_NO_DIALOGUE.aif is music and
  effects without dialogue; ToS/copyright.txt in the same folder puts "the
  Original Soundtrack files" under CC BY-ND 3.0, so it is marked CC-BY-ND-3.0
  (evaluation-only at the licence gate) until that is decided otherwise.
- Sintel: CC-BY-3.0 (Internet Archive item Sintel, licenseurl by/3.0).
- Elephants Dream: CC-BY-3.0-US (licenseurl of the Internet Archive item).
- Studio films: the licence in PEERTUBE with the page that states it (the film's
  licensing page, the PeerTube licence field, or the Blender Studio terms:
  "Unless notified otherwise, all digital content ... is available under the
  Creative Commons Attribution 4.0").

Not taken: Singularity (CC BY-SA, downloads disabled), Agent 327: Operation
Barbershop (no CC BY statement; its description says "released as Creative
Commons for Blender Cloud subscribers"), ED-ME-5.1-DVD.ac3 and
sintel-m+e-st.flac (only on download.blender.org).
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
    log,
    write_licences,
    write_manifest,
)

MIRROR = "https://mirrors.dotsrc.org/blender/demo/movies/"
PEERTUBE = "https://video.blender.org"
BY3 = "https://creativecommons.org/licenses/by/3.0/"
BY4 = "https://creativecommons.org/licenses/by/4.0/"
TERMS = "https://studio.blender.org/terms-and-conditions/"
TOS_COPYRIGHT = MIRROR + "ToS/copyright.txt"
STUDIO = "Blender Studio (Blender Foundation)"

# (directory, shortUUID, licence, licence_url as the evidence)
PEERTUBE_FILMS = [
    ("cosmos_laundromat", "wfW3bDTkUhQKRnEfT9Wpeq", "CC-BY", "peertube"),
    ("sprite_fright", "mziZQzmf95pGMSqk7BPZvi", "CC-BY-4.0", TERMS),
    ("charge", "1AKMruUNPJQQQ4bvtHYVZQ", "CC-BY-4.0", TERMS),
    (
        "wing_it",
        "pkDDcE7HiLU3UTGmRV1nJG",
        "CC-BY-4.0",
        "https://studio.blender.org/projects/wing-it/pages/licensing/",
    ),
    ("spring", "8B5QmoLSS3mWJ4fmZZu3Ye", "CC-BY", "peertube"),
    (
        "glass_half",
        "dnayqfMS4PAE9pSZ3kyVUb",
        "CC-BY-3.0",
        "https://studio.blender.org/projects/glass-half/pages/licensing/",
    ),
    ("caminandes_2", "x3QKwgFXGZQ7yvCTD2wHJA", "CC-BY-4.0", TERMS),
    ("caminandes_3", "5ruRYckTcQMTMQvyqMnVsr", "CC-BY", "peertube"),
    (
        "coffee_run",
        "xymLD6rkpHNug3fzzMyhmZ",
        "CC-BY-4.0",
        "https://studio.blender.org/projects/coffee-run/pages/licensing/",
    ),
    ("hero", "8KerEHuf8qyGtCVoAoEdzQ", "CC-BY", "peertube"),
    ("daily_dweebs", "gdfAeNysimipmpo7AbgaTp", "CC-BY", "peertube"),
    ("big_buck_bunny", "dmhvQNzwBnrWy1iYzVv5g7", "CC-BY", "peertube"),
]


def mirror_listing(fetch, url):
    """File names of a dotsrc directory listing (directories left out)."""
    page = fetch.raw(url).decode("utf-8", "replace")
    names = re.findall(r'<span href="([^"/]+)" title=', page)
    return [urllib.parse.unquote(html.unescape(n)) for n in names]


def peertube_file(video):
    """(url, size, resolution) of the audio-only file, else the smallest MP4."""
    files = list(video.get("files") or [])
    for p in video.get("streamingPlaylists") or []:
        files += p.get("files") or []
    # fileUrl is the stable object-storage URL; fileDownloadUrl redirects to a
    # presigned URL that expires after a day
    files = [f for f in files if "/web_videos/" in f.get("fileUrl", "")]
    f = min(files, key=lambda f: (f["resolution"]["id"] != 0, f["size"]))
    return f["fileUrl"], f["size"], f["resolution"]["id"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default="blender_films")
    ap.add_argument("--out", type=Path, default=DOWNLOADS)
    a = ap.parse_args()
    fetch = Fetcher()
    rows, lic_rows, secs = [], [], 0.0

    def add(rel, url, size, check, lic, lic_url, page, creator=STUDIO):
        rows.append((rel, url, size, check))
        lic_rows.append((rel, lic, lic_url, creator, page))
        log(f"{a.corpus}: {lic:13s} {int(size or 0) / 1e6:9.1f} MB {rel}")

    tos = "https://mango.blender.org/"
    for name in ("TOS_DVDSTEREOMIX.aif", "copyright.txt"):
        url = MIRROR + "ToS/" + urllib.parse.quote(name)
        add(
            f"ARCHIVES/tears_of_steel/{name}",
            url,
            fetch.size(url),
            "",
            "CC-BY-3.0",
            BY3,
            tos,
        )
    name = "TOS_MUSIC+FX_NO_DIALOGUE.aif"
    url = MIRROR + "ToS/" + urllib.parse.quote(name)
    add(
        f"ARCHIVES/tears_of_steel/{name}",
        url,
        fetch.size(url),
        "",
        "CC-BY-ND-3.0",
        TOS_COPYRIGHT,
        tos,
    )
    for name in mirror_listing(fetch, MIRROR + "ToS/subtitles/"):
        if name.endswith(".srt"):
            url = MIRROR + "ToS/subtitles/" + urllib.parse.quote(name)
            add(
                f"ARCHIVES/tears_of_steel/subtitles/{name}",
                url,
                fetch.size(url),
                "",
                "CC-BY-3.0",
                BY3,
                tos,
            )

    url = MIRROR + "Sintel.2010.1080p.mkv"
    md5 = fetch.raw(url + ".md5").decode().split()[0]
    add(
        "ARCHIVES/sintel/Sintel.2010.1080p.mkv",
        url,
        fetch.size(url),
        f"md5:{md5}",
        "CC-BY-3.0",
        BY3,
        "https://archive.org/details/Sintel",
    )

    item = fetch.json("https://archive.org/metadata/ElephantsDream")
    f = next(x for x in item["files"] if x["name"] == "ED-CM-St-16bit.flac")
    add(
        "ARCHIVES/elephants_dream/ED-CM-St-16bit.flac",
        "https://archive.org/download/ElephantsDream/ED-CM-St-16bit.flac",
        f["size"],
        f"md5:{f['md5']}",
        "CC-BY-3.0-US",
        item["metadata"]["licenseurl"],
        "https://archive.org/details/ElephantsDream",
        creator="Orange Open Movie Project Studio (Blender Foundation)",
    )
    secs += float(f.get("length") or 0)

    res = Counter()
    for d, uuid, lic, evidence in PEERTUBE_FILMS:
        v = fetch.json(f"{PEERTUBE}/api/v1/videos/{uuid}")
        if not v.get("downloadEnabled"):
            log(f"{a.corpus}: downloads disabled for {v['name']}; nothing written")
            sys.exit(1)
        page = f"{PEERTUBE}/w/{uuid}"
        if evidence == "peertube":
            if v["licence"]["id"] != 1:
                log(f"{a.corpus}: {v['name']} licence {v['licence']}; nothing written")
                sys.exit(1)
            evidence = page
        url, size, r = peertube_file(v)
        name = url.rsplit("/", 1)[-1]
        add(f"ARCHIVES/{d}/{name}", url, size, "", lic, evidence, page)
        secs += v.get("duration") or 0
        res["audio only" if r == 0 else f"{r}p"] += 1

    rows.sort()
    lic_rows.sort()
    nbytes = sum(int(r[2] or 0) for r in rows)
    out = write_manifest(
        a.out,
        a.corpus,
        rows,
        [
            "Blender open movies: Tears of Steel and Sintel from the dotsrc.org "
            "mirror of download.blender.org/demo/movies, Elephants Dream from "
            "archive.org/details/ElephantsDream, Blender Studio films from "
            "video.blender.org; CC BY parts only (TOS music+FX marked CC-BY-ND-3.0); "
            f"per-file licence: {a.corpus}.licences.tsv",
            f"{len(rows)} files, {nbytes / 1e9:.2f} GB; {secs / 3600:.2f} h of films "
            "excluding Tears of Steel and Sintel",
        ],
    )
    write_licences(a.out, a.corpus, lic_rows)
    print(
        f"{a.corpus}: {len(rows)} files, {nbytes / 1e9:.2f} GB; PeerTube files "
        f"{dict(res)} -> {out}"
    )


if __name__ == "__main__":
    main()
