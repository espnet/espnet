#!/usr/bin/env python3
"""Write the tatoeba_audio download manifest, licence table and sentence text table.

    python3 local/v2/tatoeba_manifest.py [--corpus tatoeba_audio] \
        [--out local/v2/downloads]

1. Reads the weekly export sentences_with_audio.csv
   (downloads.tatoeba.org/exports/sentences_with_audio.tar.bz2; columns
   sentence_id, audio_id, username, license, attribution_url) and keeps the rows
   whose audio licence is "CC BY 4.0" or "CC0 1.0". An empty licence means the
   contributor allowed use on Tatoeba only; it is dropped like NC, ND and SA.
2. Reads sentences.csv (exports/sentences.tar.bz2; sentence_id, lang, text) as a
   stream and keeps the sentences of the kept audio.

Writes:
- <corpus>.tsv: ARCHIVES/<audio_id>.mp3 from
  https://tatoeba.org/en/audio/download/<audio_id>; no size, no checksum.
- <corpus>.licences.tsv: licence, licence URL, the contributor (username, and the
  attribution URL when one is given) and the sentence page.
- <corpus>.text.tsv: relative_path, audio_id, sentence_id, lang, text. The
  sentence text is CC BY 2.0 FR (Tatoeba sentences), separate from the audio
  licence.

The download has about 42k files from one host: run download.py for this corpus
alone with --jobs 1 (one request at a time).
"""

import argparse
import csv
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manifest_common import (  # noqa: E402
    DOWNLOADS,
    UA,
    clean,
    log,
    write_licences,
    write_manifest,
)

EXPORTS = "https://downloads.tatoeba.org/exports"
KEEP = {
    "CC BY 4.0": ("CC-BY-4.0", "https://creativecommons.org/licenses/by/4.0/"),
    "CC0 1.0": ("CC0-1.0", "https://creativecommons.org/publicdomain/zero/1.0/"),
}


def tsv_lines(name):
    """Lines of the single CSV member of exports/<name>.tar.bz2, read as a stream."""
    req = urllib.request.Request(f"{EXPORTS}/{name}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=600) as r:
        with tarfile.open(fileobj=r, mode="r|bz2") as t:
            for m in t:
                if m.isfile() and m.name.endswith(".csv"):
                    for line in t.extractfile(m):
                        yield line.decode("utf-8")
                    return


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default="tatoeba_audio")
    ap.add_argument("--out", type=Path, default=DOWNLOADS)
    a = ap.parse_args()

    audio, excluded = {}, Counter()
    for line in tsv_lines("sentences_with_audio.tar.bz2"):
        sid, aid, user, lic, attr = (line.rstrip("\n").split("\t") + [""] * 5)[:5]
        if lic not in KEEP:
            excluded[lic if lic not in ("", "\\N") else "no licence"] += 1
        elif aid in audio:
            excluded["audio listed twice"] += 1
        else:
            audio[aid] = (sid, user, lic, "" if attr == "\\N" else attr)
    log(f"{a.corpus}: {len(audio)} audio kept; excluded {dict(excluded)}")
    wanted = {sid for sid, *_ in audio.values()}
    text = {}
    for line in tsv_lines("sentences.tar.bz2"):
        sid, lang, txt = (line.rstrip("\n").split("\t", 2) + ["", ""])[:3]
        if sid in wanted:
            text[sid] = (lang, txt)
    log(f"{a.corpus}: text found for {len(text)} of {len(wanted)} sentences")

    rows, lic_rows, kept = [], [], Counter()
    path = a.out / f"{a.corpus}.text.tsv"
    with path.open("w", newline="") as o:
        w = csv.writer(o, delimiter="\t", lineterminator="\n")
        w.writerow(["relative_path", "audio_id", "sentence_id", "lang", "text"])
        for aid in sorted(audio, key=int):
            sid, user, lic, attr = audio[aid]
            rel = f"ARCHIVES/{aid}.mp3"
            rows.append((rel, f"https://tatoeba.org/en/audio/download/{aid}", "", ""))
            creator = f"{user} ({attr})" if attr else user
            page = f"https://tatoeba.org/en/sentences/show/{sid}"
            lic_rows.append((rel, *KEEP[lic], creator, page))
            lang, txt = text.get(sid, ("", ""))
            w.writerow([rel, aid, sid, lang, clean(txt)])
            kept[KEEP[lic][0]] += 1
    out = write_manifest(
        a.out,
        a.corpus,
        rows,
        [
            "Tatoeba audio (exports/sentences_with_audio.csv), audio licence "
            "CC BY 4.0 or CC0 1.0 only; per-file licence and contributor: "
            f"{a.corpus}.licences.tsv; sentence text (CC BY 2.0 FR): "
            f"{a.corpus}.text.tsv",
            f"{len(rows)} files from one host: download with --jobs 1",
        ],
    )
    write_licences(a.out, a.corpus, lic_rows)
    langs = Counter(text.get(audio[x][0], ("?",))[0] for x in audio)
    print(
        f"{a.corpus}: {len(rows)} files; kept by licence {dict(kept)}; excluded "
        f"{dict(excluded)}; languages {dict(langs.most_common(15))} -> {out}"
    )


if __name__ == "__main__":
    main()
