#!/usr/bin/env python3
"""Write a download manifest and licence table from a Europeana dataset dump.

    python3 local/v2/europeana_dump_manifest.py <corpus> <dataset id> \
        [--keep-id REGEX] [--keep-text REGEX] [--zip FILE] \
        [--out local/v2/downloads]

Europeana publishes every dataset as a ZIP of EDM XML records at
download.europeana.eu/dataset/XML/<dataset id>.zip, without an API key. This
script reads that ZIP (--zip, or downloaded to a temporary directory and deleted
afterwards; the SLS dataset 2021007 is about 300 MB) and:

1. Takes the records that have audio (an edm:WebResource with dc:format audio/*).
2. Keeps a record when its record id matches --keep-id or its dc:description
   matches --keep-text (both optional; with neither, every audio record), and
   its edm:rights is CC0, PDM or CC BY (manifest_common.licence_accepted).
3. Takes the record's audio web resources as files; when a record has the
   same recording twice (WAV master and mp3, durations within 1 s), only the
   WAV.

Rows: corpus, ARCHIVES/<record id>/<last part of the URL>.<mp3|wav>, the web
resource URL, ebucore:fileByteSize (empty when 0); no checksum. The licence table
gives the licence id, the edm:rights URL, the dc:contributor names (informant and
recorder) and the Europeana item page. Hours come from ebucore:duration (ms).

SLS Finland-Swedish emigrant interviews (dataset 2021007, Svenska
litteratursällskapet i Finland): the 1971 America expedition (SLS 1859_1971) and
the interviews of returned emigrants in the other collections:

    python3 local/v2/europeana_dump_manifest.py sls_emigrant_interviews 2021007 \
        --keep-id '^_SLS_1859_1971_' \
        --keep-text 'Emigrerade|Vistades i (USA|Kanada|Amerika)'
"""

import argparse
import re
import sys
import tempfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manifest_common import (  # noqa: E402
    DOWNLOADS,
    UA,
    clean,
    licence_accepted,
    licence_id,
    log,
    write_licences,
    write_manifest,
)

DUMP = "https://download.europeana.eu/dataset/XML/{}.zip"
WEBRES = re.compile(
    r'<edm:WebResource rdf:about="([^"]+)">(.*?)</edm:WebResource>', re.S
)
EXT = {
    "audio/mpeg": "mp3",
    "audio/vnd.wave": "wav",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
}


def texts(xml, tag):
    """The text values of the elements tag in xml."""
    return [clean(v) for v in re.findall(rf"<{tag}(?:\s[^>]*)?>([^<]*)</{tag}>", xml)]


def audio_files(xml):
    """(url, extension, bytes, ms) of the audio web resources of a record.

    A record that has the same recording twice (durations within one second,
    e.g. a WAV master and its mp3) keeps only the WAV.
    """
    found = []
    for url, body in WEBRES.findall(xml):
        mime = (texts(body, "dc:format") or [""])[0]
        if not mime.startswith("audio/"):
            continue
        ext = EXT.get(mime) or (texts(body, "edm:codecName") or ["bin"])[0]
        size = (texts(body, "ebucore:fileByteSize") or ["0"])[0]
        ms = int((texts(body, "ebucore:duration") or ["0"])[0] or 0)
        found.append((url, ext, size if size != "0" else "", ms))
    found.sort(key=lambda f: f[1] != "wav")  # WAV first
    out = []
    for f in found:
        if not any(abs(f[3] - g[3]) < 1000 and f[3] for g in out):
            out.append(f)
    return out


def fetch_zip(dataset, into: Path) -> Path:
    """Download the XML dump of dataset into the directory into."""
    path = into / f"{dataset}.zip"
    req = urllib.request.Request(DUMP.format(dataset), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=600) as r, path.open("wb") as o:
        while block := r.read(1 << 20):
            o.write(block)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("corpus")
    ap.add_argument("dataset", help="Europeana dataset id, e.g. 2021007")
    ap.add_argument("--keep-id", default="", help="keep record ids matching this")
    ap.add_argument("--keep-text", default="", help="keep descriptions matching this")
    ap.add_argument("--zip", type=Path, default=None, help="dump already downloaded")
    ap.add_argument("--out", type=Path, default=DOWNLOADS)
    a = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        path = a.zip or fetch_zip(a.dataset, Path(tmp))
        rows, lic_rows, kept, excluded, secs, nbytes = (
            [],
            [],
            Counter(),
            Counter(),
            0,
            0,
        )
        with zipfile.ZipFile(path) as z:
            for name in sorted(z.namelist()):
                xml = z.read(name).decode("utf-8", "replace")
                files = audio_files(xml)
                if not files:
                    continue
                rec = Path(name).stem
                desc = " ".join(texts(xml, "dc:description"))
                chosen = not (a.keep_id or a.keep_text)
                chosen |= bool(a.keep_id and re.search(a.keep_id, rec))
                chosen |= bool(a.keep_text and re.search(a.keep_text, desc))
                if not chosen:
                    excluded["audio record not selected"] += 1
                    continue
                rights = re.findall(r'<edm:rights rdf:resource="([^"]+)"', xml)
                lid = licence_id(" ".join(sorted(set(rights))))
                if not licence_accepted(lid):
                    excluded[f"licence {rights or 'none'}"] += 1
                    continue
                kept[lid] += 1
                for url, ext, size, ms in files:
                    stem = url.rstrip("/").rsplit("/", 1)[-1]
                    rel = f"ARCHIVES/{rec}/{stem}.{ext}"
                    rows.append((rel, url, size, ""))
                    lic_rows.append(
                        (
                            rel,
                            lid,
                            rights[0],
                            "; ".join(texts(xml, "dc:contributor")),
                            f"https://www.europeana.eu/item/{a.dataset}/{rec}",
                        )
                    )
                    secs += ms / 1000
                    nbytes += int(size or 0)
    rows.sort()
    lic_rows.sort()
    out = write_manifest(
        a.out,
        a.corpus,
        rows,
        [
            f"Europeana dataset {a.dataset} dump ({DUMP.format(a.dataset)}), audio "
            f"records with id ~ {a.keep_id!r} or description ~ {a.keep_text!r}, "
            f"edm:rights CC0/PDM/CC BY only; per-file licence: {a.corpus}.licences.tsv",
            f"{len(rows)} files, {secs / 3600:.1f} h",
        ],
    )
    write_licences(a.out, a.corpus, lic_rows)
    log(f"{a.corpus}: records kept by licence {dict(kept)}; excluded {dict(excluded)}")
    print(
        f"{a.corpus}: {len(rows)} files, {nbytes / 1e9:.2f} GB, {secs / 3600:.1f} h "
        f"-> {out}"
    )


if __name__ == "__main__":
    main()
