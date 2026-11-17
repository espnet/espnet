#!/usr/bin/env python3
"""Per-file licence manifests of the downloaded music corpora (read from their archives).

    python3 local/v2/music_manifests.py nsynth <corpus dir> <out.tsv>
    python3 local/v2/music_manifests.py musicnet <corpus dir> <out.tsv>
    python3 local/v2/music_manifests.py musopen <corpus dir> <out.tsv>

One row per file: corpus, file (archive member or path), licence, licence_url,
may_have_vocals, label, duration_s (when the metadata gives it). Singing is speech:
may_have_vocals=1 marks files that must never be used as non-speech.

- nsynth: CC BY 4.0 (magenta.tensorflow.org/datasets/nsynth); the notes of each split's
  examples.json; vocal = instrument_family_str "vocal"; 4 s per note.
- musicnet: CC BY 4.0 (Zenodo 5120004); the recordings of musicnet_metadata.csv; vocal =
  an ensemble or composition naming voice, choir or singing (none expected).
- musopen: the licence of each Internet Archive item (its metadata licenseurl); the FLAC
  files and the Kickstarter ZIP's members; vocal = a file name or title naming voice,
  choir, opera, mass, requiem, aria, song or lieder.
"""

import csv
import json
import re
import sys
import tarfile
import zipfile
from pathlib import Path

VOCAL_WORDS = re.compile(
    r"voice|vocal|choir|chor(al|us)|opera|\bmass\b|requiem|aria|\bsongs?\b|lied|"
    r"cantata|oratorio|soprano|tenor|alto\b|baritone|madrigal|motet|psalm|hymn",
    re.I,
)
HEAD = "corpus\tfile\tlicence\tlicence_url\tmay_have_vocals\tlabel\tduration_s\n"


def nsynth(d: Path, out):
    lic, url = "CC BY 4.0", "https://magenta.tensorflow.org/datasets/nsynth"
    n = v = 0
    for split in ("train", "valid", "test"):
        tgz = d / "ARCHIVES" / f"nsynth-{split}.jsonwav.tar.gz"
        ex = None
        with tarfile.open(tgz, "r|gz") as t:
            for m in t:
                if m.name.endswith("examples.json"):
                    ex = json.load(t.extractfile(m))
                    break
        if ex is None:
            raise SystemExit(f"{tgz}: no examples.json")
        for note, e in sorted(ex.items()):
            vocal = int(e.get("instrument_family_str") == "vocal")
            out.write(
                f"nsynth\tnsynth-{split}/audio/{note}.wav\t{lic}\t{url}\t{vocal}\t"
                f"{e.get('instrument_family_str', '')}\t4.0\n"
            )
            n += 1
            v += vocal
    print(f"nsynth: {n} notes, {v} vocal ({v * 4 / 3600:.1f} h)")


def musicnet(d: Path, out):
    lic, url = "CC BY 4.0", "https://zenodo.org/records/5120004"
    n = v = 0
    for r in csv.DictReader(open(d / "ARCHIVES" / "musicnet_metadata.csv")):
        text = f"{r.get('ensemble', '')} {r.get('composition', '')} {r.get('movement', '')}"
        vocal = int(bool(VOCAL_WORDS.search(text)))
        out.write(
            f"musicnet\t{r['id']}.wav\t{lic}\t{url}\t{vocal}\t{r.get('ensemble', '')}"
            f"\t{r.get('seconds', '')}\n"
        )
        n += 1
        v += vocal
    print(f"musicnet: {n} recordings, {v} with a vocal word")


def musopen(d: Path, out):
    n = v = 0
    for item in sorted(p for p in (d / "ARCHIVES").iterdir() if p.is_dir()):
        meta = json.load(open(item / "ia_metadata.json"))
        url = meta.get("metadata", {}).get("licenseurl", "")
        lic = (
            "CC0 1.0"
            if "zero" in url
            else "Public Domain Mark 1.0" if "publicdomain/mark" in url else url
        )
        title = {f["name"]: f.get("title", "") for f in meta.get("files", [])}
        for f in sorted(item.iterdir()):
            names = []
            if f.suffix.lower() == ".flac":
                names = [(f.name, title.get(f.name, ""))]
            elif f.suffix.lower() == ".zip":
                with zipfile.ZipFile(f) as z:
                    names = [
                        (f"{f.name}:{m}", "")
                        for m in z.namelist()
                        if m.lower().endswith((".flac", ".wav", ".mp3"))
                    ]
            for name, t in names:
                vocal = int(bool(VOCAL_WORDS.search(f"{name} {t}")))
                out.write(
                    f"musopen\t{item.name}/{name}\t{lic}\t{url}\t{vocal}\t{t}\t\n"
                )
                n += 1
                v += vocal
    print(f"musopen: {n} files, {v} with a vocal word")


def main():
    kind, d, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    with open(out, "w") as f:
        f.write(HEAD)
        {"nsynth": nsynth, "musicnet": musicnet, "musopen": musopen}[kind](d, f)


if __name__ == "__main__":
    main()
