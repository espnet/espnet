#!/usr/bin/env python3
"""Per-file licence manifest of the candidate noise corpora (read from their archives).

    python3 local/v2/noise_manifest.py ${VAD_CORPORA} <out.tsv> \
        [--sources sonyc_ust,fsdnoisy18k,fsd50k,esc50,clotho]

One row per audio file: source, file (archive member), licence, class
(adapters_nonspeech.licence_class), speech flag, and the label the flag comes
from. Nothing is extracted and no list is changed: the manifest is what the
noise lists are later counted and filtered by.

- sonyc_ust: CC BY 4.0 (README); speech = any annotator marked a human-voice
  presence column (talking, shouting, crowd, amplified speech) in
  annotations.csv.
- fsdnoisy18k: per clip (FSDnoisy18k.doc/LICENSE-INDIVIDUAL-CLIPS.csv);
  speech = never (20 non-speech classes), the label is recorded.
- fsd50k: per clip (FSD50K.metadata/*_clips_info_FSD50K.json); speech =
  adapters_nonspeech.fsd_kind() is None (speech, singing, crowds, broadcast) or
  "vocal" (human non-speech vocal sounds) is recorded as such.
- esc50: per clip, the source licence named in its LICENSE ("[CC0]", "[CC-BY]");
  the ESC-50 compilation itself is CC BY-NC (ESC-10 CC BY), recorded in the
  licence field; speech = never (no speech classes; crying, laughing, coughing,
  breathing, snoring are vocal non-speech).
- clotho: per file (metadata license link); speech = the file name or keywords
  name a voice (adapters_nonspeech.VOICE_RE).
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adapters_nonspeech as ans  # noqa: E402

VOICE_COL = re.compile(r"human-voice|talking|shouting|crowd|amplified-speech", re.I)


def sonyc(root):
    ann = ans.archive(root / "sonyc_ust", "annotations.csv")
    rows = list(csv.DictReader(open(ann)))
    cols = [c for c in rows[0] if c.endswith("_presence") and VOICE_COL.search(c)]
    voice = {}
    for r in rows:
        hit = [c for c in cols if (r.get(c) or "").strip() in ("1", "1.0")]
        f = r["audio_filename"]
        voice.setdefault(f, set()).update(hit)
    for f, hit in sorted(voice.items()):
        yield ("sonyc_ust", f, "CC BY 4.0", "BY", bool(hit), ";".join(sorted(hit)))


def fsdnoisy18k(root):
    doc = ans.archive(root / "fsdnoisy18k", "FSDnoisy18k.doc.zip")
    meta = ans.archive(root / "fsdnoisy18k", "FSDnoisy18k.meta.zip")
    lic_rows = ans.read_csv(
        ans.read_zip_member(doc, "FSDnoisy18k.doc/LICENSE-INDIVIDUAL-CLIPS.csv")
    )
    lc = ans.find_column(lic_rows[0], "licen")
    ic = ans.find_column(lic_rows[0], "fname", "file", "id")
    lic = {r[ic].rsplit(".", 1)[0]: r[lc] for r in lic_rows}
    for part in ("train", "test"):
        for r in ans.read_csv(
            ans.read_zip_member(meta, f"FSDnoisy18k.meta/{part}.csv")
        ):
            stem = r["fname"].rsplit(".", 1)[0]
            text = lic.get(stem, "")
            yield (
                "fsdnoisy18k",
                f"FSDnoisy18k.audio_{part}/{r['fname']}",
                text,
                ans.licence_class(text),
                False,
                r["label"],
            )


def fsd50k(root):
    md = ans.archive(root / "fsd50k", "FSD50K.metadata.zip")
    gt = ans.archive(root / "fsd50k", "FSD50K.ground_truth.zip")
    for part in ("dev", "eval"):
        info = json.loads(
            ans.read_zip_member(md, f"FSD50K.metadata/{part}_clips_info_FSD50K.json")
        )
        for r in ans.read_csv(
            ans.read_zip_member(gt, f"FSD50K.ground_truth/{part}.csv")
        ):
            meta = info.get(r["fname"], {})
            labels = r["labels"].split(",")
            tags = " ".join([meta.get("title", "")] + list(meta.get("tags", [])))
            kind = ans.fsd_kind(labels, tags)
            text = meta.get("license", "")
            yield (
                "fsd50k",
                f"FSD50K.{part}_audio/{r['fname']}.wav",
                text,
                ans.licence_class(text),
                kind is None,
                f"{kind or 'speech/singing/crowd'}:{r['labels']}",
            )


ESC_LINE = re.compile(r"^- \[([^\]]+\.ogg)\]: .*\[([^\]]+)\]\s*$")


def esc50(root):
    text = ans.archive(root / "esc50", "LICENSE").read_text(errors="replace")
    for line in text.splitlines():
        m = ESC_LINE.match(line.strip())
        if not m:
            continue
        src = m.group(2)
        lic = f"source {src}; ESC-50 compilation CC BY-NC 3.0 (ESC-10: CC BY 3.0)"
        yield ("esc50", f"audio/{m.group(1)}", lic, ans.licence_class(src), False, "")


def clotho(root):
    for part in ("development", "validation", "evaluation"):
        path = ans.archive(root / "clotho", f"clotho_metadata_{part}.csv")
        for r in ans.read_csv(path.read_bytes()):
            name = r.get("file_name", "")
            text = f"{name} {r.get('keywords', '')}".replace(";", " ").replace("_", " ")
            lic = r.get("license", "")
            yield (
                "clotho",
                f"{part}/{name}",
                lic,
                ans.licence_class(lic),
                bool(ans.VOICE_RE.search(text)),
                r.get("keywords", "")[:200],
            )


SOURCES = {
    "sonyc_ust": sonyc,
    "fsdnoisy18k": fsdnoisy18k,
    "fsd50k": fsd50k,
    "esc50": esc50,
    "clotho": clotho,
}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--sources", default=",".join(SOURCES))
    a = p.parse_args()
    counts = {}
    with a.out.open("w") as f:
        f.write("source\tfile\tlicence\tclass\tspeech\tlabel\n")
        for name in a.sources.split(","):
            for src, file, lic, cls, speech, label in SOURCES[name](a.root):
                f.write(f"{src}\t{file}\t{lic}\t{cls}\t{int(speech)}\t{label}\n")
                k = (src, cls, int(speech))
                counts[k] = counts.get(k, 0) + 1
    for (src, cls, speech), n in sorted(counts.items()):
        print(f"{src:12s} {cls:10s} speech={speech} {n:7d}")


if __name__ == "__main__":
    main()
