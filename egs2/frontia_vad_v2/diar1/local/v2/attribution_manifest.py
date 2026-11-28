#!/usr/bin/env python3
"""Per-file attribution manifest of a trained model's data, from its final lists.

    python3 local/v2/attribution_manifest.py ${VAD_CORPORA} <dump dir> \\
        <noise_pool.tsv> <music_pool.tsv> <rir.scp> <opennoise lists dir> <out.tsv.gz>

Writes one row per training file, gzipped TSV, with the columns
    kind     speech, noise, music or rir
    source   corpus or source name
    record   the training record (dump rec_id, noise or music pack id, RIR id)
    item     the file within the record (archive member, clip or piece)
    licence  the file's own licence where its source gives one per file,
             otherwise the corpus licence
    author, title, url
             the file's creator, title and page, where the source gives them
             (Freesound uploads, MUSAN, OpenNoise pieces); empty otherwise
    seconds  the item's share of the record's duration

Speech: each record of <dump dir>/manifest.tsv, split into the clips its source
column names. FSD50K, BSD10k, BSD35k-CS, CrowdioSet, FSDnoisy18k, Designed
Vocalizations and OpenMIC-2018 clips get their licence and credit from the
corpus metadata; FUSS mixtures are listed by their FSD50K source clips; MUSAN
files get the licence and URL of their LICENSE block. Every other corpus gets its
corpus licence (licence_audit.CORPUS_CLASS where it corrects the dump's string).

Noise, music and RIR: the rows of the OpenNoise train manifests
(<kind>_train_manifest.tsv) whose entry is a pack of <noise_pool.tsv> or
<music_pool.tsv>, or an id of <rir.scp>.

It prints the number of records and rows of each kind, which must equal the
training lists, and, per source, the rows whose licence requires credit (BY) but
which have no author.
"""

import argparse
import csv
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adapters_nonspeech as ans  # noqa: E402
import licence_audit as la  # noqa: E402
from sources import iter_parquet  # noqa: E402

COLUMNS = (
    "kind",
    "source",
    "record",
    "item",
    "licence",
    "author",
    "title",
    "url",
    "seconds",
)


# Licences the OpenNoise manifests write as SPDX LicenseRef ids, and the
# generated audio: classes as licence_audit.OK names them. Open-AoE and the
# UIowa MIS terms were accepted for OpenNoise (both allow commercial use and
# derivatives).
LICENCE_REF_CLASS = {
    "licenseref-public-domain": "PD/CC0",
    "licenseref-pdm": "PD/CC0",
    "licenseref-public-domain-mark-1.0": "PD/CC0",
    "licenseref-public-domain-pdsounds": "PD/CC0",
    "licenseref-us-public-domain": "PD/CC0",
    "licenseref-open-aoe-dataset-license": "permissive",
    "licenseref-uiowa-mis-free-use": "permissive",
    "generated": "generated",
}


def licence_class(text: str) -> str:
    """Return the licence class of a manifest row's licence."""
    return LICENCE_REF_CLASS.get(text.strip().lower()) or la.classify(text)


def freesound_url(sound_id: str) -> str:
    """Return the Freesound page of a sound id, or "" for an id that is not one."""
    sound_id = str(sound_id).strip()
    return f"https://freesound.org/s/{sound_id}/" if sound_id.isdigit() else ""


def freesound_id_of(url: str) -> str:
    """Return the sound id in a Freesound page or API URL, or ""."""
    m = re.search(
        r"freesound\.org/(?:apiv2/)?(?:s|sounds|people/[^/]+/sounds)/(\d+)", url
    )
    return m.group(1) if m else ""


class Credits(la.Lookups):
    """Per-clip licence and credit tables, loaded on first use."""

    def fsd50k_credits(self):
        def load():
            md = ans.archive(self.root / "fsd50k", "FSD50K.metadata.zip")
            out = {}
            for part in ("dev", "eval"):
                info = json.loads(
                    ans.read_zip_member(
                        md, f"FSD50K.metadata/{part}_clips_info_FSD50K.json"
                    )
                )
                for fname, m in info.items():
                    out[(part, fname)] = (
                        m.get("license", ""),
                        m.get("uploader", ""),
                        m.get("title", ""),
                    )
            return out

        return self.get("fsd50k_credits", load)

    def bsd_credits(self, corpus_dir: str):
        def load():
            md = ans.archive(self.root / corpus_dir, "metadata.zip")
            name = next(n for n in ans.zip_names(md) if n.endswith("_metadata.csv"))
            return {
                r["sound_id"]: (
                    r.get("license", ""),
                    r.get("uploader", ""),
                    r.get("title", ""),
                )
                for r in ans.read_csv(ans.read_zip_member(md, name))
            }

        return self.get(("bsd_credits", corpus_dir), load)

    def crowdioset_credits(self):
        def load():
            rows = ans.read_csv(
                (self.root / "crowdioset/crowdioset_metadata.csv").read_bytes()
            )
            return {
                r["final_name"]: (
                    r["license"],
                    r.get("username", ""),
                    r.get("original_name", ""),
                    freesound_url(freesound_id_of(r.get("url", ""))),
                )
                for r in rows
            }

        return self.get("crowdioset_credits", load)

    def designed_vocalizations_credits(self):
        def load():
            base = self.root / "designed_vocalizations" / "data"
            cols = [
                "file_path",
                "license",
                "freesound_uploader",
                "freesound_title",
                "freesound_url",
            ]
            raw = {}
            for pq in sorted((base / "raw").glob("*.parquet")):
                for r in iter_parquet(pq, cols):
                    raw[r["file_path"]] = (
                        r.get("license") or "",
                        r.get("freesound_uploader") or "",
                        r.get("freesound_title") or "",
                        r.get("freesound_url") or "",
                    )
            src = {}
            for pq in sorted((base / "designed").glob("*.parquet")):
                for r in iter_parquet(pq, ["file_path", "source"]):
                    src[r["file_path"]] = r.get("source") or ""
            return raw, src

        return self.get("designed_vocalizations_credits", load)

    def openmic_credits(self):
        def load():
            path = ans.archive(self.root / "openmic2018", "openmic-2018-v1.0.0.tgz")
            for name, data in ans.iter_tar_select(
                path,
                lambda n: n.endswith("openmic-2018-metadata.csv")
                and not ans.is_junk(n),
            ):
                rows = ans.read_csv(data)
                lic = ans.find_column(rows[0], "licen")
                return {
                    r["sample_key"]: (
                        r[lic],
                        r.get("artist_name", ""),
                        r.get("track_title", ""),
                        r.get("track_url", "") or r.get("artist_url", ""),
                    )
                    for r in rows
                }
            return {}

        return self.get("openmic_credits", load)


def speech_rows(cr: Credits, manifest: Path):
    """Yield the manifest row of every clip of every speech record."""
    rows = list(csv.DictReader(open(manifest), delimiter="\t"))
    fuss = [m for r in rows if r["corpus"] == "fuss" for m in r["source"].split(";")]
    fuss_src = cr.fuss_sources(set(fuss)) if fuss else {}
    for r in rows:
        corpus = r["corpus"]
        members = [m for m in r["source"].split(";") if m] or [r["rec_id"]]
        seconds = float(r["seconds"]) / len(members)
        for item, lic, author, title, url, sec in clips(
            cr, r, corpus, members, seconds, fuss_src
        ):
            yield ("speech", corpus, r["rec_id"], item, lic, author, title, url, sec)


def clips(cr, r, corpus, members, seconds, fuss_src):
    """[(item, licence, author, title, url, seconds)] of one speech record."""
    out = []
    if corpus.startswith("fsd50k"):
        t = cr.fsd50k_credits()
        for m in members:
            mm = re.match(r"FSD50K\.(dev|eval)_audio/(\d+)\.wav$", m)
            lic, author, title = (
                t.get((mm.group(1), mm.group(2)), ("", "", "")) if mm else ("", "", "")
            )
            out.append(
                (
                    m,
                    lic,
                    author,
                    title,
                    freesound_url(mm.group(2)) if mm else "",
                    seconds,
                )
            )
    elif corpus == "fuss":
        t = cr.fsd50k_credits()
        by_fname = {}
        for (part, fname), v in t.items():
            by_fname.setdefault(fname, v)
        for m in members:
            ids = fuss_src.get(m) or []
            share = seconds / max(1, len(ids))
            for i in ids:
                lic, author, title = by_fname.get(i, ("", "", ""))
                out.append((f"{m}#{i}", lic, author, title, freesound_url(i), share))
            if not ids:
                out.append((m, "", "", "", "", seconds))
    elif corpus.startswith(("bsd10k", "bsd35k_cs")):
        t = cr.bsd_credits("bsd10k" if corpus.startswith("bsd10k") else "bsd35k_cs")
        for m in members:
            sid = Path(m).stem
            lic, author, title = t.get(sid, ("", "", ""))
            out.append((m, lic, author, title, freesound_url(sid), seconds))
    elif corpus == "crowdioset":
        t = cr.crowdioset_credits()
        for m in members:
            lic, author, title, url = t.get(Path(m).name, ("", "", "", ""))
            out.append((Path(m).name, lic, author, title, url, seconds))
    elif corpus == "fsdnoisy18k":
        t = cr.fsdnoisy18k()
        for m in members:
            sid = Path(m).stem
            # The corpus gives each clip's licence, not its uploader.
            out.append((m, t.get(sid, ""), "", "", freesound_url(sid), seconds))
    elif corpus == "designed_vocalizations":
        raw, src = cr.designed_vocalizations_credits()
        for m in members:
            fp = m.split(":", 1)[1] if ":" in m else m
            lic, author, title, url = (
                raw.get(fp) or raw.get(src.get(fp, ""), None) or ("", "", "", "")
            )
            out.append((fp, lic, author, title, url, seconds))
    elif corpus == "openmic2018":
        t = cr.openmic_credits()
        for m in members:
            lic, author, title, url = t.get(Path(m).stem, ("", "", "", ""))
            out.append((Path(m).stem, lic, author, title, url, seconds))
    elif corpus in ("musan", "musan_noise"):
        for m in members:
            lic, url = musan_licence_and_url(m)
            out.append((m, lic, "", "", url, seconds))
    else:
        lic = [
            c for c in la.clip_licences(cr, corpus, r["lang"], r["licence"], members)
        ]
        for item, licence, _ in lic:
            out.append((item, licence, "", "", "", seconds))
    return out


_MUSAN = {}


def musan_licence_and_url(path: str):
    """Return (licence, URL) of a MUSAN file from its directory's LICENSE."""
    p = Path(path)
    if p.parent not in _MUSAN:
        stems = [q.stem for q in p.parent.glob("*.wav")]
        _MUSAN[p.parent] = ans.musan_licence_table(p.parent / "LICENSE", stems)
    return _MUSAN[p.parent].get(p.stem, ("", ""))


def augmentation_rows(kind: str, keep: set, manifest: Path):
    """Yield the OpenNoise manifest rows of the kept packs or ids."""
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["entry"] not in keep:
                continue
            seconds = (
                float(r["end_s"]) - float(r["start_s"])
                if r.get("end_s")
                else float(r.get("duration_s") or 0)
            )
            yield (
                kind,
                r["source"],
                r["entry"],
                r["segment"] or r["entry"],
                r["spdx"] or r["licence"],
                r["author"],
                r["title"],
                r["orig_url"],
                seconds,
            )


def pool_ids(path: Path) -> set:
    with open(path, newline="") as f:
        return {r["id"] for r in csv.DictReader(f, delimiter="\t")}


def clean(field) -> str:
    """Return a field as one TSV cell: tabs and newlines become spaces."""
    return re.sub(r"\s+", " ", str(field)).strip()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("root", help="the corpora root (VAD_CORPORA)")
    p.add_argument("dump", help="the training dump directory (with manifest.tsv)")
    p.add_argument("noise_pool")
    p.add_argument("music_pool")
    p.add_argument("rir_scp")
    p.add_argument("opennoise_lists", help="opennoise data/lists directory")
    p.add_argument("out", help="output .tsv.gz")
    a = p.parse_args(argv)

    cr = Credits(Path(a.root))
    lists = Path(a.opennoise_lists)
    rir_ids = {line.split()[0] for line in open(a.rir_scp) if line.strip()}
    sources = (
        ("speech", speech_rows(cr, Path(a.dump) / "manifest.tsv")),
        (
            "noise",
            augmentation_rows(
                "noise",
                pool_ids(Path(a.noise_pool)),
                lists / "noise_train_manifest.tsv",
            ),
        ),
        (
            "music",
            augmentation_rows(
                "music",
                pool_ids(Path(a.music_pool)),
                lists / "music_train_manifest.tsv",
            ),
        ),
        ("rir", augmentation_rows("rir", rir_ids, lists / "rir_train_manifest.tsv")),
    )
    records = defaultdict(set)
    rows = Counter()
    hours = Counter()
    uncredited = Counter()
    classes = Counter()
    with gzip.open(a.out, "wt", newline="") as fo:
        fo.write("\t".join(COLUMNS) + "\n")
        for kind, it in sources:
            for row in it:
                fo.write(
                    "\t".join(
                        clean(x) if i < 8 else f"{x:.3f}" for i, x in enumerate(row)
                    )
                    + "\n"
                )
                records[kind].add(row[2])
                rows[kind] += 1
                hours[kind] += row[8] / 3600
                c = licence_class(row[4])
                classes[(kind, c)] += 1
                if c == "BY" and not row[5].strip():
                    uncredited[(kind, row[1])] += 1
    for kind in ("speech", "noise", "music", "rir"):
        print(
            f"{kind}: {len(records[kind])} records, {rows[kind]} rows, "
            f"{hours[kind]:.1f} h"
        )
    print("licence classes (rows):")
    for (kind, c), n in sorted(classes.items()):
        print(f"  {kind:7s} {c:12s} {n}")
    print("rows under a BY licence with no author (credit by the source's page):")
    for (kind, src), n in uncredited.most_common():
        print(f"  {kind:7s} {src:30s} {n}")


if __name__ == "__main__":
    main()
