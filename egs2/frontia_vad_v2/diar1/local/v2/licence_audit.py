#!/usr/bin/env python3
"""Licence audit of the training data by its output: every clip of a dump and every
entry of the augmentation lists, classified by its own licence.

    python3 local/v2/licence_audit.py dump ${VAD_CORPORA} <dump dir> <out.tsv>
    python3 local/v2/licence_audit.py lists ${VAD_CORPORA} <scp> [<scp> ...]
    python3 local/v2/licence_audit.py musan ${VAD_CORPORA}/musan

dump: each record of <dump dir>/manifest.tsv (a pack names its clips in the
source column, ";"-separated) is resolved to the licence of each of its clips:
MUSAN per file (its LICENSE block; music also needs the ANNOTATIONS vocals
flag N), FSD50K, BSD10k, BSD35k-CS, CrowdioSet, FSDnoisy18k, Designed
Vocalizations, OpenMIC-2018 per clip from their metadata, FUSS per source clip
(FSD50K licence of every source named in the mixture's .jams), WAXAL per
language; every other corpus by its corpus licence. Writes one row per clip
(corpus, record, clip, licence, class, hours) and prints the table of corpus x
class (records, clips, hours).

lists: the noise, music and impulse-response lists (data/noise.scp,
data/music.scp, data/noise_aug.scp, data/rir.scp): MUSAN files per file (music
without vocals only), CHiME-3 backgrounds, the BUT ReverbDB impulse responses
(data/rir/), the packs of data/aug_banks (every clip of licences.tsv).

musan: per directory, the number of files of each class and of music with
vocals or without a vocals annotation.

Accepted classes: PD/CC0, BY, permissive (Apache-2.0, MIT, BSD, ODC-BY,
CDLA-Permissive), DUA (a data use agreement that allows training: OpenYAP),
generated. Anything else (SA, ND, NC, Sampling+, unknown, unresolved) is a
failure: the command exits with status 1 and lists it.
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adapters_nonspeech as ans  # noqa: E402
from sources import iter_parquet  # noqa: E402

OK = {"PD/CC0", "BY", "permissive", "DUA", "generated"}

# Corpus licences that licence_class() does not name, and corrections of the
# licence strings recorded in older dumps.
CORPUS_CLASS = {
    # CHiME-5/6 re-licensed CC BY 4.0 (Hugging Face chimechallenge/chime6,
    # LICENSE.txt); dumps made before record "CC BY-SA 4.0 (SA to be removed)"
    "chime6": ("CC BY 4.0 (chimechallenge/chime6 LICENSE.txt)", "BY"),
    "slakh2100": ("CC BY 4.0 (Zenodo 4599666)", "BY"),
    "openyap": ("OpenYAP DUA (training OK)", "DUA"),
    "synth_lownoise": ("generated", "generated"),
}
PERMISSIVE_RE = re.compile(r"apache|\bmit\b|\bbsd\b|odc-by|cdla-permissive", re.I)
# WAXAL folders of the University of Ghana parts, CC BY 4.0 (dataset card: ASR
# Akan, Ewe, Dagbani, Dagaare, Ikposo; TTS Akan (Fante, Twi), Baoule, Ewe); every
# other provider's part is CC BY-SA 4.0
WAXAL_BY = {"aka", "ewe", "dag", "dga", "kpo", "twi", "fat", "bau"}


def classify(text: str) -> str:
    c = ans.licence_class(text)
    if c == "unknown" and PERMISSIVE_RE.search(text or ""):
        return "permissive"
    return c


# ---------------------------------------------------------------- per-clip lookups


class Lookups:
    """Per-clip licence tables of the corpora, loaded on first use."""

    def __init__(self, root: Path):
        self.root = root
        self.cache = {}

    def get(self, name, load):
        if name not in self.cache:
            self.cache[name] = load()
        return self.cache[name]

    def musan(self, path: str):
        p = Path(path)

        def load():
            stems = [q.stem for q in p.parent.glob("*.wav")]
            table = ans.musan_licence_table(p.parent / "LICENSE", stems)
            vocals = {}
            ann = p.parent / "ANNOTATIONS"
            if ann.exists():
                for line in ann.read_text(errors="replace").splitlines():
                    f = line.split()
                    if len(f) >= 3 and f[2] in ("Y", "N"):
                        vocals[f[0].rsplit(".", 1)[0]] = f[2] == "Y"
            return table, vocals

        table, vocals = self.get(("musan", str(p.parent)), load)
        lic, _ = table.get(p.stem, ("", ""))
        if "/music/" in str(p) and vocals.get(p.stem, True):
            return f"{lic} [vocals or not annotated]", "vocals"
        return lic, classify(lic)

    def fsd50k(self):
        def load():
            md = ans.archive(self.root / "fsd50k", "FSD50K.metadata.zip")
            out = {}
            for part in ("dev", "eval"):
                info = json.loads(
                    ans.read_zip_member(
                        md, f"FSD50K.metadata/{part}_clips_info_FSD50K.json"
                    )
                )
                for fname, meta in info.items():
                    out[(part, fname)] = meta.get("license", "")
            return out

        return self.get("fsd50k", load)

    def bsd(self, corpus_dir: str):
        def load():
            md = ans.archive(self.root / corpus_dir, "metadata.zip")
            name = next(n for n in ans.zip_names(md) if n.endswith("_metadata.csv"))
            return {
                r["sound_id"]: r.get("license", "")
                for r in ans.read_csv(ans.read_zip_member(md, name))
            }

        return self.get(("bsd", corpus_dir), load)

    def crowdioset(self):
        def load():
            rows = ans.read_csv(
                (self.root / "crowdioset/crowdioset_metadata.csv").read_bytes()
            )
            return {r["final_name"]: r["license"] for r in rows}

        return self.get("crowdioset", load)

    def fsdnoisy18k(self):
        def load():
            doc = ans.archive(self.root / "fsdnoisy18k", "FSDnoisy18k.doc.zip")
            rows = ans.read_csv(
                ans.read_zip_member(doc, "FSDnoisy18k.doc/LICENSE-INDIVIDUAL-CLIPS.csv")
            )
            lic = ans.find_column(rows[0], "licen")
            idc = ans.find_column(rows[0], "fname", "file", "id")
            return {r[idc].rsplit(".", 1)[0]: r[lic] for r in rows}

        return self.get("fsdnoisy18k", load)

    def designed_vocalizations(self):
        def load():
            base = self.root / "designed_vocalizations" / "data"
            raw = {}
            for pq in sorted((base / "raw").glob("*.parquet")):
                for r in iter_parquet(pq, ["file_path", "license"]):
                    raw[r["file_path"]] = r.get("license") or ""
            src = {}
            for pq in sorted((base / "designed").glob("*.parquet")):
                for r in iter_parquet(pq, ["file_path", "source"]):
                    src[r["file_path"]] = r.get("source") or ""
            return raw, src

        return self.get("designed_vocalizations", load)

    def openmic(self):
        def load():
            path = ans.archive(self.root / "openmic2018", "openmic-2018-v1.0.0.tgz")
            for name, data in ans.iter_tar_select(
                path,
                lambda n: n.endswith("openmic-2018-metadata.csv")
                and not ans.is_junk(n),
            ):
                rows = ans.read_csv(data)
                lic = ans.find_column(rows[0], "licen")
                return {r["sample_key"]: r[lic] for r in rows}
            return {}

        return self.get("openmic", load)

    def fuss_sources(self, wanted):
        """{member: [FSD50K fname, ...]} of the FUSS mixtures in `wanted`."""

        def load():
            path = ans.archive(self.root / "fuss", "FUSS_ssdata_reverb.tar.gz")
            jams = {m[:-4] + ".jams" for m in wanted}
            out = {}
            for name, data in ans.iter_tar_select(path, lambda n: n in jams):
                files = re.findall(r'"source_file"\s*:\s*"([^"]+)"', data.decode())
                out[name[:-5] + ".wav"] = [
                    f.rsplit("/", 1)[-1].rsplit(".", 1)[0] for f in files
                ]
            return out

        return self.get("fuss", load)


def clip_licences(lk: Lookups, corpus: str, lang: str, corpus_licence: str, members):
    """[(clip, licence, class)] of one record."""
    out = []
    if corpus in ("musan", "musan_noise"):
        for m in members:
            lic, c = lk.musan(m)
            out.append((Path(m).stem, lic, c))
    elif corpus.startswith("fsd50k"):
        t = lk.fsd50k()
        for m in members:
            mm = re.match(r"FSD50K\.(dev|eval)_audio/(\d+)\.wav$", m)
            lic = t.get((mm.group(1), mm.group(2)), "") if mm else ""
            out.append((m, lic, classify(lic)))
    elif corpus.startswith(("bsd10k", "bsd35k_cs")):
        t = lk.bsd("bsd10k" if corpus.startswith("bsd10k") else "bsd35k_cs")
        for m in members:
            lic = t.get(Path(m).stem, "")
            out.append((m, lic, classify(lic)))
    elif corpus == "crowdioset":
        t = lk.crowdioset()
        for m in members:
            lic = t.get(Path(m).name, "")
            out.append((Path(m).name, lic, classify(lic)))
    elif corpus == "fsdnoisy18k":
        t = lk.fsdnoisy18k()
        for m in members:
            lic = t.get(Path(m).stem, "")
            out.append((m, lic, classify(lic)))
    elif corpus == "designed_vocalizations":
        raw, src = lk.designed_vocalizations()
        for m in members:
            fp = m.split(":", 1)[1] if ":" in m else m
            lic = raw.get(fp) if fp in raw else raw.get(src.get(fp, ""), "")
            out.append((fp, lic or "", classify(lic or "")))
    elif corpus == "openmic2018":
        t = lk.openmic()
        for m in members:
            lic = t.get(Path(m).stem, "")
            out.append((Path(m).stem, lic, classify(lic)))
    elif corpus == "waxal":
        ok = lang in WAXAL_BY
        lic = f"WAXAL {lang}: " + ("CC BY 4.0" if ok else "CC BY-SA 4.0")
        out = [(m, lic, "BY" if ok else "SA") for m in members]
    elif corpus in CORPUS_CLASS:
        lic, c = CORPUS_CLASS[corpus]
        out = [(m, lic, c) for m in members]
    else:
        out = [(m, corpus_licence, classify(corpus_licence)) for m in members]
    return out


def cmd_dump(a):
    lk = Lookups(Path(a.root))
    rows = list(csv.DictReader(open(Path(a.dump) / "manifest.tsv"), delimiter="\t"))
    fuss = [m for r in rows if r["corpus"] == "fuss" for m in r["source"].split(";")]
    if fuss:
        srcs = lk.fuss_sources(set(fuss))
        fsd = lk.fsd50k()
        by_fname = {}
        for (part, fname), lic in fsd.items():
            by_fname.setdefault(fname, lic)
    table = defaultdict(lambda: [set(), 0, 0.0])
    bad = []
    with open(a.out, "w") as fo:
        fo.write("corpus\trecord\tclip\tlicence\tclass\thours\n")
        for r in rows:
            corpus = r["corpus"]
            members = [m for m in r["source"].split(";") if m] or [r["rec_id"]]
            hours = float(r["seconds"]) / 3600 / len(members)
            if corpus == "fuss":
                clips = []
                for m in members:
                    ids = srcs.get(m)
                    if not ids:
                        clips.append((m, "no .jams sources", "unknown"))
                        continue
                    lics = [by_fname.get(i, "") for i in ids]
                    worst = next(
                        (lic for lic in lics if classify(lic) not in OK), lics[0]
                    )
                    clips.append((m, worst, classify(worst)))
            else:
                clips = clip_licences(lk, corpus, r["lang"], r["licence"], members)
            for clip, lic, c in clips:
                t = table[(corpus, c)]
                t[0].add(r["rec_id"])
                t[1] += 1
                t[2] += hours
                fo.write(f"{corpus}\t{r['rec_id']}\t{clip}\t{lic}\t{c}\t{hours:.6f}\n")
                if c not in OK:
                    bad.append((corpus, r["rec_id"], clip, lic, c, hours))
    print(f"== {a.dump}: {len(rows)} records")
    for (corpus, c), (recs, n, h) in sorted(table.items()):
        flag = "" if c in OK else "   <-- FAIL"
        print(
            f"  {corpus:24s} {c:11s} {len(recs):7d} rec {n:8d} clips {h:9.2f} h{flag}"
        )
    report_failures(bad)


def report_failures(bad):
    if not bad:
        print("  all clips: accepted licences")
        return
    agg = defaultdict(lambda: [0, 0.0])
    for corpus, _, _, lic, c, h in bad:
        agg[(corpus, c, lic[:120])][0] += 1
        agg[(corpus, c, lic[:120])][1] += h
    print(f"  FAILURES: {len(bad)} clips, {sum(b[-1] for b in bad):.2f} h")
    for (corpus, c, lic), (n, h) in sorted(agg.items()):
        print(f"    {corpus:22s} {c:10s} {n:6d} {h:7.2f} h  {lic}")
    sys.exit(1)


def cmd_lists(a):
    lk = Lookups(Path(a.root))
    bad = []
    for scp in a.scp:
        table = defaultdict(lambda: [0, 0.0])
        manifest = {}
        for line in open(scp):
            key, path = line.split(maxsplit=1)
            path = path.strip()
            if "/musan/" in path:
                lic, c = lk.musan(path)
                src = "musan/" + "/".join(Path(path).parts[-3:-1])
            elif "chime3_background" in path:
                lic, c, src = "CC BY 4.0 (Hugging Face)", "BY", "chime3_background"
            elif "/data/rir/" in path or path.startswith("data/rir/"):
                lic, c, src = "BUT ReverbDB: Apache-2.0", "permissive", "but_reverbdb"
            elif "/aug_banks/" in path:
                bank = Path(path).parent.parent
                if bank not in manifest:
                    manifest[bank] = defaultdict(list)
                    for r in csv.DictReader(
                        open(bank / "licences.tsv"), delimiter="\t"
                    ):
                        manifest[bank][str(Path(r["file"]).resolve())].append(r)
                rows = manifest[bank].get(str(Path(path).resolve()), [])
                classes = [
                    "BY" if r["corpus"] == "slakh2100" else classify(r["licence"])
                    for r in rows
                ]
                worst = next((x for x in classes if x not in OK), "BY")
                lic = f"{len(rows)} clips"
                c = worst if rows else "unknown"
                src = f"aug_banks/{rows[0]['corpus']}" if rows else "aug_banks/?"
            else:
                lic, c, src = "", "unknown", "unresolved"
            table[(src, c)][0] += 1
            if c not in OK:
                bad.append((src, key, path, lic, c, 0.0))
        print(f"== {scp}")
        for (src, c), (n, _) in sorted(table.items()):
            flag = "" if c in OK else "   <-- FAIL"
            print(f"  {src:36s} {c:11s} {n:6d}{flag}")
    report_failures(bad)


def cmd_musan(a):
    lk = Lookups(Path("."))
    for part in ("music", "noise"):
        for d in sorted(p for p in (Path(a.musan) / part).iterdir() if p.is_dir()):
            counts = defaultdict(int)
            for wav in sorted(d.glob("*.wav")):
                lic, c = lk.musan(str(wav))
                counts[c] += 1
            print(
                f"{part}/{d.name}\t"
                + "\t".join(f"{k}={v}" for k, v in sorted(counts.items()))
            )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("root")
    d.add_argument("dump")
    d.add_argument("out")
    li = sub.add_parser("lists")
    li.add_argument("root")
    li.add_argument("scp", nargs="+")
    mu = sub.add_parser("musan")
    mu.add_argument("musan")
    a = p.parse_args()
    {"dump": cmd_dump, "lists": cmd_lists, "musan": cmd_musan}[a.cmd](a)


if __name__ == "__main__":
    main()
