#!/usr/bin/env python3
"""Write one report per corpus of what the data preparation read and kept.

    python3 local/v2/corpus_report.py dump/v2 dump/v2/reports

For every corpus directory of the dump root, reports/<corpus>.json holds the
source files (from local/v2/downloads/<corpus>.tsv: path, size, checksum),
licence, domain, label source, reader options and hour caps, and per split
the recordings and hours written by the reader, the recordings and hours kept
after the alignment, the speech hours, and the reasons and counts of every
exclusion (unreadable or filtered sources, recordings over a cap, recordings
dropped by the alignment), and the final split: the groups of a corpus
split by group, its dev groups, the recordings moved between splits, and the
recordings and hours of each final split (local/v2/splits.py).
reports/not_used.json lists the registered
corpora left out, with the reason. The reports have no timestamps or
machine-specific paths, so two runs of the same recipe can be compared with
diff.
"""

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adapters  # noqa: E402
from splits import corpus_dumps, manifest_rows, resolve  # noqa: E402


def dump_dirs(split_dir: Path):
    if (split_dir / "wav.scp").exists():
        return [split_dir]
    return sorted(p for p in split_dir.glob("shard*") if (p / "wav.scp").exists())


def hours(manifest: Path, column: str) -> float:
    if not manifest.exists():
        return 0.0
    with manifest.open() as f:
        cols = f.readline().rstrip("\n").split("\t")
        i = cols.index(column)
        return sum(float(line.rstrip("\n").split("\t")[i]) for line in f) / 3600


def lines(path: Path):
    return [x for x in path.read_text().splitlines() if x] if path.exists() else []


def reason(text: str) -> str:
    """The kind of an exclusion: the reason without ids, numbers or paths."""
    words = []
    for w in text.split():
        if any(c.isdigit() for c in w) or "/" in w:
            break
        words.append(w)
    return " ".join(words).rstrip(":") or "unspecified"


def sources(directory: str):
    tsv = Path(adapters.__file__).resolve().parent / "downloads" / f"{directory}.tsv"
    out = []
    for line in lines(tsv):
        if not line.startswith("#"):
            f = (line.split("\t") + [""] * 6)[:6]
            out.append(dict(path=f[1], size=int(f[3] or 0), checksum=f[4]))
    return out


def report(corpus: str, root: Path):
    spec = adapters.ADAPTERS.get(corpus)
    rep = dict(
        corpus=corpus,
        directory=spec.directory if spec else corpus,
        domain=spec.domain if spec else "meeting",
        label_source=spec.label_source if spec else "H",
        licence=spec.licence if spec else "",
        options=adapters.OPTIONS.get(corpus, {}),
        max_hours=adapters.MAX_HOURS_PER_CORPUS.get(corpus, adapters.MAX_HOURS),
        max_hours_per_lang=adapters.MAX_HOURS_PER_LANG.get(corpus),
        sources=sources(spec.directory) if spec and spec.directory else [],
        splits={},
    )
    for split_dir in sorted(p for p in (root / corpus).iterdir() if p.is_dir()):
        dirs = dump_dirs(split_dir)
        if not dirs:
            continue
        s = collections.Counter()
        skipped, dropped = collections.Counter(), collections.Counter()
        for d in dirs:
            before = d / "before_align" / "manifest.tsv"
            written = before if before.exists() else d / "manifest.tsv"
            s["recordings_written"] += max(len(lines(written)) - 1, 0)
            s["hours_written"] += hours(written, "seconds")
            s["recordings_kept"] += max(len(lines(d / "manifest.tsv")) - 1, 0)
            s["hours_kept"] += hours(d / "manifest.tsv", "seconds")
            s["speech_hours_kept"] += hours(d / "manifest.tsv", "speech_seconds")
            summary = json.loads((d / "summary.json").read_text())
            s["recordings_over_cap"] += summary.get("not_added_over_cap", 0)
            for x in lines(d / "skipped.txt"):
                skipped[reason(x.split("\t", 1)[-1])] += 1
            for x in lines(d / "align_dropped.tsv"):
                dropped[reason(x.split("\t", 1)[-1])] += 1
        rep["splits"][split_dir.name] = dict(
            {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()},
            skipped=dict(sorted(skipped.items())),
            align_dropped=dict(sorted(dropped.items())),
        )
        cfg = sorted(split_dir.glob("**/align_config.json"))
        if cfg:
            c = json.loads(cfg[0].read_text())
            rep["alignment"] = {
                k: c[k]
                for k in ("model_tag", "revision", "min_pause", "min_word_score")
            }
    final, summary = resolve(root / corpus)
    totals = collections.Counter()
    for _, dump in corpus_dumps(root / corpus):
        for r in manifest_rows(dump):
            split = final[r["rec_id"]]
            totals[(split, "recordings")] += 1
            totals[(split, "hours")] += float(r["seconds"]) / 3600
            totals[(split, "speech_hours")] += float(r["speech_seconds"]) / 3600
    rep["final_split"] = dict(
        summary,
        **{
            f"{split}_{k}": round(v, 3) if isinstance(v, float) else v
            for (split, k), v in sorted(totals.items())
        },
    )
    return rep


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for corpus_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if corpus_dir.name in ("log", out.name):
            continue
        rep = report(corpus_dir.name, root)
        (out / f"{corpus_dir.name}.json").write_text(
            json.dumps(rep, indent=1, sort_keys=True) + "\n"
        )
    (out / "not_used.json").write_text(
        json.dumps(dict(sorted(adapters.NOT_USED.items())), indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
