#!/usr/bin/env python3
"""Combine corpus dumps into the training and validation sets.

    python3 local/v2/combine.py dump/raw dump/v2/ami dump/v2/fsd50k ...

Each input is a corpus directory written by prep.py or from_kaldi.py
(<corpus>/<split>[/shard<k>]). The final split of every recording comes from
splits.resolve(): official splits are kept, and corpora split by group get
their dev groups there. Writes <out>/train and <out>/dev: wav.scp,
utt2num_samples, utt2category, espnet_rttm, manifest.tsv (sorted; the Omni-IO
entries keep pointing at the archives of the inputs), feats_type, and
domains.tsv (recordings and hours per domain and corpus). Exits with an error
if a corpus has no dev recording.
"""

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from splits import SPLITS, corpus_dumps, manifest_rows, resolve  # noqa: E402
from vad_dump import MANIFEST_FIELDS  # noqa: E402

KEYED = ("wav.scp", "utt2num_samples", "utt2category")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("corpora", type=Path, nargs="+")
    args = parser.parse_args()
    lines = {s: collections.defaultdict(list) for s in SPLITS}
    no_dev = []
    for corpus_dir in args.corpora:
        final, _ = resolve(corpus_dir)
        if "dev" not in final.values():
            no_dev.append(corpus_dir.name)
        for _, dump in corpus_dumps(corpus_dir):
            for name in KEYED:
                for line in (dump / name).read_text().splitlines(keepends=True):
                    split = final[line.split(maxsplit=1)[0]]
                    if split != "drop":
                        lines[split][name].append(line)
            for line in (dump / "espnet_rttm").read_text().splitlines(keepends=True):
                split = final[line.split()[1]]
                if split != "drop":
                    lines[split]["espnet_rttm"].append(line)
            # manifests written before a column was added get it empty
            for r in manifest_rows(dump):
                split = final[r["rec_id"]]
                if split != "drop":
                    row = "\t".join(r.get(k, "") for k in MANIFEST_FIELDS)
                    lines[split]["manifest.tsv"].append(row + "\n")
    if no_dev:
        sys.exit(f"no dev recordings in: {', '.join(no_dev)}")
    header = "\t".join(MANIFEST_FIELDS) + "\n"
    cols = list(MANIFEST_FIELDS)
    for split in SPLITS:
        out = args.out / split
        out.mkdir(parents=True, exist_ok=True)
        for name in KEYED:
            keys = [x.split(maxsplit=1)[0] for x in lines[split][name]]
            dup = [k for k, c in collections.Counter(keys).items() if c > 1]
            if dup:
                raise ValueError(f"{split}/{name}: duplicate ids, e.g. {dup[:3]}")
            (out / name).write_text("".join(sorted(lines[split][name])))
        (out / "espnet_rttm").write_text("".join(sorted(lines[split]["espnet_rttm"])))
        (out / "manifest.tsv").write_text(
            header + "".join(sorted(lines[split]["manifest.tsv"]))
        )
        (out / "feats_type").write_text("raw\n")
        hours = collections.defaultdict(lambda: [0, 0.0, 0.0])
        for row in lines[split]["manifest.tsv"]:
            r = dict(zip(cols, row.rstrip("\n").split("\t")))
            h = hours[(r["domain"], r["corpus"])]
            h[0] += 1
            h[1] += float(r["seconds"]) / 3600
            h[2] += float(r["speech_seconds"]) / 3600
        with (out / "domains.tsv").open("w") as f:
            f.write("domain\tcorpus\trecordings\thours\tspeech_hours\n")
            for (dom, cor), (n, t, s) in sorted(hours.items()):
                f.write(f"{dom}\t{cor}\t{n}\t{t:.2f}\t{s:.2f}\n")


if __name__ == "__main__":
    main()
