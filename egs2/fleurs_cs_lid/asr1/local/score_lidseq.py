#!/usr/bin/env python3
"""Score decoded language-token sequences against utt2langs references.

Reports the exact unordered language-set accuracy and splits two-language
references into pairs seen / unseen in the CS-FLEURS training data
(--seen_utt2langs). Several --ref/--hyp pairs are scored as one set (the
"CS-FLEURS all" rows of the README).
"""

import argparse
from collections import defaultdict
from pathlib import Path


def read_kv(path, allow_empty=False):
    """Read ``<utt> <lang> ...`` lines as {utt: (lang, ...)} without ``<>``."""
    rows = {}
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            fields = line.split()
            if not fields or (len(fields) == 1 and not allow_empty):
                raise ValueError(f"{path}:{n}: expected '<utt> <lang> ...'")
            if fields[0] in rows:
                raise ValueError(f"{path}:{n}: duplicate utterance id {fields[0]}")
            rows[fields[0]] = tuple(t.strip("<>") for t in fields[1:])
    return rows


def read_refs(path):
    """Read a utt2langs file; a repeated language is an error."""
    refs = read_kv(path)
    # read_kv rejects blank and duplicate lines: entry n is line n
    for n, (utt, ref) in enumerate(refs.items(), 1):
        if len(set(ref)) != len(ref):
            raise ValueError(f"{path}:{n}: repeated language in {utt}")
    return refs


def read_seen_pairs(path):
    """Return the two-language sets of a utt2langs file."""
    return {frozenset(v) for v in read_refs(path).values() if len(v) == 2}


def read_hyps(path):
    """Read decoded text; an id without tokens is an empty prediction."""
    return read_kv(path, allow_empty=True)


def merge(paths, read):
    """Read several files into one dict; an id in two files is an error."""
    rows = {}
    for path in paths:
        new = read(path)
        if set(rows) & set(new):
            raise ValueError(f"duplicate utterance ids in {path}")
        rows.update(new)
    return rows


def load(ref_paths, hyp_paths):
    """Return refs and hyps; every id must have exactly one of each."""
    refs, hyps = merge(ref_paths, read_refs), merge(hyp_paths, read_hyps)
    missing = sorted(set(refs) - set(hyps))
    if missing:
        raise ValueError(f"{len(missing)} references without hypothesis: {missing[:5]}")
    extra = sorted(set(hyps) - set(refs))
    if extra:
        raise ValueError(f"{len(extra)} hypotheses without reference: {extra[:5]}")
    return refs, hyps


def is_exact(hyp, ref):
    """Compare as multisets: order is ignored, a repeated token is wrong."""
    return sorted(hyp) == sorted(ref)


def score(refs, hyps, seen_pairs):
    """Group exact-match results into all / N-language / seen / unseen."""
    groups = defaultdict(list)
    for utt, ref in refs.items():
        exact = int(is_exact(hyps[utt], ref))
        groups["all"].append(exact)
        groups[f"{len(ref)}-language"].append(exact)
        if len(ref) == 2:
            groups["seen" if frozenset(ref) in seen_pairs else "unseen"].append(exact)
    return groups


def write_results(path, groups):
    """Write the accuracy per group to ``path`` and print it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("group\tn\taccuracy\n")
        for group, values in groups.items():
            line = f"{group}\t{len(values)}\t{100 * sum(values) / len(values):.2f}"
            f.write(line + "\n")
            print(f"{path.parent.name}\t{line}")


def main(argv=None):
    """Write the accuracy per group (all, N-language, seen, unseen) to --out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", type=Path, nargs="+", required=True, help="utt2langs")
    parser.add_argument(
        "--hyp", type=Path, nargs="+", required=True, help="decoded text"
    )
    parser.add_argument(
        "--seen_utt2langs",
        type=Path,
        default=Path("data/train_fleurscs_cs/utt2langs"),
        help="Two-language sets in this file count as seen",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.ref) != len(args.hyp):
        parser.error("give one --hyp per --ref")

    refs, hyps = load(args.ref, args.hyp)
    write_results(args.out, score(refs, hyps, read_seen_pairs(args.seen_utt2langs)))


if __name__ == "__main__":
    main()
