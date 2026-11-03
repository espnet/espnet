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


def read_kv(path):
    """Read ``<utt> <lang> ...`` lines as {utt: (lang, ...)} without ``<>``."""
    rows = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            key, *tokens = line.split()
            rows[key] = tuple(t.strip("<>") for t in tokens)
    return rows


def main():
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
    args = parser.parse_args()
    if len(args.ref) != len(args.hyp):
        parser.error("give one --hyp per --ref")

    refs, hyps = {}, {}
    for ref, hyp in zip(args.ref, args.hyp):
        ref_rows = read_kv(ref)
        if set(refs) & set(ref_rows):
            raise ValueError(f"duplicate utterance ids in {ref}")
        refs.update(ref_rows)
        hyps.update(read_kv(hyp))
    missing = sorted(set(refs) - set(hyps))
    if missing:
        raise ValueError(f"{len(missing)} references without hypothesis: {missing[:5]}")
    seen_pairs = {
        frozenset(v) for v in read_kv(args.seen_utt2langs).values() if len(v) == 2
    }
    groups = defaultdict(list)
    for utt, ref in refs.items():
        exact = int(set(hyps[utt]) == set(ref))
        groups["all"].append(exact)
        groups[f"{len(ref)}-language"].append(exact)
        if len(ref) == 2:
            groups["seen" if frozenset(ref) in seen_pairs else "unseen"].append(exact)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write("group\tn\taccuracy\n")
        for group, values in groups.items():
            line = f"{group}\t{len(values)}\t{100 * sum(values) / len(values):.2f}"
            f.write(line + "\n")
            print(f"{args.out.parent.name}\t{line}")


if __name__ == "__main__":
    main()
