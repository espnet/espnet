#!/usr/bin/env python3
"""Exact language-set accuracy, called by lid.sh stage 7 for one test set.

lid.sh passes <lid_exp>/inference/<model>/<test_set>/<test_set>_lids. The class
scores are recomputed from the per-utterance embeddings written next to it by
stage 6 (--extract_embd true) and the sub-center weights of <lid_exp>/<model>.pth:
a reference with k languages is scored with the top-k classes, or with the
top-1 atomic pair class (expanded to its languages) when the training classes
are pairs such as ara-eng. References are read from the utt2langs next to
--target_lids. Two-language references are also split into pairs seen / unseen
in the CS-FLEURS training data (--seen_utt2langs).

With --details, the per-utterance files of several test sets are merged
instead (the "CS-FLEURS all" rows of the README).
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

DETAILS = "lang_set_details"


def read_kv(path):
    """Read a Kaldi-style ``<key> <value> ...`` file as {key: [values]}."""
    rows = {}
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"{path}:{n}: expected '<key> <value> ...'")
            key, values = fields[0], fields[1:]
            if key in rows:
                raise ValueError(f"{path}:{n}: duplicate key {key}")
            rows[key] = values
    return rows


def language_set(labels):
    """Expand labels such as ``ara-eng`` and return the set of languages."""
    return frozenset(x for label in labels for x in label.split("-"))


def read_refs(path):
    """Read utt2langs as {utt: language set}; a repeated language is an error."""
    refs = {}
    # read_kv rejects blank and duplicate lines: entry n is line n
    for n, (utt, labels) in enumerate(read_kv(path).items(), 1):
        langs = [x for label in labels for x in label.split("-")]
        if len(set(langs)) != len(langs):
            raise ValueError(f"{path}:{n}: repeated language in {utt}")
        refs[utt] = frozenset(langs)
    return refs


def read_seen_pairs(path):
    """Return the two-language sets of a utt2langs file."""
    return {ref for ref in read_refs(path).values() if len(ref) == 2}


def class_logits(embds, weight, num_classes, scale):
    """Scaled sub-center cosine logits of the trained head (no margins).

    ``weight`` has K sub-center rows per class, grouped by class (C * K, D).
    """
    cosine = torch.nn.functional.linear(
        torch.nn.functional.normalize(embds), torch.nn.functional.normalize(weight)
    )
    cosine = cosine.reshape(len(embds), num_classes, -1).max(dim=2).values
    return (cosine * scale).numpy()


def predict_set(logits, labels, ref):
    """Predicted language set for one utterance with reference set ``ref``.

    Independent classes: the top k = len(ref) classes. Atomic pair classes:
    the top-1 class. Ties go to the lower class index.
    """
    k = 1 if any("-" in label for label in labels) else len(ref)
    ranked = np.argsort(-logits, kind="stable")[:k]
    return language_set(labels[i] for i in ranked)


def load_head(lid_exp, model_name):
    """Return the sub-center weight and logit scale of a trained model."""
    config = yaml.safe_load((lid_exp / "config.yaml").read_text(encoding="utf-8"))
    scale = float(config["loss_conf"].get("scale", 32))
    state = torch.load(lid_exp / f"{model_name}.pth", map_location="cpu")
    return state["loss.weight"].float(), scale


def load_embeddings(path, refs):
    """Return the sorted reference ids and their stacked embeddings."""
    with np.load(path) as npz:
        missing = sorted(set(refs) - set(npz))
        extra = sorted(set(npz) - set(refs))
        if missing or extra:
            raise ValueError(
                f"{path}: {len(missing)} references without embedding "
                f"{missing[:5]}, {len(extra)} embeddings without reference {extra[:5]}"
            )
        utts = sorted(refs)
        return utts, torch.from_numpy(np.stack([npz[u] for u in utts])).float()


def predict(pred_lids, target_lids, train_lang2utt):
    """Return [(utt, reference set, predicted set)] for one lid.sh test set."""
    infer_dir = Path(pred_lids).parent
    model_name = infer_dir.parent.name
    lid_exp = infer_dir.parent.parent.parent
    weight, scale = load_head(lid_exp, model_name)
    # Class index = line order of lang2utt, as LIDPreprocessor maps labels
    labels = list(read_kv(train_lang2utt))
    refs = read_refs(Path(target_lids).parent / "utt2langs")
    utts, embds = load_embeddings(infer_dir / f"{infer_dir.name}_utt_embds.npz", refs)
    logits = class_logits(embds, weight, len(labels), scale)
    return [
        (utt, refs[utt], predict_set(row, labels, refs[utt]))
        for utt, row in zip(utts, logits)
    ]


def write_details(path, rows):
    """Write the per-utterance results read by --details."""
    with Path(path).open("w", encoding="utf-8") as f:
        f.write("utt\tref\tpred\texact\n")
        for utt, ref, pred in rows:
            ref_str, pred_str = " ".join(sorted(ref)), " ".join(sorted(pred))
            f.write(f"{utt}\t{ref_str}\t{pred_str}\t{int(pred == ref)}\n")


def read_details(paths):
    """Merge the per-utterance results of several test sets."""
    rows, seen = [], set()
    for path in paths:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        for n, line in enumerate(lines[1:], 2):
            fields = line.split("\t")
            if len(fields) != 4 or not fields[1] or fields[0] in seen:
                raise ValueError(f"{path}:{n}: malformed or duplicate line")
            utt, ref, pred, _ = fields
            seen.add(utt)
            rows.append((utt, frozenset(ref.split()), frozenset(pred.split())))
    return rows


def summarize(rows, seen_pairs):
    """Group exact-match results into all / N-language / seen / unseen."""
    groups = defaultdict(list)
    for _, ref, pred in rows:
        exact = int(pred == ref)
        groups["all"].append(exact)
        groups[f"{len(ref)}-language"].append(exact)
        if len(ref) == 2:
            groups["seen" if ref in seen_pairs else "unseen"].append(exact)
    return groups


def write_results(path, groups):
    """Write the accuracy per group to ``path`` and print it."""
    with Path(path).open("w", encoding="utf-8") as f:
        f.write("# Exact language-set accuracy (%)\ngroup\tn\taccuracy\n")
        for group, values in groups.items():
            line = f"{group}\t{len(values)}\t{100 * np.mean(values):.2f}"
            f.write(line + "\n")
            print(f"{Path(path).parent.name}\t{line}")


def main(argv=None):
    """Score one lid.sh test set, or merge the details of several sets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred_lids", type=Path)
    parser.add_argument("--target_lids", type=Path)
    parser.add_argument("--train_lang2utt", type=Path)
    parser.add_argument("--details", type=Path, nargs="+")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--seen_utt2langs",
        type=Path,
        default=Path("data/train_fleurscs_cs/utt2langs"),
        help="Two-language sets in this file count as seen",
    )
    args = parser.parse_args(argv)

    if args.details:
        rows = read_details(args.details)
    elif args.pred_lids and args.target_lids and args.train_lang2utt:
        rows = predict(args.pred_lids, args.target_lids, args.train_lang2utt)
    else:
        parser.error(
            "give --pred_lids, --target_lids and --train_lang2utt or --details"
        )
    seen_pairs = read_seen_pairs(args.seen_utt2langs)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    if not args.details:
        write_details(args.results.parent / DETAILS, rows)
    write_results(args.results, summarize(rows, seen_pairs))


if __name__ == "__main__":
    main()
