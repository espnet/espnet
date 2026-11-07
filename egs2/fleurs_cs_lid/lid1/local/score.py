#!/usr/bin/env python3
"""Exact language-set accuracy, called by lid.sh stage 7 for one test set.

lid.sh passes <lid_exp>/inference/<model>/<test_set>/<test_set>_lids. The class
scores are recomputed from the per-utterance embeddings written next to it by
stage 6 (--extract_embd true) and the sub-center weights of <lid_exp>/<model>.pth:
single-language references are scored with the top-1 class, two-language
references with the top-2 classes, or with the top-1 atomic pair class
(expanded to its languages) when the training classes are pairs such as
ara-eng. References are read from the utt2langs next to --target_lids.
Two-language references are also split into pairs seen / unseen in the
CS-FLEURS training data (--seen_utt2langs).

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
        for line in f:
            key, value = line.rstrip("\n").split(maxsplit=1)
            rows[key] = value.split()
    return rows


def language_set(labels):
    """Expand labels such as ``ara-eng`` and return the set of languages."""
    return frozenset(x for label in labels for x in label.split("-"))


def class_logits(embds, weight, num_classes, scale):
    """Scaled sub-center cosine logits of the trained head (no margins)."""
    cosine = torch.nn.functional.linear(
        torch.nn.functional.normalize(embds), torch.nn.functional.normalize(weight)
    )
    cosine = cosine.reshape(len(embds), num_classes, -1).max(dim=2).values
    return (cosine * scale).numpy()


def predict(pred_lids, target_lids, train_lang2utt):
    """Return [(utt, reference set, predicted set)] for one lid.sh test set."""
    infer_dir = Path(pred_lids).parent
    model_name = infer_dir.parent.name
    lid_exp = infer_dir.parent.parent.parent
    config = yaml.safe_load((lid_exp / "config.yaml").read_text(encoding="utf-8"))
    scale = float(config["loss_conf"].get("scale", 32))
    state = torch.load(lid_exp / f"{model_name}.pth", map_location="cpu")
    weight = state["loss.weight"].float()

    labels = list(read_kv(train_lang2utt))
    atomic = any("-" in label for label in labels)
    with np.load(infer_dir / f"{infer_dir.name}_utt_embds.npz") as npz:
        utt_ids = list(npz)
        embds = torch.from_numpy(np.stack([npz[utt] for utt in utt_ids])).float()
    logits = class_logits(embds, weight, len(labels), scale)
    index = {utt: i for i, utt in enumerate(utt_ids)}

    rows = []
    for utt, ref in sorted(read_kv(Path(target_lids).parent / "utt2langs").items()):
        ref = language_set(ref)
        k = 1 if atomic else len(ref)
        ranked = np.argsort(-logits[index[utt]], kind="stable")[:k]
        rows.append((utt, ref, language_set(labels[i] for i in ranked)))
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


def main():
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
    args = parser.parse_args()
    args.results.parent.mkdir(parents=True, exist_ok=True)

    if args.details:
        rows = []
        for path in args.details:
            for line in path.read_text(encoding="utf-8").splitlines()[1:]:
                utt, ref, pred, _ = line.split("\t")
                rows.append((utt, frozenset(ref.split()), frozenset(pred.split())))
    elif args.pred_lids and args.target_lids and args.train_lang2utt:
        rows = predict(args.pred_lids, args.target_lids, args.train_lang2utt)
        with (args.results.parent / DETAILS).open("w", encoding="utf-8") as f:
            f.write("utt\tref\tpred\texact\n")
            for utt, ref, pred in rows:
                ref_str, pred_str = " ".join(sorted(ref)), " ".join(sorted(pred))
                f.write(f"{utt}\t{ref_str}\t{pred_str}\t{int(pred == ref)}\n")
    else:
        parser.error(
            "give --pred_lids, --target_lids and --train_lang2utt or --details"
        )

    seen_pairs = {
        language_set(v)
        for v in read_kv(args.seen_utt2langs).values()
        if len(language_set(v)) == 2
    }
    with args.results.open("w", encoding="utf-8") as f:
        f.write("# Exact language-set accuracy (%)\ngroup\tn\taccuracy\n")
        for group, values in summarize(rows, seen_pairs).items():
            line = f"{group}\t{len(values)}\t{100 * np.mean(values):.2f}"
            f.write(line + "\n")
            print(f"{args.results.parent.name}\t{line}")


if __name__ == "__main__":
    main()
