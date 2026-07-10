#!/usr/bin/env python3
"""Overwrite fbank_mean / fbank_std in a recipe YAML config with corpus stats.

The values are written in-place. Used by ssl1/beats.sh after stage 4 to keep
the encoder, tokenizer-train and tokenizer-inference configs in sync with the
fbank statistics computed over the recipe's training set.

Edits are line-based regex substitutions, so comments, blank lines, key order,
and the file's existing indentation style are all preserved. Already-equal
values (within a small tolerance) are left untouched so re-running stage 4
does not produce noisy git diffs.
"""

import argparse
import logging
import math
import re
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

EQUAL_TOL = 1e-4
LINE_RE = re.compile(
    r"^(?P<indent>\s*)(?P<key>fbank_(?:mean|std)):\s*(?P<val>[-0-9.eE+]+)\s*$"
)


def _load_stats(path):
    mean = std = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("fbank_mean:"):
                mean = float(line.split(":", 1)[1])
            elif line.startswith("fbank_std:"):
                std = float(line.split(":", 1)[1])
    if mean is None or std is None:
        raise ValueError(f"fbank_mean / fbank_std not found in {path}")
    return mean, std


def overlay(config_path, mean, std):
    new_vals = {"fbank_mean": mean, "fbank_std": std}
    with open(config_path) as f:
        lines = f.readlines()

    out = []
    changes = []
    for line in lines:
        m = LINE_RE.match(line)
        if m and m.group("key") in new_vals:
            cur = float(m.group("val"))
            new = new_vals[m.group("key")]
            if math.isclose(cur, new, abs_tol=EQUAL_TOL):
                out.append(line)
                continue
            indent = m.group("indent")
            key = m.group("key")
            out.append(f"{indent}{key}: {new:.5f}\n")
            changes.append((key, cur, new))
        else:
            out.append(line)

    if not changes:
        logger.info(
            "%s: no fbank_mean/fbank_std field found, or already up-to-date.",
            config_path,
        )
        return

    with open(config_path, "w") as f:
        f.writelines(out)
    summary = ", ".join(f"{k}: {old:g} -> {new:.5f}" for k, old, new in changes)
    logger.info("%s: %s", config_path, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats_file", required=True, help="fbank_stats.txt path")
    parser.add_argument("--config", required=True, help="YAML config to update")
    args = parser.parse_args()

    mean, std = _load_stats(args.stats_file)
    overlay(args.config, mean, std)


if __name__ == "__main__":
    sys.exit(main())
