#!/usr/bin/env python3
# Copyright 2026 ESPnet contributors
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Prepare language targets outside the generic LID preprocessor."""

import argparse
import logging
from pathlib import Path

import numpy as np


def prepare_targets(languages, feature, excluded, get_features):
    """Require valid features for every language not explicitly excluded."""
    targets = {}
    missing = []
    dimension = None
    for language in languages:
        if language in excluded:
            logging.warning("Omitting %s from auxiliary targets only", language)
            continue
        try:
            values = get_features([language], feature)[language]
        except (KeyError, ValueError) as error:
            missing.append(f"{language} ({error})")
            continue
        vector = np.asarray(values, dtype=np.float32)
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            raise ValueError(f"Invalid lang2vec target for {language}")
        if dimension is not None and len(vector) != dimension:
            raise ValueError(f"Inconsistent lang2vec dimension for {language}")
        dimension = len(vector)
        targets[language] = vector
    if missing:
        raise ValueError("Missing lang2vec features: " + "; ".join(missing))
    if not targets:
        raise ValueError("At least one language must have auxiliary targets")
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang2utt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feature",
        default="geo",
        choices=("geo", "phonology_knn", "syntax_knn", "inventory_knn"),
    )
    parser.add_argument("--exclude", nargs="*", default=[])
    args = parser.parse_args()
    try:
        from lang2vec.lang2vec import get_features
    except ImportError as error:
        raise ImportError(
            "Install target preparation tools: cd tools && make lang2vec.done"
        ) from error
    languages = [
        line.split()[0]
        for line in args.lang2utt.read_text().splitlines()
        if line.strip()
    ]
    if len(languages) != len(set(languages)):
        parser.error("lang2utt must contain one row per language")
    targets = prepare_targets(languages, args.feature, set(args.exclude), get_features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            language
            + " "
            + " ".join(format(float(value), ".9g") for value in vector)
            + "\n"
            for language, vector in sorted(targets.items())
        )
    )
    logging.warning(
        "Wrote %d %s targets to %s", len(targets), args.feature, args.output
    )


if __name__ == "__main__":
    main()
