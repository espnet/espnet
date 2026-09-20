#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""The ``splet-score`` command.

Deliberately the same shape as ``versa-score``: ``--pred`` and ``--gt``
carry the two sides, ``--score_config`` names a YAML list of metrics,
``--output_file`` receives one JSON object per utterance, and ``--io``
selects how the inputs are read. A recipe that already knows how to call
VERSA for its audio can call SPLET for its text without learning a second
convention.

``--hyp`` and ``--ref`` are accepted as aliases, because that is what the
text side of the field calls these two files.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional, Sequence

import yaml

from splet.scorer_shared import (
    METRIC_CHOICES,
    list_scoring,
    load_score_modules,
    load_summary,
)
from splet.utils_shared import IO_CHOICES, text_loader_setup


def get_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(description="Spoken Language Evaluation Toolkit")
    parser.add_argument(
        "--pred",
        "--hyp",
        dest="pred",
        type=str,
        help="Hypothesis text (Kaldi text file, JSONL, or directory).",
    )
    parser.add_argument(
        "--gt",
        "--ref",
        dest="gt",
        type=str,
        default=None,
        help="Reference text, in the same format as --pred.",
    )
    parser.add_argument(
        "--score_config", type=str, default=None, help="Configuration of Score Config"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to write per-utterance results as JSON lines.",
    )
    parser.add_argument(
        "--io",
        type=str,
        default="kaldi",
        choices=list(IO_CHOICES),
        help="io interface to use",
    )
    parser.add_argument(
        "--verbose",
        default=1,
        type=int,
        help="Verbosity level. Higher is more logging.",
    )
    parser.add_argument(
        "--list_metrics",
        action="store_true",
        help="Print the available metrics and exit.",
    )
    return parser


def _configure_logging(verbose: int) -> None:
    """Set the log level from the verbosity flag."""
    level = (
        logging.DEBUG if verbose > 1 else logging.INFO if verbose > 0 else logging.WARN
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Score a hypothesis file against a reference file.

    Args:
        argv: Command line arguments; ``sys.argv[1:]`` when omitted.

    Returns:
        A process exit status.
    """
    args = get_parser().parse_args(argv)
    _configure_logging(args.verbose)

    if args.list_metrics:
        for name, choice in sorted(METRIC_CHOICES.items()):
            print(f"{name}\t{choice['tier']}")
        return 0

    if args.pred is None or args.score_config is None:
        logging.error("--pred and --score_config are required")
        return 2

    with open(args.score_config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    # A config may be a bare list of metrics, as in VERSA, or a mapping with
    # a shared `normalize:` pipeline and the list under `metrics:`.
    if isinstance(config, dict):
        score_config = config.get("metrics", [])
        normalize = config.get("normalize")
    else:
        score_config, normalize = config, None

    pred_texts = text_loader_setup(args.pred, args.io)
    gt_texts = text_loader_setup(args.gt, args.io) if args.gt else None
    logging.info("The number of utterances = %d", len(pred_texts))

    score_modules = load_score_modules(score_config, normalize=normalize)
    if not score_modules:
        logging.error("no utterance-level scoring function is provided")
        return 2

    score_info = list_scoring(
        pred_texts, score_modules, gt_texts, output_file=args.output_file
    )
    summary = load_summary(score_info)
    logging.info("Summary: %s", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
