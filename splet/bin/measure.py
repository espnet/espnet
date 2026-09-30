#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""The ``splet-measure`` command.

Deliberately the same shape as ``versa-score``: ``--pred`` and ``--gt``
carry the two sides, ``--metrics_config`` names a YAML list of metrics,
``--output_file`` receives one JSON object per utterance, and ``--io``
selects how the inputs are read. The printed summary ends with a
``metadata`` block (``splet/metadata.py``) recording the configuration that
produced it. A recipe that already knows how to call
VERSA for its audio can call SPLET for its text without learning a second
convention.

``--hyp`` and ``--ref`` are accepted as aliases, because that is what the
text side of the field calls these two files.

The command is ``measure``, after espnet3's stage of the same name, and not
``score``: in ESPnet a scorer is a beam-search component, and the word is
avoided throughout SPLET for that reason.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional, Sequence

import yaml

from splet.metadata import metadata
from splet.metric_registry import (
    METRIC_CHOICES,
    load_corpus_metrics,
    load_metrics,
    measure_corpus,
    measure_utterances,
)
from splet.summary import summarize
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
        "--metrics_config",
        type=str,
        default=None,
        help="YAML list of metrics, optionally under `metrics:` with a shared "
        "`normalize:` pipeline.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to write per-utterance results as JSON lines. Corpus-level "
        "metrics such as BLEU only appear in the printed summary.",
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
    """Measure a hypothesis file against a reference file.

    Args:
        argv: Command line arguments; ``sys.argv[1:]`` when omitted.

    Returns:
        A process exit status.
    """
    args = get_parser().parse_args(argv)
    _configure_logging(args.verbose)

    if args.list_metrics:
        for name, spec in sorted(METRIC_CHOICES.items()):
            requires = ",".join(spec.requires) or "-"
            print(f"{name}\t{spec.tier}\t{requires}")
        return 0

    if args.pred is None or args.metrics_config is None:
        logging.error("--pred and --metrics_config are required")
        return 2

    with open(args.metrics_config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    # A config may be a bare list of metrics, as in VERSA, or a mapping with
    # a shared `normalize:` pipeline and the list under `metrics:`.
    if isinstance(config, dict):
        metrics_config = config.get("metrics", [])
        normalize = config.get("normalize")
    else:
        metrics_config, normalize = config, None

    pred_texts = text_loader_setup(args.pred, args.io)
    gt_texts = text_loader_setup(args.gt, args.io) if args.gt else None
    logging.info("The number of utterances = %d", len(pred_texts))

    metrics = load_metrics(metrics_config, normalize=normalize)
    corpus_metrics = load_corpus_metrics(metrics_config, normalize=normalize)
    if not metrics and not corpus_metrics:
        logging.error("no metric is configured")
        return 2

    if metrics:
        results = measure_utterances(
            pred_texts, metrics, gt_texts, output_file=args.output_file
        )
        summary = summarize(results, metrics)
    else:
        summary = {"num_utterances": len(pred_texts)}
    if corpus_metrics:
        # One result for the whole corpus. Going through summarize() still
        # rejects a key the metric did not declare.
        corpus = summarize(
            [measure_corpus(pred_texts, corpus_metrics, gt_texts)], corpus_metrics
        )
        del corpus["num_utterances"]
        summary.update(corpus)
    logging.info("Summary: %s", summary)
    # The summary carries what produced it: a number without its
    # normalization, tokenizer and backend cannot be reproduced or compared.
    summary["metadata"] = metadata({**metrics, **corpus_metrics})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
