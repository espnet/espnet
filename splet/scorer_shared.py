#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Turning a score config into scores.

The same five-step shape as ``versa/scorer_shared.py``: load the config,
build the modules it names, run them over the corpus, write one JSON object
per utterance, summarize. A metric is a pair of plain functions --
``*_setup`` builds whatever state it needs, ``*_metric`` scores one item and
returns a flat dict -- which is VERSA's contract unchanged.

One thing is deliberately not copied. VERSA dispatches with a long
``if config["name"] == ...`` chain; SPLET uses the table below. The config
file, the metric function signatures and the output are identical either
way, and a table is what lets ``splet --list-metrics`` and the tests
enumerate what exists.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from splet import metrics as metric_keys
from splet.utterance_metrics import error_rate

# name -> how to build it and how to call it.
#   tier:     which loop runs it (utterance, session or corpus)
#   setup:    factory called with the config entry's keyword arguments
#   metric:   scorer called per item
#   defaults: keyword arguments implied by the name itself
METRIC_CHOICES: Dict[str, Dict[str, Any]] = {
    "wer": {
        "tier": "utterance",
        "setup": error_rate.error_rate_setup,
        "metric": error_rate.error_rate_metric,
        "defaults": {"name": "wer", "tokenizer": "word"},
    },
    "cer": {
        "tier": "utterance",
        "setup": error_rate.error_rate_setup,
        "metric": error_rate.error_rate_metric,
        "defaults": {"name": "cer", "tokenizer": "char"},
    },
}


def load_score_modules(
    score_config: Sequence[Dict[str, Any]],
    tier: str = "utterance",
    normalize: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build the metrics of one tier from a score config.

    Args:
        score_config: The parsed config: a list of ``{"name": ..., **kwargs}``
            entries, as in VERSA.
        tier: Which tier to build. Entries belonging to another tier are
            skipped, so the same config drives all three loops.
        normalize: A normalization pipeline applied to every metric that does
            not name its own. Making the default explicit at the top of the
            config is what keeps one normalization from being applied to WER
            and a different one to BLEU without anyone noticing.

    Returns:
        Metric name to its callable and state.

    Raises:
        ValueError: If an entry has no name, or names an unknown metric.
    """
    modules: Dict[str, Dict[str, Any]] = {}
    for entry in score_config:
        entry = dict(entry)
        name = entry.pop("name", None)
        if name is None:
            raise ValueError(f"score config entry has no name: {entry}")
        if name not in METRIC_CHOICES:
            raise ValueError(
                f"unknown metric '{name}'. Available: {sorted(METRIC_CHOICES)}"
            )
        choice = METRIC_CHOICES[name]
        if choice["tier"] != tier:
            continue

        kwargs = {**choice["defaults"], **entry}
        kwargs.setdefault("normalize", list(normalize) if normalize else None)
        logging.info("Loading %s evaluation...", name)
        modules[name] = {
            "module": choice["metric"],
            "scorer": choice["setup"](**kwargs),
            "config": {"name": name, **kwargs},
        }
        logging.info("Initiate %s evaluation successfully.", name)
    return modules


def load_session_modules(
    score_config: Sequence[Dict[str, Any]],
    normalize: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build the session-tier metrics.

    None exist yet; :mod:`splet.session_metrics` documents the contract they
    will follow.
    """
    return load_score_modules(score_config, tier="session", normalize=normalize)


def load_corpus_modules(
    score_config: Sequence[Dict[str, Any]],
    normalize: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build the corpus-tier metrics.

    None exist yet; :mod:`splet.corpus_metrics` documents the contract they
    will follow.
    """
    return load_score_modules(score_config, tier="corpus", normalize=normalize)


def list_scoring(
    pred_texts: Dict[str, str],
    score_modules: Dict[str, Dict[str, Any]],
    gt_texts: Optional[Dict[str, str]] = None,
    output_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Score every utterance with every utterance-tier metric.

    Args:
        pred_texts: Utterance id to hypothesis text.
        score_modules: From :func:`load_score_modules`.
        gt_texts: Utterance id to reference text.
        output_file: Where to write one JSON object per utterance. The file
            is written as scoring proceeds, so a crash halfway through still
            leaves the scores computed so far.

    Returns:
        One result dict per utterance, each carrying its ``key``.

    Raises:
        KeyError: If a hypothesis has no reference. Scoring the utterances
            that happen to match and reporting the average would silently
            answer a different question than the one asked.
    """
    handle = open(output_file, "w", encoding="utf-8") if output_file else None
    try:
        score_info = []
        for key in pred_texts:
            if gt_texts is not None and key not in gt_texts:
                raise KeyError(f"no reference for hypothesis '{key}'")
            utt_score: Dict[str, Any] = {"key": key}
            for name, module in score_modules.items():
                utt_score.update(
                    module["module"](
                        module["scorer"],
                        pred_texts[key],
                        gt_texts[key] if gt_texts is not None else None,
                    )
                )
            score_info.append(utt_score)
            if handle is not None:
                handle.write(json.dumps(utt_score, ensure_ascii=False) + "\n")
        return score_info
    finally:
        if handle is not None:
            handle.close()


def session_scoring(*args, **kwargs):
    """Score every session. No session-tier metric exists yet.

    Raises:
        NotImplementedError: Always. See :mod:`splet.session_metrics` for the
            contract this loop will follow.
    """
    raise NotImplementedError(
        "the session tier has no metrics yet; see splet/session_metrics"
    )


def corpus_scoring(*args, **kwargs):
    """Score the corpus as a whole. No corpus-tier metric exists yet.

    Raises:
        NotImplementedError: Always. See :mod:`splet.corpus_metrics` for the
            contract this loop will follow.
    """
    raise NotImplementedError(
        "the corpus tier has no metrics yet; see splet/corpus_metrics"
    )


def load_summary(score_info: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize per-utterance results into corpus figures.

    Counts are summed. An error rate is recomputed from the summed counts --
    ``sum(errors) / sum(ref_len)`` -- which is what SCTK reports and is not
    the mean of the per-utterance rates unless every utterance is the same
    length. Anything else numeric is averaged. Text keys are skipped.

    Args:
        score_info: Per-utterance results from :func:`list_scoring`.

    Returns:
        The corpus figures, plus ``num_utterances``.
    """
    if not score_info:
        return {"num_utterances": 0}

    keys = [key for key in score_info[0] if key != "key"]
    summary: Dict[str, Any] = {"num_utterances": len(score_info)}

    for key in keys:
        if metric_keys.is_str_key(key):
            continue
        values = [score[key] for score in score_info if key in score]
        if metric_keys.is_count_key(key):
            summary[key] = sum(values)
        elif (
            f"{key}{metric_keys.ERROR_SUFFIX}" in keys
            and f"{key}{metric_keys.REF_LEN_SUFFIX}" in keys
        ):
            errors = sum(
                score[f"{key}{metric_keys.ERROR_SUFFIX}"] for score in score_info
            )
            ref_len = sum(
                score[f"{key}{metric_keys.REF_LEN_SUFFIX}"] for score in score_info
            )
            summary[key] = errors / ref_len if ref_len else 0.0
        else:
            summary[key] = sum(values) / len(values)
    return summary
