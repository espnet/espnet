#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Turning a metrics config into results.

The same five-step shape as ``versa/metric_registry.py``: load the config,
build the modules it names, run them over the corpus, write one JSON object
per utterance, summarize. A metric is a pair of plain functions --
``*_setup`` builds whatever state it needs, ``*_metric`` measures one item and
returns a flat dict -- which is VERSA's contract unchanged.

One thing is deliberately not copied. VERSA dispatches with a long
``if config["name"] == ...`` chain; SPLET registers each metric as a
:class:`MetricSpec` in :data:`METRIC_CHOICES`. The config file, the metric
function signatures and the output are identical either way, and a table is
what lets ``splet-measure --list_metrics`` and the tests enumerate what
exists.

A spec says four things about a metric that nothing else should have to
guess: which **tier** runs it, what input it **requires** beyond the
hypothesis, how each result key it reports is **reduced** over the corpus,
and which **version** of the implementation produced a number. A config
entry names the implementation (``name``) and may give the instance its own
``id``; the id is the prefix of every key the instance reports, so the same
implementation can run twice in one config, raw and normalized, say,
without one overwriting the other.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from splet.utterance_metrics import error_rate

#: The tiers, i.e. which loop runs a metric.
TIER_CHOICES = ("utterance", "session", "corpus")

#: What a metric may require of its input beyond the hypothesis text.
#:
#: ``reference``
#:     A reference on the ``--gt`` side, matched to the hypotheses by id.
#: ``timestamps``
#:     Every turn carries ``start`` and ``end`` (DER, JER; session tier).
#: ``speakers``
#:     Every turn carries a ``speaker`` (cpWER, DER, JER; session tier).
REQUIREMENT_CHOICES = ("reference", "timestamps", "speakers")


@dataclass(frozen=True)
class MetricSpec:
    """Everything the registry knows about one metric implementation."""

    #: Which loop runs it: ``utterance``, ``session`` or ``corpus``.
    tier: str
    #: Factory called with the config entry's keyword arguments.
    setup: Callable[..., Any]
    #: Called per item with the state ``setup`` returned.
    metric: Callable[..., Dict[str, Any]]
    #: Result key suffix (after the configured id) -> reduction rule; see
    #: ``splet/summary.py``. Every key the metric reports must be here.
    outputs: Mapping[str, str]
    #: Inputs the metric needs beyond the hypothesis; see
    #: :data:`REQUIREMENT_CHOICES`. Checked before anything is measured.
    requires: Tuple[str, ...] = ()
    #: Keyword arguments implied by the registered name itself.
    defaults: Dict[str, Any] = field(default_factory=dict)
    #: Version of the implementation; bump it when the computation changes,
    #: so a saved result says which one produced it.
    version: str = "1"

    def __post_init__(self) -> None:
        """Reject a spec that names a tier or requirement that does not exist."""
        if self.tier not in TIER_CHOICES:
            raise ValueError(f"unknown tier '{self.tier}'; expected {TIER_CHOICES}")
        unknown = [r for r in self.requires if r not in REQUIREMENT_CHOICES]
        if unknown:
            raise ValueError(
                f"unknown requirements {unknown}; expected {REQUIREMENT_CHOICES}"
            )


METRIC_CHOICES: Dict[str, MetricSpec] = {
    "wer": MetricSpec(
        tier="utterance",
        setup=error_rate.error_rate_setup,
        metric=error_rate.error_rate_metric,
        outputs=error_rate.OUTPUTS,
        requires=("reference",),
        defaults={"tokenizer": "word"},
    ),
    "cer": MetricSpec(
        tier="utterance",
        setup=error_rate.error_rate_setup,
        metric=error_rate.error_rate_metric,
        outputs=error_rate.OUTPUTS,
        requires=("reference",),
        defaults={"tokenizer": "char"},
    ),
}


def load_metrics(
    metrics_config: Sequence[Dict[str, Any]],
    tier: str = "utterance",
    normalize: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build the metrics of one tier from a metrics config.

    Args:
        metrics_config: The parsed config: a list of ``{"name": ..., **kwargs}``
            entries, as in VERSA. An entry may add ``id`` to run one
            implementation under its own identifier (``name: wer, id:
            wer_norm``); without it the id is the name.
        tier: Which tier to build. Entries belonging to another tier are
            skipped, so the same config drives all three loops.
        normalize: A normalization pipeline applied to every metric that does
            not name its own. Making the default explicit at the top of the
            config is what keeps one normalization from being applied to WER
            and a different one to BLEU without anyone noticing.

    Returns:
        Configured id to ``{"name", "spec", "config", "module", "state"}``:
        the implementation name, its spec, the resolved keyword arguments
        the state was built from (what the summary's ``metadata`` reports),
        the metric callable and the state to pass it.

    Raises:
        ValueError: If an entry has no name, names an unknown metric, or
            repeats an id. Two instances under one id would report into the
            same keys and the later would silently replace the earlier.
    """
    modules: Dict[str, Dict[str, Any]] = {}
    for entry in metrics_config:
        entry = dict(entry)
        name = entry.pop("name", None)
        if name is None:
            raise ValueError(f"metrics config entry has no name: {entry}")
        if name not in METRIC_CHOICES:
            raise ValueError(
                f"unknown metric '{name}'. Available: {sorted(METRIC_CHOICES)}"
            )
        spec = METRIC_CHOICES[name]
        metric_id = str(entry.pop("id", name))
        if spec.tier != tier:
            continue
        if metric_id in modules:
            raise ValueError(
                f"metric id '{metric_id}' is configured twice; give the second "
                "entry its own `id:` so the two do not report into the same keys"
            )

        kwargs = {**spec.defaults, **entry}
        kwargs.setdefault("normalize", list(normalize) if normalize else None)
        logging.info("Loading %s evaluation as '%s'...", name, metric_id)
        modules[metric_id] = {
            "name": name,
            "spec": spec,
            "config": kwargs,
            "module": spec.metric,
            "state": spec.setup(metric_id=metric_id, **kwargs),
        }
        logging.info("Initiate %s evaluation successfully.", metric_id)
    return modules


def load_session_metrics(
    metrics_config: Sequence[Dict[str, Any]],
    normalize: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build the session-tier metrics.

    None exist yet; :mod:`splet.session_metrics` documents the contract they
    will follow.
    """
    return load_metrics(metrics_config, tier="session", normalize=normalize)


def load_corpus_metrics(
    metrics_config: Sequence[Dict[str, Any]],
    normalize: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build the corpus-tier metrics.

    None exist yet; :mod:`splet.corpus_metrics` documents the contract they
    will follow.
    """
    return load_metrics(metrics_config, tier="corpus", normalize=normalize)


def validate_requirements(
    metrics: Mapping[str, Mapping[str, Any]], gt: Optional[Mapping[str, Any]]
) -> None:
    """Refuse to measure unless every metric's declared requirements are met.

    Checked once, before any item is measured, so a missing reference fails
    up front with the metric named rather than as a ``TypeError`` from
    inside one metric half way through.

    Args:
        metrics: From :func:`load_metrics`.
        gt: The reference side, or None when none was given.

    Raises:
        ValueError: Naming the metric and the requirement it cannot be given.
        NotImplementedError: For ``timestamps`` and ``speakers``, which only
            the session tier can check; it checks them when it exists.
    """
    for metric_id, module in metrics.items():
        for requirement in module["spec"].requires:
            if requirement == "reference" and gt is None:
                raise ValueError(
                    f"metric '{metric_id}' requires a reference; pass --gt"
                )
            if requirement != "reference":
                raise NotImplementedError(
                    f"metric '{metric_id}' requires {requirement}; the session "
                    "tier that checks it does not exist yet"
                )


def require_matching_keys(
    pred_texts: Mapping[str, Any], gt_texts: Optional[Mapping[str, Any]]
) -> None:
    """Refuse to measure unless every utterance is on both sides.

    Every tier calls this before it looks at any text. A hypothesis without a
    reference, or a reference without a hypothesis, is a data problem, and
    dropping either side would change the denominator and improve the result
    without anyone asking for it. An utterance the system produced nothing for
    is not a missing hypothesis: it appears in the hypothesis file as its ID
    with an empty text, and every reference word counts as deleted.

    Args:
        pred_texts: Utterance id to hypothesis text.
        gt_texts: Utterance id to reference text, or None when the metrics
            need no reference.

    Raises:
        KeyError: Naming the first offending utterance on either side.
    """
    if gt_texts is None:
        return
    missing = [key for key in gt_texts if key not in pred_texts]
    if missing:
        more = f" and {len(missing) - 1} more" if len(missing) > 1 else ""
        raise KeyError(
            f"no hypothesis for reference '{missing[0]}'{more}; an utterance the "
            "system produced nothing for must still appear in the hypothesis "
            "file with an empty text"
        )
    extra = [key for key in pred_texts if key not in gt_texts]
    if extra:
        more = f" and {len(extra) - 1} more" if len(extra) > 1 else ""
        raise KeyError(f"no reference for hypothesis '{extra[0]}'{more}")


def measure_utterances(
    pred_texts: Dict[str, str],
    metrics: Dict[str, Dict[str, Any]],
    gt_texts: Optional[Dict[str, str]] = None,
    output_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Measure every utterance with every utterance-tier metric.

    Args:
        pred_texts: Utterance id to hypothesis text.
        metrics: From :func:`load_metrics`.
        gt_texts: Utterance id to reference text.
        output_file: Where to write one JSON object per utterance. The file
            is written as measurement proceeds, so a crash halfway through still
            leaves the results computed so far.

    Returns:
        One result dict per utterance, each carrying its ``key``.

    Raises:
        ValueError: If a metric requires a reference and none was given
            (:func:`validate_requirements`).
        KeyError: If a hypothesis has no reference, or a reference has no
            hypothesis. Measuring the utterances that happen to match and
            reporting the average would silently answer a different question
            than the one asked. An utterance the system produced nothing for
            is not a missing hypothesis: it appears in the hypothesis file with
            an empty text and every reference word counts as deleted.
    """
    validate_requirements(metrics, gt_texts)
    require_matching_keys(pred_texts, gt_texts)
    handle = open(output_file, "w", encoding="utf-8") if output_file else None
    try:
        results = []
        for key in pred_texts:
            utt_result: Dict[str, Any] = {"key": key}
            for metric_id, module in metrics.items():
                utt_result.update(
                    module["module"](
                        module["state"],
                        pred_texts[key],
                        gt_texts[key] if gt_texts is not None else None,
                    )
                )
            results.append(utt_result)
            if handle is not None:
                handle.write(json.dumps(utt_result, ensure_ascii=False) + "\n")
        return results
    finally:
        if handle is not None:
            handle.close()


def measure_batch(
    pred_texts: Sequence[str],
    gt_texts: Optional[Sequence[str]],
    metrics: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Measure parallel sequences of hypotheses and references, no ids.

    The entry point for a training loop's validation step: it has the
    decoded hypotheses and their references as two lists for one batch, no
    utterance ids and no files. Each item is measured by every loaded
    metric; the results are fed to :class:`~splet.summary.Accumulator`.

    Only the position pairs the two sides, so :func:`require_matching_keys`
    has nothing to check here; a length mismatch is the one error it can
    catch, and it does.

    Args:
        pred_texts: Hypotheses, one per item.
        gt_texts: References in the same order, or None when no loaded
            metric requires one.
        metrics: From :func:`load_metrics`.

    Returns:
        One result dict per item, in order, without a ``key``.

    Raises:
        ValueError: If the two sides differ in length, or a metric requires
            a reference and ``gt_texts`` is None.
    """
    if gt_texts is not None and len(gt_texts) != len(pred_texts):
        raise ValueError(f"{len(pred_texts)} hypotheses but {len(gt_texts)} references")
    if gt_texts is None:
        for metric_id, module in metrics.items():
            if "reference" in module["spec"].requires:
                raise ValueError(f"metric '{metric_id}' requires a reference")
    results = []
    for index, pred in enumerate(pred_texts):
        gt = gt_texts[index] if gt_texts is not None else None
        item_result: Dict[str, Any] = {}
        for module in metrics.values():
            item_result.update(module["module"](module["state"], pred, gt))
        results.append(item_result)
    return results


def measure_sessions(*args, **kwargs):
    """Measure every session. No session-tier metric exists yet.

    When it exists it starts with :func:`validate_requirements` and
    :func:`require_matching_keys`, as the utterance tier does.

    Raises:
        NotImplementedError: Always. See :mod:`splet.session_metrics` for the
            contract this loop will follow.
    """
    raise NotImplementedError(
        "the session tier has no metrics yet; see splet/session_metrics"
    )


def measure_corpus(*args, **kwargs):
    """Measure the corpus as a whole. No corpus-tier metric exists yet.

    When it exists it starts with :func:`validate_requirements` and
    :func:`require_matching_keys`, as the utterance tier does.

    Raises:
        NotImplementedError: Always. See :mod:`splet.corpus_metrics` for the
            contract this loop will follow.
    """
    raise NotImplementedError(
        "the corpus tier has no metrics yet; see splet/corpus_metrics"
    )
