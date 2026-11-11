"""Collect metrics for a completed trial."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from espnet3.bin.atlas.core.errors import MetricNotFoundError
from espnet3.bin.atlas.core.models import StageResult
from espnet3.bin.atlas.core.serialization import pattern_to_regex
from espnet3.bin.atlas.core.stage import AtlasStage


def _extract_from_json(path: Path, key: str) -> Any:
    data = json.loads(path.read_text(encoding="utf-8"))
    current: Any = data
    if isinstance(current, dict) and key in current:
        return current[key]
    parts = str(key).split(".")
    index = 0
    while index < len(parts):
        if not isinstance(current, dict):
            raise MetricNotFoundError(f"Missing JSON key '{key}' in {path}")
        for end in range(len(parts), index, -1):
            candidate = ".".join(parts[index:end])
            if candidate in current:
                current = current[candidate]
                index = end
                break
        else:
            raise MetricNotFoundError(f"Missing JSON key '{key}' in {path}")
    return current


def _extract_from_pattern(path: Path, pattern: str) -> float:
    regex = pattern_to_regex(pattern)
    match = regex.search(path.read_text(encoding="utf-8", errors="replace"))
    if match is None:
        raise MetricNotFoundError(f"Pattern '{pattern}' not found in {path}")
    return float(match.group("value"))


def _extract_metrics(trial_dir: Path, extraction_rules) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for rule in extraction_rules or []:
        rel_path = str(getattr(rule, "path", None) or rule.get("path"))
        matches = sorted(trial_dir.glob(rel_path))
        if not matches:
            continue
        path = matches[0]
        key = getattr(rule, "name", None) or rule.get("name")
        json_key = getattr(rule, "key", None) or rule.get("key")
        if json_key is not None:
            value = _extract_from_json(path, json_key)
            aggregate = getattr(rule, "aggregate", None) or rule.get("aggregate")
            if aggregate == "mean":
                sub_key = getattr(rule, "sub_key", None) or rule.get("sub_key") or "WER"
                if not isinstance(value, dict):
                    raise MetricNotFoundError(
                        f"aggregate:mean requires a dict value at key '{json_key}'"
                    )
                values = [
                    float(item[sub_key])
                    for item in value.values()
                    if isinstance(item, dict) and sub_key in item
                ]
                if not values:
                    raise MetricNotFoundError(
                        f"No '{sub_key}' sub-values found under '{json_key}'"
                    )
                value = sum(values) / len(values)
            metrics[key or str(json_key)] = value
        else:
            pattern = getattr(rule, "pattern", None) or rule.get("pattern")
            if pattern is not None:
                metrics[key or path.stem] = _extract_from_pattern(path, pattern)
    return metrics


def _resolve_primary_score(metrics: dict[str, Any], score_name: str) -> float:
    if score_name not in metrics:
        raise MetricNotFoundError(f"Score metric '{score_name}' was not extracted.")
    return float(metrics[score_name])


class CollectResultStage(AtlasStage):
    """Read metric files and update persistent trial state."""

    study_lock_name = "study_controller"

    def run(self, context) -> StageResult:
        trial = context.current_trial
        assert trial is not None
        metric_cfg = context.config.atlas.metric
        try:
            metrics = _extract_metrics(
                context.trial_dir(),
                metric_cfg.extraction,
            )
            score = _resolve_primary_score(metrics, str(metric_cfg.name))
        except Exception as exc:  # noqa: BLE001
            trial.status = "failed"
            context.state.update_trial(trial)
            return StageResult(status="failure", message=str(exc))

        trial.metrics = metrics
        trial.score = score
        trial.score_name = str(metric_cfg.name)
        if trial.status not in {"accepted", "rejected"}:
            trial.status = "completed"
        context.state.update_trial(trial)
        return StageResult(
            status="success",
            message=f"Collected score={score}",
            payload={"score": score},
        )
