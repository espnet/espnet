"""Study reports: the ``trials.csv`` sheet."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from espnet3.bin.atlas.core.utils import flatten_dict, write_text_atomically


def sync_trials_csv(state, study_dir: Path, study_id: str) -> None:
    """Regenerate ``study_dir/trials.csv`` from a study's trial records.

    Rewrites the whole file from the current set of trials, with one column
    per distinct ``config_patch`` key and one per distinct flattened metric
    key seen across all trials, so the sheet grows as new hyperparameters or
    metrics appear.

    Args:
        state: ``StateStore`` to read trial records from.
        study_dir: Study directory; the CSV is written to
            ``study_dir/trials.csv``.
        study_id: Study whose trials to include.

    Returns:
        None.

    Example:
        >>> sync_trials_csv(state, Path("exp/atlas/my_study"), "my_study")
    """
    trials = state.list_trials(study_id)
    all_patch_keys = sorted({k for trial in trials for k in trial.config_patch})
    all_metric_keys = sorted(
        {
            key
            for trial in trials
            for key in flatten_dict(trial.metrics if trial.metrics else {}).keys()
        }
    )
    rows = []
    for trial in trials:
        flat_metrics = flatten_dict(trial.metrics if trial.metrics else {})
        row = {
            "trial_id": trial.trial_id,
            "status": trial.status,
            "parent_trial_id": trial.parent_trial_id or "",
            "score": trial.score if trial.score is not None else "",
            "score_name": trial.score_name or "",
            "decision": trial.decision or "",
            "resolved_config_path": (
                str(trial.resolved_config_path) if trial.resolved_config_path else ""
            ),
            "rationale": trial.rationale,
            "expected_effect": trial.expected_effect,
            "risk": trial.risk,
            "attempt_count": trial.attempt_count,
            "created_at": trial.created_at,
            "updated_at": trial.updated_at,
            "reflection_summary": trial.reflection,
            "next_suggestion": trial.next_suggestion,
        }
        for key in all_patch_keys:
            row[f"patch.{key}"] = trial.config_patch.get(key, "")
        for key in all_metric_keys:
            row[f"metric.{key}"] = flat_metrics.get(key, "")
        rows.append(row)
    fieldnames = (
        [
            "trial_id",
            "status",
            "parent_trial_id",
            "score",
            "score_name",
            "decision",
            "resolved_config_path",
        ]
        + [f"patch.{key}" for key in all_patch_keys]
        + ["rationale", "expected_effect", "risk"]
        + [f"metric.{key}" for key in all_metric_keys]
        + [
            "attempt_count",
            "created_at",
            "updated_at",
            "reflection_summary",
            "next_suggestion",
        ]
    )
    path = study_dir / "trials.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    write_text_atomically(path, buffer.getvalue())
