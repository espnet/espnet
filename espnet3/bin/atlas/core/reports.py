"""Study reports: the ``trials.csv`` sheet and the final ``summary.md``."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from espnet3.bin.atlas.core.serialization import atomic_write_text, flatten_dict


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
            "resolved_config_path": str(trial.resolved_config_path) if trial.resolved_config_path else "",
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
        + ["attempt_count", "created_at", "updated_at", "reflection_summary", "next_suggestion"]
    )
    path = study_dir / "trials.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    atomic_write_text(path, buffer.getvalue())


def write_summary(study_dir: Path, objective: str, trials, best_trial) -> Path:
    """Write ``study_dir/summary.md`` with trial counts, the best trial, and suggestions.

    Args:
        study_dir: Study directory; the summary is written to
            ``study_dir/summary.md``.
        objective: Study objective text (usually ``program.md``'s content).
        trials: Iterable of ``Trial`` records to summarize.
        best_trial: The current best ``Trial``, or ``None`` if no trial has
            been accepted yet.

    Returns:
        Path: Path to the written ``summary.md`` file.

    Example:
        >>> write_summary(Path("exp/atlas/my_study"), "Minimize WER", trials, best)
        PosixPath('exp/atlas/my_study/summary.md')
    """
    accepted = sum(1 for trial in trials if trial.status == "accepted")
    rejected = sum(1 for trial in trials if trial.status == "rejected")
    failed = sum(1 for trial in trials if trial.status in {"failed", "timeout"})
    lines = [
        "# Atlas Summary",
        "",
        "## Objective",
        objective.strip() or "(empty)",
        "",
        "## Counts",
        f"- Trials: {len(trials)}",
        f"- Accepted: {accepted}",
        f"- Rejected: {rejected}",
        f"- Failed: {failed}",
        "",
    ]
    if best_trial is not None:
        lines.extend(
            [
                "## Best Trial",
                f"- Trial: {best_trial.trial_id}",
                f"- Score: {best_trial.score} ({best_trial.score_name})",
                f"- Decision: {best_trial.decision or ''}",
                "",
            ]
        )
    lines.extend(
        [
            "## Next Suggestions",
            *[
                f"- {trial.trial_id}: {trial.next_suggestion}"
                for trial in trials
                if trial.next_suggestion
            ],
            "",
        ]
    )
    path = study_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
