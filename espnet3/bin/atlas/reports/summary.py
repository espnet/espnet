"""Study summary rendering."""

from __future__ import annotations

from pathlib import Path


def write_summary(study_dir: Path, objective: str, trials, best_trial) -> Path:
    """Write a study summary markdown file."""
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
