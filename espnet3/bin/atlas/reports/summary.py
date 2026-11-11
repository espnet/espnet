"""Study summary rendering."""

from __future__ import annotations

from pathlib import Path


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
