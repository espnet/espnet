# espnet3/utils/stages.py
from __future__ import annotations

from typing import Any, Iterable, List, Sequence


def resolve_stages(
    requested: Sequence[str],
    stages: Sequence[str],
) -> List[str]:
    if "all" in requested:
        return list(stages)
    requested_set = set(requested)
    return [s for s in stages if s in requested_set]


def run_stages(
    system: Any,
    stages_to_run: Iterable[str],
    *,
    dry_run: bool = False,
) -> None:
    for stage in stages_to_run:
        fn = getattr(system, stage, None)
        if fn is None:
            raise AttributeError(f"System has no stage method: {stage}")

        if dry_run:
            print(f"[DRY RUN] would run stage: {stage}")
            continue

        print(f"=== Running stage: {stage} ===")
        try:
            fn()
        except TypeError as e:
            raise TypeError(
                f"Stage '{stage}' does not accept CLI arguments; "
                "put all settings in the YAML config."
            ) from e
