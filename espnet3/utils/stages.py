# espnet3/utils/stages.py
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence


def _parse_scalar(value: str) -> Any:
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def resolve_stages(
    requested: Sequence[str],
    stages: Sequence[str],
) -> List[str]:
    if "all" in requested:
        return list(stages)
    requested_set = set(requested)
    return [s for s in stages if s in requested_set]


def parse_stage_unknown_args(
    unknown: Sequence[str],
    valid_stages: Sequence[str],
    *,
    error: Callable[[str], None],
) -> Dict[str, Dict[str, Any]]:
    valid = set(valid_stages)
    result: Dict[str, Dict[str, Any]] = {}

    unknown = list(unknown)
    i = 0
    bad_tokens: List[str] = []

    while i < len(unknown):
        token = unknown[i]

        # Raise error without unknown argument like stage.*
        if not token.startswith("--stage."):
            bad_tokens.append(token)
            i += 1
            continue

        # "--stage.train.foo=1" format or "--stage.train.foo 1"
        if "=" in token:
            opt, value_str = token.split("=", 1)
            i += 1
        else:
            opt = token
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("-"):
                value_str = unknown[i + 1]
                i += 2
            else:
                value_str = "true"
                i += 1

        opt = opt.lstrip("-")
        parts = opt.split(".")
        if len(parts) < 3 or parts[0] != "stage":
            bad_tokens.append(token)
            continue

        _, stage_name, *key_parts = parts
        if stage_name not in valid:
            error(
                f"Unknown stage in stage-arg: {stage_name!r} "
                f"(valid stages: {sorted(valid)})"
            )

        key = ".".join(key_parts)
        value = _parse_scalar(value_str)

        stage_dict = result.setdefault(stage_name, {})
        stage_dict[key] = value

    if bad_tokens:
        error(f"unrecognized arguments: {' '.join(bad_tokens)}")

    return result


def run_stages(
    system: Any,
    stages_to_run: Iterable[str],
    *,
    dry_run: bool = False,
    stage_args: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    stage_args = stage_args or {}

    for stage in stages_to_run:
        fn = getattr(system, stage, None)
        if fn is None:
            raise AttributeError(f"System has no stage method: {stage}")

        kwargs = dict(stage_args.get(stage, {}))

        if dry_run:
            if kwargs:
                print(f"[DRY RUN] would run stage: {stage} with args: {kwargs}")
            else:
                print(f"[DRY RUN] would run stage: {stage}")
            continue

        print(f"=== Running stage: {stage} ===")
        fn(**kwargs)
