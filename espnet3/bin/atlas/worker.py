"""Internal entry point for graph-node worker jobs."""

from __future__ import annotations

import argparse
from pathlib import Path

from espnet3.bin.atlas.command import _load_runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--recipe-dir", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trial-id")
    parser.add_argument("--attempt-id", default="attempt_000000")
    args = parser.parse_args()
    runtime = _load_runtime(Path(args.study_dir).resolve(), Path(args.recipe_dir).resolve())
    if runtime.gpu_self_controller_enabled() and runtime.graph.nodes[args.node].execution == "submission":
        result = runtime.run_gpu_self_controller(
            node_name=args.node,
            run_id=args.run_id,
            trial_id=args.trial_id,
            attempt_id=args.attempt_id,
        )
    else:
        result = runtime.run_node(
            node_name=args.node,
            run_id=args.run_id,
            trial_id=args.trial_id,
            attempt_id=args.attempt_id,
            controller_mode=False,
        )
    return 0 if result.status not in {"failure", "timeout"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
