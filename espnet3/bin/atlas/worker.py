"""Internal entry point for graph-node worker jobs."""

from __future__ import annotations

import argparse
from pathlib import Path

from espnet3.bin.atlas.command import _load_runtime


def main() -> int:
    """Run one graph node (or its GPU self-controller) as a submitted job.

    This is the command Slurm/local job submission runs: it parses the
    ``--study-dir``/``--node``/``--run-id`` (and related) arguments, rebuilds
    the study runtime, and executes either the node's GPU self-controller
    loop (when the node is ``execution: submission``) or the node itself.

    Args:
        None (arguments are read from ``sys.argv`` via ``argparse``).

    Returns:
        int: ``0`` on success, ``1`` if the node result status is
        ``"failure"`` or ``"timeout"``.

    Example:
        >>> # python -m espnet3.bin.atlas.worker --study-dir exp/atlas/s \\
        ... #     --recipe-dir . --node run_trial --run-id run_1
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--recipe-dir", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trial-id")
    parser.add_argument("--attempt-id", default="attempt_000000")
    args = parser.parse_args()
    runtime = _load_runtime(
        Path(args.study_dir).resolve(), Path(args.recipe_dir).resolve()
    )
    if runtime.graph.nodes[args.node].execution == "submission":
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
