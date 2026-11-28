#!/usr/bin/env python3
"""Assign dump directories (stdin, one per line) to N tasks by their encoder work.

    printf '%s\\n' dump/... | python3 local/v2/assign_tasks.py N > list

A dump's work is the number of encoder buffers its alignment needs: with a
clips file one buffer per started 22 s of every clip (each clip is aligned on
its own, ctc_batch.BufferGeometry: 30 s buffers with 4 s of context on each
side), else per started 22 s of every recording (utt2num_samples). The dumps
go, heaviest first, to the task with the least work so far. Prints "<task>
<dump>" lines (tasks from 1) and the work per task on stderr.
"""

import math
import sys
from pathlib import Path

CHUNK = 22 * 16000  # samples of posteriors one 30 s buffer gives


def work(d: Path) -> int:
    if (d / "clips").exists():
        lengths = [
            int(x.split("\t")[3]) - int(x.split("\t")[2])
            for x in (d / "clips").read_text().splitlines()
        ]
    else:
        lengths = [int(x.split()[1]) for x in (d / "utt2num_samples").open()]
    return sum(max(1, math.ceil(n / CHUNK)) for n in lengths)


def main():
    n = int(sys.argv[1])
    dumps = [Path(x) for x in sys.stdin.read().split()]
    loads = [0] * n
    out = []
    for w, d in sorted(((work(d), d) for d in dumps), key=lambda x: (-x[0], str(x[1]))):
        k = loads.index(min(loads))
        loads[k] += w
        out.append((k + 1, d))
    for k, d in sorted(out, key=lambda x: (x[0], str(x[1]))):
        print(k, d)
    print(
        f"{len(dumps)} dumps, {sum(loads)} buffers over {n} tasks: "
        f"max {max(loads)}, min {min(loads)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
