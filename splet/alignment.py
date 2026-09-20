#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Edit-distance alignment, the primitive every error rate is built on.

Two backends produce the same alignment:

``python``
    A dependency-free dynamic program with a backtrace. It is the reference
    implementation: readable, exercised by the tests, and the definition of
    what SPLET means by an alignment.
``rapidfuzz``
    ``rapidfuzz.distance.Levenshtein.editops``, used when the package is
    installed. It is a C++ Hirschberg alignment, which matters because the
    reference implementation allocates a table of ``len(ref) * len(hyp)``
    cells: fine for a 20-word utterance, impossible for the 5000-word
    per-speaker transcript that cpWER concatenates out of a one-hour meeting.

The two agree on the total number of errors, and they do not always agree
on how that total splits into substitutions, deletions and insertions. When
several alignments share the minimum cost -- which is common -- the split
depends on how ties are broken, and the two implementations break them
differently. The default backend is therefore ``"python"`` rather than
whichever is installed: a results table reports S/D/I, and a number that
changes with the contents of the environment is not a reproducible number.
Pass ``backend="rapidfuzz"`` deliberately, for inputs the reference
implementation cannot hold.

Both use unit costs. sclite's DP instead weights insertion and deletion at 3
and substitution at 4, which is a third tie-break again. Reproducing
sclite's exact triple is a validation task that has not been done yet; see
espnet/espnet#6760. Until it is, do not claim these S/D/I figures match
sclite's -- the totals should, the split may not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

try:
    from rapidfuzz.distance import Levenshtein as _RapidfuzzLevenshtein
except ImportError:  # pragma: no cover - exercised by the no-extras CI job
    _RapidfuzzLevenshtein = None

# ("hit" | "sub" | "del" | "ins", reference token, hypothesis token). The
# token is None on the side that has nothing at this position.
Operation = Tuple[str, Optional[str], Optional[str]]


@dataclass
class AlignmentResult:
    """The outcome of aligning one hypothesis against one reference."""

    hits: int
    substitutions: int
    deletions: int
    insertions: int
    ref_len: int
    hyp_len: int
    operations: List[Operation]

    @property
    def errors(self) -> int:
        """Return the total error count: substitutions, deletions, insertions."""
        return self.substitutions + self.deletions + self.insertions

    @property
    def error_rate(self) -> float:
        """Return errors divided by reference length.

        An empty reference with a non-empty hypothesis has an undefined rate;
        this returns 1.0 for it, matching what every toolkit does in practice,
        and 0.0 when both sides are empty.
        """
        if self.ref_len == 0:
            return 1.0 if self.hyp_len > 0 else 0.0
        return self.errors / self.ref_len

    def to_string(self, width: int = 0) -> str:
        """Render the alignment as three aligned lines: REF, HYP, and ops."""
        ref_cells, hyp_cells, op_cells = [], [], []
        for op, ref_token, hyp_token in self.operations:
            ref_token = ref_token if ref_token is not None else "*"
            hyp_token = hyp_token if hyp_token is not None else "*"
            cell = max(len(ref_token), len(hyp_token), width)
            ref_cells.append(ref_token.ljust(cell))
            hyp_cells.append(hyp_token.ljust(cell))
            op_cells.append(_OP_CODE[op].ljust(cell))
        return "\n".join(
            [
                "REF: " + " ".join(ref_cells),
                "HYP: " + " ".join(hyp_cells),
                "OP:  " + " ".join(op_cells),
            ]
        )


_OP_CODE = {"hit": "C", "sub": "S", "del": "D", "ins": "I"}

# Above this many cells the pure-Python table is not worth building. The
# figure is where a table stops fitting comfortably in memory and time
# (~4M cells is a second or two and tens of MB), not a hard limit of the
# algorithm.
_PYTHON_BACKEND_CELL_LIMIT = 4_000_000


def levenshtein_alignment(
    ref: Sequence[str],
    hyp: Sequence[str],
    backend: str = "python",
) -> AlignmentResult:
    """Align a hypothesis against a reference.

    Args:
        ref: Reference tokens.
        hyp: Hypothesis tokens.
        backend: ``"python"`` (the default) is the reference implementation
            and always produces the same S/D/I split. ``"rapidfuzz"`` is the
            linear-memory one, for inputs too long for a full table.
            ``"auto"`` prefers rapidfuzz when it is installed, and is
            therefore only appropriate when the totals are what matter.

    Returns:
        The alignment, its counts, and the operations that produced them.

    Raises:
        ValueError: If ``backend`` is not one of the three accepted values.
        ImportError: If ``backend="rapidfuzz"`` and it is not installed.
        MemoryError: If the pure Python backend is asked for a table larger
            than it is willing to build. Install rapidfuzz for these.
    """
    ref = list(ref)
    hyp = list(hyp)

    if backend == "auto":
        backend = "rapidfuzz" if _RapidfuzzLevenshtein is not None else "python"
    if backend == "rapidfuzz":
        if _RapidfuzzLevenshtein is None:
            raise ImportError(
                "backend='rapidfuzz' requires rapidfuzz: pip install rapidfuzz"
            )
        operations = _editops_rapidfuzz(ref, hyp)
    elif backend == "python":
        cells = (len(ref) + 1) * (len(hyp) + 1)
        if cells > _PYTHON_BACKEND_CELL_LIMIT:
            raise MemoryError(
                f"aligning {len(ref)} reference and {len(hyp)} hypothesis "
                f"tokens needs a {cells:,}-cell table. Pass "
                "backend='rapidfuzz' (pip install rapidfuzz), which aligns "
                "this in linear memory, or segment the input."
            )
        operations = _editops_python(ref, hyp)
    else:
        raise ValueError(
            f"unknown backend '{backend}': expected 'auto', 'python' or 'rapidfuzz'"
        )

    counts = {"hit": 0, "sub": 0, "del": 0, "ins": 0}
    for op, _, _ in operations:
        counts[op] += 1
    return AlignmentResult(
        hits=counts["hit"],
        substitutions=counts["sub"],
        deletions=counts["del"],
        insertions=counts["ins"],
        ref_len=len(ref),
        hyp_len=len(hyp),
        operations=operations,
    )


def _editops_python(ref: List[str], hyp: List[str]) -> List[Operation]:
    """Align with a dynamic program and a backtrace.

    The table is ``(len(ref) + 1) x (len(hyp) + 1)``; cell ``(i, j)`` is the
    cost of turning the first ``i`` reference tokens into the first ``j``
    hypothesis tokens. Ties are broken substitution first, then deletion,
    then insertion, so that the output is deterministic.
    """
    n, m = len(ref), len(hyp)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        ref_token = ref[i - 1]
        row, previous_row = cost[i], cost[i - 1]
        for j in range(1, m + 1):
            substitute = previous_row[j - 1] + (ref_token != hyp[j - 1])
            delete = previous_row[j] + 1
            insert = row[j - 1] + 1
            row[j] = min(substitute, delete, insert)

    operations: List[Operation] = []
    i, j = n, m
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1])
        ):
            op = "hit" if ref[i - 1] == hyp[j - 1] else "sub"
            operations.append((op, ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            operations.append(("del", ref[i - 1], None))
            i -= 1
        else:
            operations.append(("ins", None, hyp[j - 1]))
            j -= 1
    operations.reverse()
    return operations


def _editops_rapidfuzz(ref: List[str], hyp: List[str]) -> List[Operation]:
    """Align with rapidfuzz and expand its edit script into operations.

    rapidfuzz reports only the edits. The positions in between are hits, and
    they have to be filled back in, because a hit count is what an error rate
    is checked against.
    """
    operations: List[Operation] = []
    i = j = 0
    for edit in _RapidfuzzLevenshtein.editops(ref, hyp):
        while i < edit.src_pos and j < edit.dest_pos:
            operations.append(("hit", ref[i], hyp[j]))
            i, j = i + 1, j + 1
        if edit.tag == "replace":
            operations.append(("sub", ref[i], hyp[j]))
            i, j = i + 1, j + 1
        elif edit.tag == "delete":
            operations.append(("del", ref[i], None))
            i += 1
        elif edit.tag == "insert":
            operations.append(("ins", None, hyp[j]))
            j += 1
        else:  # pragma: no cover - rapidfuzz emits no other tags
            raise ValueError(f"unexpected rapidfuzz edit tag: {edit.tag}")
    while i < len(ref) and j < len(hyp):
        operations.append(("hit", ref[i], hyp[j]))
        i, j = i + 1, j + 1
    return operations
