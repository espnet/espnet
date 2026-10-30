#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Edit-distance alignment, the primitive every error rate is built on.

Two things vary independently here: what an edit costs, and which
implementation searches for the cheapest sequence of them.

Costs
-----

``sclite``
    The weights SCTK's ``sclite`` uses, from ``src/sclite/word.c``: a
    deletion or insertion costs 3, a substitution 4, a correct token 0, and
    an optionally deletable token 2. This is the default, because a WER
    reported by ESPnet should be the WER ``sclite`` reports for the same
    files.
``unit``
    Every edit costs 1, which is plain Levenshtein distance and what jiwer
    and ``rapidfuzz`` compute.

The two do not merely split the errors differently, they can disagree on
how many there are. ``sclite`` minimises cost, not edit count, so it takes
five insertions and deletions (15) over four substitutions (16) whenever
the alignment allows: its total is an upper bound on the minimum edit
distance, never below it. On clean speech the two agree; on a corpus with
a high character error rate they do not.

Backends
--------

``python``
    A dependency-free dynamic program that records the choice it made in
    each cell and reads the alignment back off that table. It supports
    every cost model and is the definition of what SPLET means by an
    alignment.
``rapidfuzz``
    ``rapidfuzz.distance.Levenshtein.editops``, a C++ Hirschberg alignment.
    It matters because the Python table is ``len(ref) * len(hyp)`` cells:
    fine for a 20-word utterance, impossible for the 5000-word per-speaker
    transcript that cpWER concatenates out of a one-hour meeting. It
    computes unit-cost editops and cannot express a weighted model, so it
    is available only with ``costs="unit"``.

Under ``costs="unit"`` the two backends agree on the error count but not
always on the S/D/I split, because ties at the minimum are common and the
two break them differently. The default backend is therefore ``"python"``
rather than whichever is installed: a results table reports S/D/I, and a
number that changes with the contents of the environment is not a
reproducible number.

Known differences from sclite
-----------------------------

Two of sclite's behaviours are deliberately absent, and a corpus that uses
either will not match it:

* ``-F``, fragment matching. sclite swaps in a comparator that counts a
  hyphen-marked fragment as correct when it is a prefix or suffix of the
  reference word. SPLET compares tokens for equality.
* The ``@`` token, which sclite treats as a null with its own costs (0.001
  to insert or delete, 1.0 to substitute for another ``@``) and leaves out
  of the error tally. SPLET scores ``@`` as an ordinary token.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

from splet.summary import error_rate_from_counts

try:
    from rapidfuzz.distance import Levenshtein as _RapidfuzzLevenshtein
except ImportError:  # pragma: no cover - exercised by the no-extras CI job
    _RapidfuzzLevenshtein = None

# ("hit" | "sub" | "del" | "ins", reference token, hypothesis token). The
# token is None on the side that has nothing at this position.
Operation = Tuple[str, Optional[str], Optional[str]]


@dataclass(frozen=True)
class CostModel:
    """What each edit costs the alignment search.

    Attributes:
        substitution: Cost of aligning two tokens that differ.
        insertion: Cost of a hypothesis token with no reference token.
        deletion: Cost of a reference token with no hypothesis token.
        correct: Cost of aligning two tokens that are equal.
        optional_deletion: Cost of deleting a reference token, or inserting
            a hypothesis token, that is marked optional. sclite puts this
            between ``correct`` and ``deletion`` so that an optional token
            is dropped in preference to a real edit.
    """

    substitution: float
    insertion: float
    deletion: float
    correct: float = 0.0
    optional_deletion: float = 1.0


# src/sclite/word.c:11-17 states these as the cost structure, and
# wwd_WORD returns them.
COST_CHOICES: Dict[str, CostModel] = {
    "sclite": CostModel(
        substitution=4.0,
        insertion=3.0,
        deletion=3.0,
        correct=0.0,
        optional_deletion=2.0,
    ),
    "unit": CostModel(
        substitution=1.0,
        insertion=1.0,
        deletion=1.0,
        correct=0.0,
        optional_deletion=1.0,
    ),
}


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

        The zero-denominator policy is :func:`splet.summary.error_rate_from_counts`,
        the same function the corpus summary uses, so one utterance and the
        corpus it belongs to cannot disagree about an empty reference.
        """
        return error_rate_from_counts(self.errors, self.ref_len)

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

# Backtrace directions, in the order ties are broken.
_SUB, _DEL, _INS = 0, 1, 2


def resolve_costs(costs: Union[str, CostModel]) -> CostModel:
    """Turn a requested cost model into the one that will actually be used.

    Args:
        costs: A key of :data:`COST_CHOICES`, or a :class:`CostModel`.

    Returns:
        The cost model.

    Raises:
        ValueError: If ``costs`` is a string that names no known model.
    """
    if isinstance(costs, CostModel):
        return costs
    try:
        return COST_CHOICES[costs]
    except KeyError:
        raise ValueError(
            f"unknown cost model '{costs}': expected one of "
            f"{', '.join(sorted(COST_CHOICES))}"
        ) from None


def resolve_backend(backend: str, costs: Optional[Union[str, CostModel]] = None) -> str:
    """Turn a requested backend into the one that will actually run.

    A metric calls this once at setup and records the result, so a saved
    S/D/I split names the implementation that produced it rather than
    ``"auto"``, which means different things in different environments.

    Args:
        backend: ``"auto"``, ``"python"`` or ``"rapidfuzz"``.
        costs: The cost model the alignment will use, if it is known.
            ``"auto"`` will not choose rapidfuzz for a weighted model,
            because rapidfuzz cannot compute one.

    Returns:
        ``"python"`` or ``"rapidfuzz"``.

    Raises:
        ValueError: If ``backend`` is not one of the three accepted values,
            or ``costs`` names no known model.
        ImportError: If ``backend="rapidfuzz"`` and it is not installed.
    """
    if backend == "auto":
        if _RapidfuzzLevenshtein is None:
            return "python"
        if costs is not None and resolve_costs(costs) != COST_CHOICES["unit"]:
            return "python"
        return "rapidfuzz"
    if backend == "rapidfuzz":
        if _RapidfuzzLevenshtein is None:
            raise ImportError(
                "backend='rapidfuzz' requires rapidfuzz: pip install rapidfuzz"
            )
        return backend
    if backend == "python":
        return backend
    raise ValueError(
        f"unknown backend '{backend}': expected 'auto', 'python' or 'rapidfuzz'"
    )


def is_optional(token: str) -> bool:
    """Return whether a token is marked optional.

    sclite marks an optional token by surrounding it with parentheses in the
    trn, stm or ctm record; see ``doc/sclite.htm``. ``"()"`` is not a marked
    empty token, it is the two-character token it looks like.

    Args:
        token: One reference or hypothesis token.

    Returns:
        True if the token carries the marking.
    """
    return len(token) > 2 and token.startswith("(") and token.endswith(")")


def _bare(token: str) -> str:
    """Return a token without its optionality marking."""
    return token[1:-1] if is_optional(token) else token


def levenshtein_alignment(
    ref: Sequence[str],
    hyp: Sequence[str],
    backend: str = "python",
    costs: Union[str, CostModel] = "sclite",
    optional_deletion_is_correct: bool = False,
) -> AlignmentResult:
    """Align a hypothesis against a reference.

    Args:
        ref: Reference tokens.
        hyp: Hypothesis tokens.
        backend: ``"python"`` (the default) is the reference implementation
            and always produces the same S/D/I split. ``"rapidfuzz"`` is the
            linear-memory one, for inputs too long for a full table, and
            requires ``costs="unit"``. ``"auto"`` prefers rapidfuzz when it
            is installed and can compute the requested costs, and is
            therefore only appropriate when the totals are what matter.
        costs: ``"sclite"`` (the default), ``"unit"``, or a
            :class:`CostModel`.
        optional_deletion_is_correct: sclite's ``-D``. Score a deleted
            optional reference token, or an inserted optional hypothesis
            token, as correct rather than as an error. The marking changes
            the alignment either way, because it lowers the cost; this
            changes only how the result is counted.

    Returns:
        The alignment, its counts, and the operations that produced them.

    Raises:
        ValueError: If ``backend`` or ``costs`` is not recognised, or if a
            weighted cost model is asked of the rapidfuzz backend.
        ImportError: If ``backend="rapidfuzz"`` and it is not installed.
        MemoryError: If the pure Python backend is asked for a table larger
            than it is willing to build. Install rapidfuzz for these.
    """
    ref = list(ref)
    hyp = list(hyp)

    model = resolve_costs(costs)
    backend = resolve_backend(backend, model)

    if backend == "rapidfuzz":
        if model != COST_CHOICES["unit"]:
            raise ValueError(
                "backend='rapidfuzz' computes unit-cost editops and cannot "
                "express a weighted cost model; pass costs='unit', or use "
                "backend='python'"
            )
        operations = _editops_rapidfuzz(ref, hyp)
    else:
        cells = (len(ref) + 1) * (len(hyp) + 1)
        if cells > _PYTHON_BACKEND_CELL_LIMIT:
            raise MemoryError(
                f"aligning {len(ref)} reference and {len(hyp)} hypothesis "
                f"tokens needs a {cells:,}-cell table. Pass "
                "backend='rapidfuzz' (pip install rapidfuzz), which aligns "
                "this in linear memory, or segment the input."
            )
        operations = _editops_python(ref, hyp, model, optional_deletion_is_correct)

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


def _editops_python(
    ref: List[str],
    hyp: List[str],
    model: CostModel,
    optional_deletion_is_correct: bool,
) -> List[Operation]:
    """Align with a dynamic program over a weighted cost model.

    The table is ``(len(ref) + 1) x (len(hyp) + 1)``; cell ``(i, j)`` is the
    cost of turning the first ``i`` reference tokens into the first ``j``
    hypothesis tokens. Ties are broken substitution first, then deletion,
    then insertion, so that the output is deterministic.

    Each cell records the direction it came from rather than leaving the
    backtrace to recompute it. With non-unit costs those recomputed
    comparisons are between sums of floats, and an alignment that depends on
    whether 3.0 + 4.0 + 3.0 equals 10.0 exactly is not one to rely on.
    """
    n, m = len(ref), len(hyp)
    bare_ref = [_bare(token) for token in ref]
    bare_hyp = [_bare(token) for token in hyp]
    del_cost = [
        model.optional_deletion if is_optional(token) else model.deletion
        for token in ref
    ]
    ins_cost = [
        model.optional_deletion if is_optional(token) else model.insertion
        for token in hyp
    ]

    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    came_from = [[_SUB] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = cost[i - 1][0] + del_cost[i - 1]
        came_from[i][0] = _DEL
    for j in range(1, m + 1):
        cost[0][j] = cost[0][j - 1] + ins_cost[j - 1]
        came_from[0][j] = _INS

    for i in range(1, n + 1):
        ref_token = bare_ref[i - 1]
        row, previous_row = cost[i], cost[i - 1]
        choice_row = came_from[i]
        row_del_cost = del_cost[i - 1]
        for j in range(1, m + 1):
            best = previous_row[j - 1] + (
                model.correct if ref_token == bare_hyp[j - 1] else model.substitution
            )
            direction = _SUB
            delete = previous_row[j] + row_del_cost
            if delete < best:
                best, direction = delete, _DEL
            insert = row[j - 1] + ins_cost[j - 1]
            if insert < best:
                best, direction = insert, _INS
            row[j] = best
            choice_row[j] = direction

    operations: List[Operation] = []
    i, j = n, m
    while i > 0 or j > 0:
        direction = came_from[i][j]
        if direction == _SUB:
            equal = bare_ref[i - 1] == bare_hyp[j - 1]
            operations.append(("hit" if equal else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif direction == _DEL:
            correct = optional_deletion_is_correct and is_optional(ref[i - 1])
            operations.append(("hit" if correct else "del", ref[i - 1], None))
            i -= 1
        else:
            correct = optional_deletion_is_correct and is_optional(hyp[j - 1])
            operations.append(("hit" if correct else "ins", None, hyp[j - 1]))
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
