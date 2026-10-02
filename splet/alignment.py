#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Edit-distance alignment, the primitive every error rate is built on.

Two cost models are available, and the choice decides the S/D/I split:

``sclite`` (the default)
    What ``sclite`` itself uses: insertion and deletion 3, substitution 4,
    a correct token 0 (SCTK ``src/sclite/word.c``). Deleting a reference
    token marked optionally deletable -- written in parentheses, ``(um)`` --
    costs 2.
``unit``
    Plain Levenshtein: every edit costs 1.

They are not two spellings of the same thing. Under unit costs two
substitutions (2) tie with a deletion plus an insertion (2), so a
transposition is decided by the tie-break; under sclite's they cost 8
against 6, so it is decided outright and the answer is a deletion plus an
insertion. The totals agree, the split does not: on 4000 real utterances
the two differ by 6 substitutions at word level and 146 at character
level, while reporting the same WER to the digit.

The default is therefore ``sclite``, because matching the tool every egs2
recipe scores with is the point of this package. Ties within a cost model
are broken substitution first, then deletion, then insertion, which is the
order ``sclite``'s own dynamic program uses (``src/sclite/net_dp.c``).

Two backends compute the alignment:

``python``
    A dependency-free dynamic program with a backtrace. It is the reference
    implementation, it honours both cost models, and it is the definition of
    what SPLET means by an alignment.
``rapidfuzz``
    ``rapidfuzz.distance.Levenshtein.editops``. A C++ Hirschberg alignment,
    which matters because the reference implementation allocates a table of
    ``len(ref) * len(hyp)`` cells: fine for a 20-word utterance, impossible
    for the 5000-word per-speaker transcript that cpWER concatenates out of
    a one-hour meeting. It computes unit-cost edits and cannot express a
    weighted model, so it is available only with ``costs="unit"`` and says
    so rather than silently returning a different split.

Two things sclite does that this does not, listed because an absence nobody
wrote down is indistinguishable from a bug:

``@``
    sclite charges 0.001 to insert or delete it and excludes it from the
    tally altogether -- ``a @ b`` against ``a @ b`` scores 2 correct, not 3.
    That is a scoring-layer exclusion rather than a cost, ESPnet's tokenizers
    do not emit it, and a half-characterised special case is worse than a
    documented absence.
``-F``
    Score word fragments as correct. One recipe in egs2 passes it,
    ``dynamic_superb/ps2st1/local/score.sh``. Scoring that recipe through
    SPLET would report a number slightly worse than its published one, so
    either implement this or do not use SPLET for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from rapidfuzz.distance import Levenshtein as _RapidfuzzLevenshtein
except ImportError:  # pragma: no cover - exercised by the no-extras CI job
    _RapidfuzzLevenshtein = None

# ("hit" | "sub" | "del" | "ins", reference token, hypothesis token). The
# token is None on the side that has nothing at this position.
Operation = Tuple[str, Optional[str], Optional[str]]


@dataclass(frozen=True)
class CostModel:
    """What each edit costs the dynamic program.

    Attributes:
        name: The name this model is selected by.
        substitution: Cost of aligning two tokens that differ.
        deletion: Cost of a reference token with nothing opposite it.
        insertion: Cost of a hypothesis token with nothing opposite it.
        optional_deletion: Cost of deleting a reference token marked
            optionally deletable, or None if the model has no such notion.
    """

    name: str
    substitution: float
    deletion: float
    insertion: float
    optional_deletion: Optional[float] = None


COST_CHOICES: Dict[str, CostModel] = {
    # SCTK src/sclite/word.c: 4.0 for a substitution, 3.0 for an insertion or
    # a deletion, 2.0 to delete an optionally deletable word ("the cost to
    # CORR < OD < INS/DEL"), 0.0 for a correct token.
    "sclite": CostModel(
        name="sclite",
        substitution=4.0,
        deletion=3.0,
        insertion=3.0,
        optional_deletion=2.0,
    ),
    "unit": CostModel(name="unit", substitution=1.0, deletion=1.0, insertion=1.0),
}


def _is_optionally_deletable(token: str) -> bool:
    """Return whether a reference token is marked optionally deletable.

    sclite's convention is parentheses around the word. The parentheses stay
    part of the token for comparison -- ``(um)`` against a spoken ``um`` is a
    substitution, not a hit -- so this only ever changes what a deletion
    costs.
    """
    return len(token) > 2 and token.startswith("(") and token.endswith(")")


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
    costs: str = "sclite",
    optional_deletion_is_correct: bool = False,
) -> AlignmentResult:
    """Align a hypothesis against a reference.

    Args:
        ref: Reference tokens.
        hyp: Hypothesis tokens.
        backend: ``"python"`` (the default) is the reference implementation
            and honours both cost models. ``"rapidfuzz"`` is the
            linear-memory one, for inputs too long for a full table, and is
            unit-cost only. ``"auto"`` prefers rapidfuzz when it is installed
            and the cost model allows it.
        costs: ``"sclite"`` (the default) reproduces sclite's weighted
            alignment; ``"unit"`` is plain Levenshtein.
        optional_deletion_is_correct: Score a deleted optionally-deletable
            reference token as correct rather than as a deletion. This is
            sclite's ``-D``.

    Returns:
        The alignment, its counts, and the operations that produced them.

    Raises:
        ValueError: If ``backend`` or ``costs`` is not a recognised value, or
            if ``backend="rapidfuzz"`` is combined with a weighted model.
        ImportError: If ``backend="rapidfuzz"`` and it is not installed.
        MemoryError: If the pure Python backend is asked for a table larger
            than it is willing to build. Install rapidfuzz for these.
    """
    ref = list(ref)
    hyp = list(hyp)

    if costs not in COST_CHOICES:
        raise ValueError(
            f"unknown cost model '{costs}': expected one of " f"{sorted(COST_CHOICES)}"
        )
    model = COST_CHOICES[costs]

    if backend == "auto":
        usable = _RapidfuzzLevenshtein is not None and model.name == "unit"
        backend = "rapidfuzz" if usable else "python"
    if backend == "rapidfuzz":
        if _RapidfuzzLevenshtein is None:
            raise ImportError(
                "backend='rapidfuzz' requires rapidfuzz: pip install rapidfuzz"
            )
        if model.name != "unit":
            # Silently returning unit-cost edits here would report an S/D/I
            # split that is not the one the requested model defines.
            raise ValueError(
                f"backend='rapidfuzz' computes unit-cost edits and cannot "
                f"express the '{model.name}' cost model. Use costs='unit' "
                "for it, or backend='python' for this model."
            )
        operations = _editops_rapidfuzz(ref, hyp)
    elif backend == "python":
        cells = (len(ref) + 1) * (len(hyp) + 1)
        if cells > _PYTHON_BACKEND_CELL_LIMIT:
            raise MemoryError(
                f"aligning {len(ref)} reference and {len(hyp)} hypothesis "
                f"tokens needs a {cells:,}-cell table. Pass "
                "backend='rapidfuzz' (pip install rapidfuzz) with "
                "costs='unit', which aligns this in linear memory, or "
                "segment the input."
            )
        operations = _editops_python(ref, hyp, model)
    else:
        raise ValueError(
            f"unknown backend '{backend}': expected 'auto', 'python' or 'rapidfuzz'"
        )

    if optional_deletion_is_correct:
        operations = [
            (
                ("hit", ref_token, hyp_token)
                if op == "del" and _is_optionally_deletable(ref_token)
                else (op, ref_token, hyp_token)
            )
            for op, ref_token, hyp_token in operations
        ]

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


# Which predecessor a cell was reached from, recorded during the forward
# pass. Deciding this once is what makes the backtrace agree with the
# comparison order below; re-deriving it from the costs afterwards would have
# to repeat the same tie-break to get the same answer.
_FROM_DIAGONAL, _FROM_ABOVE, _FROM_LEFT = 0, 1, 2


def _editops_python(
    ref: List[str], hyp: List[str], model: CostModel
) -> List[Operation]:
    """Align with a dynamic program and a backtrace.

    The table is ``(len(ref) + 1) x (len(hyp) + 1)``; cell ``(i, j)`` is the
    cost of turning the first ``i`` reference tokens into the first ``j``
    hypothesis tokens.

    The comparison order is sclite's own (``src/sclite/net_dp.c``): take the
    substitution when it is no worse than either alternative, otherwise the
    deletion when it beats the insertion, otherwise the insertion. Under unit
    costs that order is the tie-break; under weighted costs it mostly does
    not arise, because the alternatives rarely come out equal.
    """
    n, m = len(ref), len(hyp)
    deletion_costs = [
        (
            model.optional_deletion
            if model.optional_deletion is not None and _is_optionally_deletable(token)
            else model.deletion
        )
        for token in ref
    ]

    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    # One row of choices per reference token; the margins are unambiguous.
    choice = [bytearray(m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = cost[i - 1][0] + deletion_costs[i - 1]
        choice[i][0] = _FROM_ABOVE
    for j in range(1, m + 1):
        cost[0][j] = cost[0][j - 1] + model.insertion
        choice[0][j] = _FROM_LEFT
    for i in range(1, n + 1):
        ref_token = ref[i - 1]
        delete_cost = deletion_costs[i - 1]
        row, previous_row, choice_row = cost[i], cost[i - 1], choice[i]
        for j in range(1, m + 1):
            substitute = previous_row[j - 1] + (
                0.0 if ref_token == hyp[j - 1] else model.substitution
            )
            delete = previous_row[j] + delete_cost
            insert = row[j - 1] + model.insertion
            if substitute <= delete and substitute <= insert:
                row[j], choice_row[j] = substitute, _FROM_DIAGONAL
            elif delete < insert:
                row[j], choice_row[j] = delete, _FROM_ABOVE
            else:
                row[j], choice_row[j] = insert, _FROM_LEFT

    operations: List[Operation] = []
    i, j = n, m
    while i > 0 or j > 0:
        step = choice[i][j]
        if step == _FROM_DIAGONAL:
            op = "hit" if ref[i - 1] == hyp[j - 1] else "sub"
            operations.append((op, ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif step == _FROM_ABOVE:
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
