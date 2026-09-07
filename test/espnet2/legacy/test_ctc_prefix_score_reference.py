"""Pin the vectorized CTC prefix scorer against the loop-based one.

`CTCPrefixScoreTH.__call__` used to walk the hypotheses in Python, once per
decoding step. Those loops were replaced with tensor indexing, which is a
large speedup on an accelerator -- each iteration was a synchronisation -- but
the arithmetic has to come out bit for bit the same.

The beam search parity tests cannot check this: both the current and the
frozen beam search import the same prefix scorer, so a change here would move
the reference along with the code. Hence a frozen copy of its own.
"""

from test.espnet2.legacy.reference_ctc_prefix_score import (
    CTCPrefixScoreTH as ReferenceCTCPrefixScoreTH,
)

import numpy
import pytest
import torch

from espnet2.legacy.nets.ctc_prefix_score import CTCPrefixScoreTH

BLANK, EOS = 0, 1


def _inputs(n_utt, beam, frames, vocab, seed, uneven):
    torch.manual_seed(seed)
    x = torch.log_softmax(torch.randn(n_utt, frames, vocab, dtype=torch.float64), -1)
    if uneven:
        xlens = torch.tensor(
            [max(2, frames - 3 * b) for b in range(n_utt)], dtype=torch.long
        )
    else:
        xlens = torch.full((n_utt,), frames, dtype=torch.long)
    return x, xlens


def _run(impl, x, xlens, ys, scoring_ids, steps):
    """Score `steps` extensions, threading the state through as decoding does."""
    scorer = impl(x.clone(), xlens, BLANK, EOS, 0)
    state, out = None, []
    for i in range(steps):
        scores, state = scorer(ys[: i + 2].t().contiguous(), state, scoring_ids)
        out.append(scores)
        # keep every hypothesis where it is, which is what index_select_state
        # does for an identity permutation, so the shapes stay consistent
        n_bh, odim = scores.shape
        n_hyp = n_bh // len(xlens)
        best = (
            torch.arange(n_hyp, device=scores.device).repeat(len(xlens), 1) * odim
            + EOS
            + 1
        )
        state = scorer.index_select_state(state, best)
    return out


@pytest.mark.parametrize(
    "n_utt, beam, frames, vocab, uneven, use_scoring_ids",
    [
        (1, 1, 12, 7, False, False),
        (1, 3, 12, 7, False, True),
        (2, 3, 12, 7, False, False),
        (2, 3, 15, 9, True, True),
        (4, 2, 20, 11, True, False),
        (4, 5, 20, 11, True, True),
        (3, 4, 9, 6, True, True),
    ],
)
def test_matches_the_loop_implementation(
    n_utt, beam, frames, vocab, uneven, use_scoring_ids
):
    """Vectorizing the per-hypothesis loops must not change a single value."""
    n_bh = n_utt * beam
    steps = 4
    x, xlens = _inputs(
        n_utt, beam, frames, vocab, seed=n_utt * 31 + beam, uneven=uneven
    )

    torch.manual_seed(7)
    ys = torch.randint(BLANK + 2, vocab, (steps + 2, n_bh))
    scoring_ids = None
    if use_scoring_ids:
        snum = max(2, vocab // 2)
        scoring_ids = torch.stack([torch.randperm(vocab)[:snum] for _ in range(n_bh)])

    expected = _run(ReferenceCTCPrefixScoreTH, x, xlens, ys, scoring_ids, steps)
    actual = _run(CTCPrefixScoreTH, x, xlens, ys, scoring_ids, steps)

    assert len(actual) == len(expected)
    for i, (exp, act) in enumerate(zip(expected, actual)):
        numpy.testing.assert_allclose(
            exp.numpy(), act.numpy(), rtol=0, atol=0, err_msg=f"step {i}"
        )


def test_reference_is_a_frozen_copy():
    """Guard the reference against being edited to track the implementation."""
    import inspect
    from test.espnet2.legacy import reference_ctc_prefix_score

    source = inspect.getsource(reference_ctc_prefix_score)
    assert "Frozen copy of the loop-based" in source
    # the frozen copy is the one that still walks the hypotheses in Python
    assert "for si in range(n_bh):" in source
