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


@pytest.mark.parametrize("n_utt, beam, frames, vocab", [(1, 3, 20, 9), (3, 2, 24, 7)])
def test_a_window_wider_than_the_utterance_changes_nothing(n_utt, beam, frames, vocab):
    """`margin` is an approximation only when the window actually bites.

    A window at least as wide as the utterance covers every frame the exact
    recursion visits, so the result must be the same. It is not bit-identical:
    the window also moves `start`, so the final `logsumexp` sums a different
    number of logzero terms, which reassociates the addition.
    """
    n_bh, steps = n_utt * beam, 4
    x, xlens = _inputs(n_utt, beam, frames, vocab, seed=5, uneven=True)
    torch.manual_seed(3)
    ys = torch.randint(BLANK + 2, vocab, (steps + 2, n_bh))

    exact = _run(CTCPrefixScoreTH, x, xlens, ys, None, steps)
    wide = _run(
        lambda a, b, c, d, m=0: CTCPrefixScoreTH(a, b, c, d, margin=frames),
        x,
        xlens,
        ys,
        None,
        steps,
    )
    for i, (e, w) in enumerate(zip(exact, wide)):
        numpy.testing.assert_allclose(
            e.numpy(), w.numpy(), rtol=1e-12, atol=1e-12, err_msg=f"step {i}"
        )


def test_a_narrow_window_really_narrows_the_recursion():
    """A small margin must cut the work, not merely be accepted."""
    n_utt, beam, frames, vocab, steps = 1, 2, 60, 7, 5
    x, xlens = _inputs(n_utt, beam, frames, vocab, seed=11, uneven=False)
    torch.manual_seed(1)
    ys = torch.randint(BLANK + 2, vocab, (steps + 2, n_utt * beam))

    def logsumexp_calls(margin):
        scorer = CTCPrefixScoreTH(x.clone(), xlens, BLANK, EOS, margin)
        original = torch.logsumexp
        seen = [0]

        def spy(*a, **k):
            seen[0] += 1
            return original(*a, **k)

        torch.logsumexp = spy
        try:
            state = None
            for i in range(steps):
                scores, state = scorer(ys[: i + 2].t().contiguous(), state, None)
                odim = scores.shape[1]
                best = torch.arange(n_utt * beam).view(n_utt, beam) * odim + EOS + 1
                state = scorer.index_select_state(state, best)
        finally:
            torch.logsumexp = original
        return seen[0]

    exact, windowed = logsumexp_calls(0), logsumexp_calls(6)
    assert windowed < exact / 2, (windowed, exact)


def _silence_and_labels(layouts):
    """Peaked posteriors: per utterance, a label id per frame, 0 for silence."""
    vocab = 6
    post = torch.full((len(layouts), max(map(len, layouts)), vocab), 1e-4)
    for b, layout in enumerate(layouts):
        for t, label in enumerate(layout):
            post[b, t, label] = 1.0
    x = torch.log(post.double() / post.double().sum(-1, keepdim=True))
    xlens = torch.tensor([len(layout) for layout in layouts])
    return x, xlens, [[c for c in layout if c != BLANK] for layout in layouts]


def _follow(scorer, labels, attention=None):
    """Score each utterance's labels and then <eos>, one hypothesis each."""
    n_utt = len(labels)
    state, y, scores = None, torch.full((n_utt, 1), EOS), []
    for n, step in enumerate(zip(*[utt + [EOS] for utt in labels])):
        step = torch.tensor(step)
        local, state = scorer(y, state, None, attention and attention[n])
        scores.append(local[torch.arange(n_utt), step])
        state = scorer.index_select_state(state, step.unsqueeze(1))
        y = torch.cat([y, step.unsqueeze(1)], dim=1)
    return torch.stack(scores).numpy()


PAUSE = [0, 2, 0, 3] + [0] * 30 + [4, 0, 5, 0]
LEADING = [0] * 30 + [2, 0, 3, 0, 4, 0, 5, 0]
TRAILING = [0, 2, 0, 3, 0, 4, 0, 5] + [0] * 30


@pytest.mark.parametrize(
    "layouts",
    [[PAUSE], [LEADING], [TRAILING], [PAUSE, TRAILING, LEADING[20:]]],
    ids=["pause", "leading", "trailing", "batch"],
)
def test_a_window_is_not_stopped_by_silence_longer_than_itself(layouts):
    """Silence is free to cross, so no length of it may hide what follows.

    A hypothesis advances only inside the window. When the window ended in
    the middle of a silence it held nothing to advance to, and <eos> was
    impossible until it reached the last frame, so the label after a pause
    and the end of an utterance both scored as if they could not happen.
    The long silences here are three times the margin.
    """
    margin = 10
    x, xlens, labels = _silence_and_labels(layouts)
    exact = _follow(CTCPrefixScoreTH(x.clone(), xlens, BLANK, EOS, 0), labels)
    windowed = _follow(CTCPrefixScoreTH(x.clone(), xlens, BLANK, EOS, margin), labels)
    numpy.testing.assert_allclose(windowed, exact, atol=1e-6)


@pytest.mark.parametrize(
    "layout", [PAUSE, LEADING, TRAILING], ids=["pause", "leading", "trailing"]
)
def test_a_window_placed_by_attention_is_not_stopped_by_silence_either(layout):
    """Attention weights say where the window is, not what it has to hold.

    The attention here rests on the label that was scored last, which is a
    whole silence before the next one.
    """
    margin = 10
    x, xlens, labels = _silence_and_labels([layout])
    spoken = [0] + [t for t, label in enumerate(layout) if label != BLANK]
    attention = [
        torch.nn.functional.one_hot(torch.tensor([t]), len(layout)).to(x.dtype)
        for t in spoken
    ]
    exact = _follow(CTCPrefixScoreTH(x.clone(), xlens, BLANK, EOS, 0), labels)
    windowed = _follow(
        CTCPrefixScoreTH(x.clone(), xlens, BLANK, EOS, margin), labels, attention
    )
    numpy.testing.assert_allclose(windowed, exact, atol=1e-6)


def test_a_hypothesis_can_end_where_the_window_stops():
    """Noise the model half takes for labels stops the window before the end.

    The window is not stretched over frames whose most likely label is not
    blank, so after the speech it ends in the noise. A hypothesis that ends
    there has to stay silent through the rest, which costs what the blanks
    cost and is not impossible.
    """
    margin, vocab = 10, 6
    speech, noise = [0, 2, 0, 3, 0, 4, 0, 5], 60
    post = torch.full((1, len(speech) + noise, vocab), 1e-4, dtype=torch.float64)
    for t, label in enumerate(speech):
        post[0, t, label] = 1.0
    for t in range(len(speech), len(speech) + noise):
        post[0, t, BLANK] = 1.0
        if (t - len(speech)) % 3 == 0:  # every third frame leans to a label
            post[0, t, BLANK], post[0, t, 2 + t % 4] = 0.4, 0.6
    x = torch.log(post / post.sum(-1, keepdim=True))
    xlens = torch.tensor([x.size(1)])
    labels = [[c for c in speech if c != BLANK]]

    exact = _follow(CTCPrefixScoreTH(x.clone(), xlens, BLANK, EOS, 0), labels)
    windowed = _follow(CTCPrefixScoreTH(x.clone(), xlens, BLANK, EOS, margin), labels)
    assert -30 < exact[-1, 0] < -10  # ending is costly here, and possible
    # not to 1e-6: the window leaves out the alignments that put the last
    # label somewhere in the noise, which is its approximation
    numpy.testing.assert_allclose(windowed, exact, atol=1e-2)


def _follow_in_two_blocks(x, labels, first, margin):
    """Score `labels` as the online search does: a first block, then the rest."""
    scorer = CTCPrefixScoreTH(
        x[:, :first].clone(), torch.tensor([first]), BLANK, EOS, margin
    )
    state, y, scores = None, [EOS], []
    for n, label in enumerate(labels + [EOS]):
        if n == 2:  # the rest of the recording arrives
            scorer.extend_prob(x.clone())
            r, s, f_min, f_max = scorer.extend_state(
                (state[0][:, :, 0], state[1][0], *state[2:])
            )
            state = (r.unsqueeze(2), s.unsqueeze(0), f_min, f_max)
        local, state = scorer(torch.tensor([y]), state, None)
        scores.append(float(local[0, label]))
        state = scorer.index_select_state(state, torch.tensor([[label]]))
        y.append(label)
    return scores


def test_a_window_follows_the_posteriors_when_they_are_extended():
    """What the window knows of the silence has to grow with the posteriors.

    The online search extends the posteriors block by block.
    """
    x, _, labels = _silence_and_labels([PAUSE])
    exact = _follow_in_two_blocks(x, labels[0], first=6, margin=0)
    windowed = _follow_in_two_blocks(x, labels[0], first=6, margin=10)
    numpy.testing.assert_allclose(windowed, exact, atol=1e-6)
