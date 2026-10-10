import pytest
import torch

from espnet2.bin.s2t_inference import ScoreFilter

NOTIME, FIRST, LAST, SOS, EOS, VOCAB = 5, 10, 19, 1, 2, 30


def _filter():
    return ScoreFilter(
        notimestamps=NOTIME,
        first_time=FIRST,
        last_time=LAST,
        sos=SOS,
        eos=EOS,
        vocab_size=VOCAB,
    )


def _reference(score_filter, ys):
    """Call the original unvectorized loop reference implementation (_batch_score_loop)."""
    return score_filter._batch_score_loop(ys, [None] * len(ys), None)[0]


HANDMADE = [
    # one row per branch of ScoreFilter.score, all of the same length
    [SOS, 3, 4, NOTIME, 7, 8, 9],  # no timestamps predicted
    [SOS, 3, 4, 12, 7, 8, 9],  # one timestamp open: eos and earlier times banned
    [SOS, 3, 4, 12, 7, 14, 9],  # two timestamps, last token is text: illegal
    [SOS, 3, 4, 12, 7, 8, 14],  # two timestamps, last token closes a pair
    [SOS, 3, 4, 12, 15, 8, 17],  # three timestamps: open again
    [SOS, 3, 4, 11, 11, 8, 11],  # repeated timestamps, still counted
]


def test_vectorised_batch_score_matches_loop_on_handmade_rows():
    """Verify that vectorised batch_score matches the original _batch_score_loop on hand-crafted boundary rows."""
    score_filter = _filter()
    ys = torch.tensor(HANDMADE, dtype=torch.int64)
    scores, states = score_filter.batch_score(ys, [None] * len(ys), None)
    assert scores.shape == (len(ys), VOCAB)
    assert torch.equal(scores, _reference(score_filter, ys))
    assert states == [None] * len(ys)


def test_vectorised_batch_score_matches_loop_right_after_the_prompt():
    """Verify equivalence between vectorised batch_score and original _batch_score_loop right after the prompt."""
    score_filter = _filter()
    ys = torch.tensor([[SOS, 3, 4], [SOS, 3, NOTIME]], dtype=torch.int64)
    scores, _ = score_filter.batch_score(ys, [None, None], None)
    assert torch.equal(scores, _reference(score_filter, ys))
    # first token must be a timestamp: everything outside the range is banned
    assert (
        torch.isinf(scores[0, :FIRST]).all()
        and torch.isinf(scores[0, LAST + 1 :]).all()
    )
    assert (scores[0, FIRST : LAST + 1] == 0).all()


@pytest.mark.parametrize("ylen", [3, 4, 7, 12])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vectorised_batch_score_matches_loop_on_random_rows(ylen, seed):
    """Verify equivalence between vectorised batch_score and original _batch_score_loop on randomized sequences."""
    torch.manual_seed(seed)
    score_filter = _filter()
    n = 64
    ys = torch.randint(3, VOCAB, (n, ylen))
    ys[:, 0] = SOS
    # sprinkle timestamps and the no-timestamp token so every branch is hit
    time_mask = torch.rand(n, ylen) < 0.35
    time_mask[:, 0] = False
    ys[time_mask] = torch.randint(FIRST, LAST + 1, (int(time_mask.sum()),))
    ys[torch.rand(n) < 0.2, min(3, ylen - 1)] = NOTIME
    scores, _ = score_filter.batch_score(ys, [None] * n, None)
    assert torch.equal(scores, _reference(score_filter, ys))
