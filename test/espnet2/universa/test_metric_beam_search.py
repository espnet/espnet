import pytest
import torch

from espnet2.universa.ar_universa.universa_beam_search import ARUniVERSABeamSearch


class Scorer:
    def init_state(self, x):
        return 0

    def score(self, yseq, state, x):
        # Even unscored labels must advance the decoder state.
        assert state == len(yseq) - 1
        # Deliberately favor invalid tokens; constraints must still win.
        return torch.arange(9, dtype=x.dtype, device=x.device), state + 1


@pytest.mark.parametrize("beam_size", [1, 20])
@pytest.mark.parametrize("skip_meta", [False, True])
@pytest.mark.parametrize("fixed_order", [False, True])
def test_constrained_search_preserves_scores_and_order(
    beam_size, skip_meta, fixed_order
):
    search = ARUniVERSABeamSearch(
        scorers={"decoder": Scorer()},
        weights={"decoder": 1.0},
        beam_size=beam_size,
        vocab_size=9,
        sos=2,
        eos=3,
        meta_label_for_search=[4, 7],
        beam_masking={4: (5, 7), 7: (8, 9)},
        skip_meta_label_score=skip_meta,
        use_fixed_order=fixed_order,
    )
    result = search.forward(torch.zeros(3, 2))[0]
    pairs = list(zip(result.yseq.tolist()[1::2], result.yseq.tolist()[2::2]))
    assert sorted(pairs) == [(4, 6), (7, 8)]
    if fixed_order:
        assert pairs == [(4, 6), (7, 8)]
    assert result.score == (14 if skip_meta else 25)
    assert result.scores["decoder"] == result.score
    assert result.unused_meta_label_ids == []
    assert search.meta_label_for_search == [4, 7]
    assert result.states["decoder"] == 4


def test_empty_metric_request():
    search = ARUniVERSABeamSearch({}, {}, 1, 9, 2, 3, [])
    result = search.forward(torch.zeros(3, 2))
    assert len(result) == 1
    assert result[0].yseq.tolist() == [2]
    assert result[0].score == 0


@pytest.mark.parametrize("beam_size, labels", [(0, [4]), (1, [4, 4])])
def test_invalid_search_request(beam_size, labels):
    with pytest.raises(ValueError):
        ARUniVERSABeamSearch({}, {}, beam_size, 9, 2, 3, labels)
