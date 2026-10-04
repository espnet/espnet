import pytest
import torch

from espnet2.legacy.nets.scorer_interface import ScorerInterface
from espnet2.universa.ar_universa.universa_beam_search import ARUniVERSABeamSearch


class Scorer(ScorerInterface):
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
    assert len(result.yseq) == 5  # Fixed pair count; EOS is not appended.
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


@pytest.mark.parametrize("label", [-1, 9])
def test_invalid_metric_label(label):
    with pytest.raises(ValueError, match="label"):
        ARUniVERSABeamSearch({}, {}, 1, 9, 2, 3, [label])


@pytest.mark.parametrize("bounds", [(-1, 2), (5, 5), (7, 5), (5, 10)])
def test_invalid_value_range(bounds):
    with pytest.raises(ValueError, match="range"):
        ARUniVERSABeamSearch({}, {}, 1, 9, 2, 3, [4], beam_masking={4: bounds})


def test_module_call_and_scorer_dtype_conversion():
    class ModuleScorer(torch.nn.Module, Scorer):
        def __init__(self):
            super().__init__()
            self.register_buffer("token_scores", torch.arange(9, dtype=torch.float32))

        def score(self, yseq, state, x):
            assert self.token_scores.dtype == x.dtype
            assert self.token_scores.device == x.device
            return self.token_scores, state + 1

    scorer = ModuleScorer()
    search = ARUniVERSABeamSearch(
        {"decoder": scorer},
        {"decoder": 1.0},
        1,
        9,
        2,
        3,
        [4],
        beam_masking={4: (5, 7)},
    )
    search.to(dtype=torch.float64)
    assert scorer.token_scores.dtype == torch.float64
    assert search(torch.zeros(3, 2, dtype=torch.float64))[0].yseq.tolist() == [2, 4, 6]


def test_hypothesis_dictionaries_are_independent():
    search = ARUniVERSABeamSearch(
        {"decoder": Scorer()},
        {"decoder": 1.0},
        2,
        9,
        2,
        3,
        [4],
        beam_masking={4: (5, 7)},
    )
    x = torch.zeros(3, 2)
    first, second = search.init_hyp(x)[0], search.init_hyp(x)[0]
    first.scores["decoder"] = 99
    first.states["decoder"] = 99
    assert second.scores["decoder"] == second.states["decoder"] == 0
    children = search.search([second], x)
    children[0].scores["decoder"] = 99
    children[0].states["decoder"] = 99
    assert children[1].scores["decoder"] != 99
    assert children[1].states["decoder"] == 2


def test_skipped_label_pruning_waits_for_value():
    class ValueScorer(Scorer):
        def score(self, yseq, state, x):
            scores = x.new_zeros(9)
            scores[7] = 100  # A token-level beam of one would choose this label.
            if yseq.tolist() == [2, 4]:
                scores[6] = 1000  # The other label has the better completed pair.
            return scores, state + 1

    search = ARUniVERSABeamSearch(
        {"decoder": ValueScorer()},
        {"decoder": 1.0},
        1,
        9,
        2,
        3,
        [4, 7],
        beam_masking={4: (5, 7), 7: (8, 9)},
        skip_meta_label_score=True,
    )
    result = search(torch.zeros(3, 2))[0]
    assert result.yseq.tolist() == [2, 4, 6, 7, 8]
    assert result.score == 1000
    assert search.beam_size == 1


def test_constraint_supports_batched_utterances_and_hypotheses():
    from espnet2.legacy.nets.batch_beam_search import BatchBeamSearch
    from espnet2.universa.ar_universa.universa_beam_search import MetricConstraintScorer

    constraint = MetricConstraintScorer(9, [4, 7], {4: (5, 7), 7: (8, 9)}, True)
    search = BatchBeamSearch(
        {"constraint": constraint},
        {"constraint": 1.0},
        2,
        9,
        2,
        3,
    )
    results = search(
        torch.zeros(2, 3, 2),
        x_lengths=torch.tensor([3, 2]),
        maxlenratio=-4,
        # Fixed-pair outputs cannot terminate early. The shared batch engine
        # may otherwise collect an EOS from a padded, impossible beam slot.
        minlenratio=-3,
    )
    assert len(results) == 2
    for hypotheses in results:
        assert len(hypotheses) == 2
        for hyp in hypotheses:
            # This exercises shared token search; its default termination adds EOS.
            assert hyp.yseq.tolist() in ([2, 4, 5, 7, 8, 3], [2, 4, 6, 7, 8, 3])
            assert hyp.score == 0


def test_unrestricted_values_do_not_consume_metric_labels():
    search = ARUniVERSABeamSearch(
        {"decoder": Scorer()},
        {"decoder": 1.0},
        1,
        9,
        2,
        3,
        [4, 8],
        use_fixed_order=True,
    )
    # The first value is also a requested label; it must still be emitted as a label.
    assert search(torch.zeros(3, 2))[0].yseq.tolist() == [2, 4, 8, 8, 8]
