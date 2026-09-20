import pytest
import torch

from espnet2.universa.ar_universa.universa_beam_search import ARUniVERSABeamSearch


class Scorer:
    def init_state(self, x):
        return None

    def score(self, yseq, state, x):
        # Deliberately favor invalid tokens; constraints must still win.
        return torch.arange(9, dtype=x.dtype, device=x.device), None


@pytest.mark.parametrize("beam_size", [1, 20])
@pytest.mark.parametrize("skip_meta", [False, True])
def test_constrained_search_preserves_scores_and_order(beam_size, skip_meta):
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
        use_fixed_order=True,
    )
    result = search.forward(torch.zeros(3, 2))[0]
    assert result.yseq.tolist() == [2, 4, 6, 7, 8]
    assert result.score == (14 if skip_meta else 25)
