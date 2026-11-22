import pickle

import pytest
import torch

from espnet2.legacy.nets.beam_search import Hypothesis
from espnet2.torch_utils.device_funcs import to_device


@pytest.mark.parametrize("field", ["scores", "states", "hs"])
def test_mutable_defaults_are_independent(field):
    first = Hypothesis(torch.tensor([2]))
    second = Hypothesis(torch.tensor([2]))
    value = getattr(first, field)
    if field == "hs":
        value.append(torch.tensor([1.0]))
    else:
        value["decoder"] = 1.0
    assert not getattr(second, field)


def test_explicit_containers_are_retained():
    scores, states, hs = {}, {}, []
    hyp = Hypothesis(torch.tensor([2]), 0.0, scores, states, hs)
    assert hyp.scores is scores
    assert hyp.states is states
    assert hyp.hs is hs


def test_tuple_and_serialization_api():
    hyp = Hypothesis(torch.tensor([2, 4]), 1.0, {"decoder": torch.tensor(1.0)})
    assert isinstance(hyp, tuple)
    assert tuple(hyp)[0] is hyp.yseq
    assert hyp._asdict()["states"] is hyp.states
    updated = hyp._replace(score=2.0)
    assert type(updated) is Hypothesis
    assert updated.score == 2.0
    assert hyp.score == 1.0
    assert updated.asdict() == {
        "yseq": [2, 4],
        "score": 2.0,
        "scores": {"decoder": 1.0},
        "states": {},
        "hs": [],
    }
    restored = pickle.loads(pickle.dumps(hyp))
    assert type(restored) is Hypothesis
    torch.testing.assert_close(restored.yseq, hyp.yseq)
    assert restored.asdict() == hyp.asdict()
    moved = to_device(hyp, "cpu")
    assert type(moved) is Hypothesis
    assert moved.asdict() == hyp.asdict()
