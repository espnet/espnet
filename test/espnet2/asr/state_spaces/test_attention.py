import pytest
import torch

from espnet2.asr.state_spaces import registry, utils
from espnet2.asr.state_spaces.attention import MultiHeadedAttention


def _build(use_sdpa):
    torch.manual_seed(0)
    return MultiHeadedAttention(16, 4, dropout=0.1, use_sdpa=use_sdpa).eval()


def test_use_sdpa_defaults_to_false():
    assert MultiHeadedAttention(16, 4).use_sdpa is False


@pytest.mark.parametrize("cross", [False, True])
@pytest.mark.parametrize("mask_shape", [None, (2, 1, 7), (2, 5, 7)])
def test_use_sdpa_matches_the_default(cross, mask_shape):
    query = torch.randn(2, 5 if cross else 7, 16)
    memory = torch.randn(2, 7, 16) if cross else None
    mask = None
    if mask_shape is not None:
        if not cross and mask_shape[1] != 1:
            mask_shape = (2, 7, 7)
        mask = torch.ones(mask_shape, dtype=torch.bool)
        mask[1, ..., 4:] = False

    expected, _ = _build(False)(query, memory, mask=mask)
    actual, state = _build(True)(query, memory, mask=mask)

    assert state is None
    assert torch.allclose(actual, expected, atol=1e-6)


def test_use_sdpa_step_matches_the_default():
    query = torch.randn(2, 1, 16)
    expected, _ = _build(False).step(query, None)
    actual, _ = _build(True).step(query, None)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_use_sdpa_reaches_the_layer_from_a_config():
    layer = utils.instantiate(
        registry.layer, {"_name_": "mha", "n_head": 4, "use_sdpa": True}, 16
    )
    assert isinstance(layer, MultiHeadedAttention)
    assert layer.use_sdpa is True
