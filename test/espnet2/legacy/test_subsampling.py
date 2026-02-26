import pytest
import torch

from espnet2.legacy.nets.pytorch_backend.transformer.subsampling import (
    Conv1dSubsampling1,
    Conv1dSubsampling2,
    Conv1dSubsampling3,
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    check_short_utt,
)

SUBSAMPLING_CLASSES = (
    Conv1dSubsampling1,
    Conv1dSubsampling2,
    Conv1dSubsampling3,
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
)

TEST_IDIM = 20
TEST_ODIM = 8


@pytest.mark.parametrize(
    "dtype, device",
    [(dt, dv) for dt in ("float32", "float64") for dv in ("cpu", "cuda")],
)
@pytest.mark.parametrize("subsampling_cls", SUBSAMPLING_CLASSES)
def test_subsampling_forward_mask_and_prefix(dtype, device, subsampling_cls):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no cuda device is available")

    dtype = getattr(torch, dtype)
    bsz = 2
    tlen = 40
    plen = 3

    module = subsampling_cls(TEST_IDIM, TEST_ODIM, 0.0).to(dtype=dtype, device=device)
    x = torch.rand(bsz, tlen, TEST_IDIM, dtype=dtype, device=device)
    x_mask = torch.ones(bsz, 1, tlen, dtype=torch.bool, device=device)
    prefix_embeds = torch.rand(bsz, plen, TEST_ODIM, dtype=dtype, device=device)

    y, y_mask = module(x, x_mask)
    assert y.size(0) == bsz
    assert y.size(2) == TEST_ODIM
    assert y_mask is not None
    assert y.size(1) == y_mask.size(2)

    y_prefix, y_mask_prefix = module(x, x_mask, prefix_embeds=prefix_embeds)
    assert y_prefix.size(0) == bsz
    assert y_prefix.size(2) == TEST_ODIM
    assert y_mask_prefix is not None
    assert y_prefix.size(1) == y_mask_prefix.size(2)
    assert y_prefix.size(1) == y.size(1) + plen


@pytest.mark.parametrize(
    "subsampling_cls, limit",
    [
        (Conv1dSubsampling1, 5),
        (Conv1dSubsampling2, 5),
        (Conv1dSubsampling3, 7),
        (Conv2dSubsampling1, 5),
        (Conv2dSubsampling2, 7),
        (Conv2dSubsampling, 7),
        (Conv2dSubsampling6, 11),
        (Conv2dSubsampling8, 15),
    ],
)
def test_check_short_utt(subsampling_cls, limit):
    module = subsampling_cls(TEST_IDIM, TEST_ODIM, 0.0)

    is_short, minimum = check_short_utt(module, limit - 1)
    assert is_short is True
    assert minimum == limit

    is_short, minimum = check_short_utt(module, limit)
    assert is_short is False
    assert minimum == -1


def test_subsampling_test_dimensions_are_valid():
    lower_bounds = {
        Conv1dSubsampling1: 1,
        Conv1dSubsampling2: 1,
        Conv1dSubsampling3: 1,
        Conv2dSubsampling1: 5,
        Conv2dSubsampling2: 7,
        Conv2dSubsampling: 7,
        Conv2dSubsampling6: 11,
        Conv2dSubsampling8: 15,
    }

    assert TEST_ODIM > 0
    for subsampling_cls, minimum_idim in lower_bounds.items():
        assert TEST_IDIM >= minimum_idim
        module = subsampling_cls(TEST_IDIM, TEST_ODIM, 0.0)
        assert module is not None


@pytest.mark.parametrize(
    "subsampling_cls",
    SUBSAMPLING_CLASSES,
)
def test_subsampling_state_dict_compatibility(subsampling_cls):
    bsz = 2
    tlen = 40

    latest_module = subsampling_cls(TEST_IDIM, TEST_ODIM, 0.0)
    legacy_state_dict = {}
    for key, value in latest_module.state_dict().items():
        if key == "out.weight":
            legacy_state_dict["out.0.weight"] = value
        elif key == "out.bias":
            legacy_state_dict["out.0.bias"] = value
        elif key.startswith("pos_enc."):
            legacy_state_dict[f"out.1.{key[len('pos_enc.'):]}"] = value
        else:
            legacy_state_dict[key] = value

    reloaded_module = subsampling_cls(TEST_IDIM, TEST_ODIM, 0.0)
    reloaded_module.load_state_dict(legacy_state_dict)

    x = torch.rand(bsz, tlen, TEST_IDIM)
    x_mask = torch.ones(bsz, 1, tlen, dtype=torch.bool)

    latest_y, latest_mask = latest_module(x, x_mask)
    reloaded_y, reloaded_mask = reloaded_module(x, x_mask)

    assert torch.allclose(latest_y, reloaded_y)
    assert torch.equal(latest_mask, reloaded_mask)
