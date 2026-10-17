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
    _conv_out_length,
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


# Time-axis (kernel, stride) for each Conv* class. Used to compute the
# length a single utterance should have after subsampling.
SUBSAMPLE_CONVS = {
    Conv1dSubsampling1: ((3, 1), (3, 1)),
    Conv1dSubsampling2: ((3, 1), (3, 2)),
    Conv1dSubsampling3: ((3, 1), (5, 3)),
    Conv2dSubsampling: ((3, 2), (3, 2)),
    Conv2dSubsampling1: ((3, 1), (3, 1)),
    Conv2dSubsampling2: ((3, 2), (3, 1)),
    Conv2dSubsampling6: ((3, 2), (5, 3)),
    Conv2dSubsampling8: ((3, 2), (3, 2), (3, 2)),
}


def _expected_olens(ilens, convs):
    olens = torch.as_tensor(ilens, dtype=torch.long)
    for kernel_size, stride in convs:
        olens = _conv_out_length(olens, kernel_size, stride)
    return olens.clamp(min=0)


def _pad_mask(lengths, width):
    mask = torch.zeros(len(lengths), 1, width, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, 0, :length] = True
    return mask


@pytest.mark.parametrize("subsampling_cls", SUBSAMPLING_CLASSES)
def test_olens_is_independent_of_batch_mates(subsampling_cls):
    """A short utterance must keep the same olens next to a longer one.

    Reproduces issue 6600: slicing the padded mask from the end made the
    288-frame row 142 frames alone and 144 next to a 363-frame row.
    """
    short, long_ = 288, 363
    convs = SUBSAMPLE_CONVS[subsampling_cls]
    module = subsampling_cls(TEST_IDIM, TEST_ODIM, 0.0).eval()

    feats = torch.randn(2, long_, TEST_IDIM)
    mask_alone = _pad_mask([short], short)
    mask_mixed = _pad_mask([long_, short], long_)
    mask_pair = _pad_mask([short, short], short)

    with torch.no_grad():
        _, mask_from_alone = module(feats[1:2, :short], mask_alone)
        y_mixed, mask_from_mixed = module(feats, mask_mixed)
        _, mask_from_pair = module(feats[:, :short], mask_pair)

    olens_alone = mask_from_alone.squeeze(1).sum(1)
    olens_mixed = mask_from_mixed.squeeze(1).sum(1)
    olens_pair = mask_from_pair.squeeze(1).sum(1)
    expected = _expected_olens([long_, short], convs)

    assert olens_alone.tolist() == [expected[1].item()]
    assert olens_mixed.tolist() == expected.tolist()
    assert olens_pair.tolist() == [expected[1].item(), expected[1].item()]
    assert mask_from_mixed.size(2) == y_mixed.size(1)
    if subsampling_cls is Conv1dSubsampling2:
        assert olens_alone.tolist() == [142]
        assert olens_mixed.tolist() == [180, 142]
        assert olens_pair.tolist() == [142, 142]


def test_issue_6600_transformer_encoder_repro():
    """Pin the snippet from https://github.com/espnet/espnet/issues/6600."""
    pytest.importorskip("typeguard")
    from espnet2.asr.encoder.transformer_encoder import TransformerEncoder

    torch.manual_seed(0)
    enc = TransformerEncoder(
        input_size=8,
        output_size=4,
        attention_heads=2,
        linear_units=4,
        num_blocks=1,
        input_layer="conv1d2",
    ).eval()

    short, long_ = 288, 363
    feats = torch.randn(2, long_, 8)
    with torch.no_grad():
        _, olens_alone, _ = enc(feats[1:2, :short], torch.tensor([short]))
        _, olens_batch, _ = enc(feats, torch.tensor([long_, short]))
        _, olens_pair, _ = enc(feats[:, :short], torch.tensor([short, short]))

    assert olens_alone.tolist() == [142]
    assert olens_batch.tolist() == [180, 142]
    assert olens_pair.tolist() == [142, 142]


@pytest.mark.parametrize("subsampling_cls", SUBSAMPLING_CLASSES)
def test_uniform_batch_mask_matches_conv_length(subsampling_cls):
    """Full-length rows must still match the convolution length formula."""
    tlen = 40
    convs = SUBSAMPLE_CONVS[subsampling_cls]
    module = subsampling_cls(TEST_IDIM, TEST_ODIM, 0.0)
    x = torch.rand(2, tlen, TEST_IDIM)
    x_mask = torch.ones(2, 1, tlen, dtype=torch.bool)

    y, y_mask = module(x, x_mask)
    expected = _expected_olens([tlen, tlen], convs)

    assert y.size(1) == y_mask.size(2)
    assert y_mask.squeeze(1).sum(1).tolist() == expected.tolist()
