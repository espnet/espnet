import pytest
import torch

from espnet2.legacy.nets.pytorch_backend.transformer.subsampling import (
    _conv_out_length,
)
from espnet2.legacy.nets.pytorch_backend.transformer.subsampling_without_posenc import (
    Conv2dSubsamplingWOPosEnc,
)

TEST_IDIM = 20
TEST_ODIM = 8

# 71 / 47 / 35: reviewed in #6633.
WOPOSENC_CASES = (
    ("conv2d", [3, 3], [2, 2], 71),
    ("conv2d6", [3, 5], [2, 3], 47),
    ("conv2d8", [3, 3, 3], [2, 2, 2], 35),
)


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


@pytest.mark.parametrize("_name, kernels, strides, alone_olen", WOPOSENC_CASES)
def test_woposenc_olens_is_independent_of_batch_mates(
    _name, kernels, strides, alone_olen
):
    short, long_ = 288, 363
    convs = tuple(zip(kernels, strides))
    module = Conv2dSubsamplingWOPosEnc(
        TEST_IDIM, TEST_ODIM, 0.0, kernels, strides
    ).eval()

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

    assert olens_alone.tolist() == [alone_olen]
    assert olens_mixed[1].item() == alone_olen
    assert olens_mixed.tolist() == expected.tolist()
    assert olens_pair.tolist() == [alone_olen, alone_olen]
    assert mask_from_mixed.size(2) == y_mixed.size(1)
