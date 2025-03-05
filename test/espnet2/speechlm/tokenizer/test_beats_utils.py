import pytest
import torch

from espnet2.speechlm.tokenizer.beats_utils import (
    kmeans,
    sample_vectors,
    beats_frontend,
    forward_padding_mask_conv,
    freeze_conv_module,
)
from espnet2.speechlm.tokenizer.beats_tokenizer import BeatsRandomTokenizer


@pytest.mark.parametrize("num", [3, 15])
def test_sample_vectors(num):
    samples = torch.randn(10, 5)
    out = sample_vectors(samples, num)
    assert out.shape == (num, 5)


@pytest.mark.parametrize("use_cosine_sim", [True, False])
def test_kmeans(use_cosine_sim):
    samples = torch.randn(10, 5)
    num_clusters = 3
    num_iters = 3
    means, bins = kmeans(samples, num_clusters, num_iters, use_cosine_sim)
    assert means.shape == (num_clusters, 5)
    assert bins.shape == (num_clusters,)
    # FIXME(shikhar): Maybe ensure that all samples are closest to
    #  their respective means, later


def test_beats_forntent():
    source = torch.randn(2, 16000)
    fbank_mean = 2.0
    fbank_std = 3.0
    _ = beats_frontend(source, fbank_mean, fbank_std)


def test_forward_padding_mask_conv():
    n_dim = 128
    conv2d_module = torch.nn.Conv2d(1, 1, kernel_size=16, stride=16, bias=False)
    freeze_conv_module(conv2d_module)

    # Test case 1: No padding (fully active input)
    padding_mask = torch.zeros(1, 16000, dtype=torch.bool)
    padding_mask1 = forward_padding_mask_conv(padding_mask, n_dim, conv2d_module)
    assert not padding_mask1.all(), "All elements should not be padded"
    assert padding_mask1.shape == (
        1,
        1000 * 8,
    ), f"Expected shape (1, 1000 * 8), got {padding_mask1.shape}"

    # Test case 2: Padding half the sequence (even split)
    padding_mask = torch.zeros(1, 16000, dtype=torch.bool)
    padding_mask[:, 8000:] = True
    padding_mask2 = forward_padding_mask_conv(padding_mask, n_dim, conv2d_module)
    assert padding_mask2.shape == (
        1,
        1000 * 8,
    ), f"Expected shape (1, 1000 * 8), got {padding_mask2.shape}"
    assert (~padding_mask2[0, : 500 * 8]).all(), "First half should be unpadded"
    assert (padding_mask2[0, 500 * 8 :]).all(), "Second half should be fully padded"

    # Test case 3: Random padding (mixed padding and non-padding)
    padding_mask = torch.zeros(1, 16000, dtype=torch.bool)
    padding_mask[:, 4000:6000] = True  # Padding only a segment
    padding_mask3 = forward_padding_mask_conv(padding_mask, n_dim, conv2d_module)
    assert padding_mask3.shape == (1, 1000 * 8)
    assert (~padding_mask3[0, : 250 * 8]).all(), "First quarter should be unpadded"
    assert (
        padding_mask3[0, 250 * 8 : 375 * 8].sum() == 125 * 8
    ), "Middle quarter should be partially padded"
    assert (~padding_mask3[0, 375 * 8 :]).all(), "Rest should be unpadded"

    # Test case 4: All padding (fully masked input)
    padding_mask = torch.ones(1, 16000, dtype=torch.bool)
    padding_mask4 = forward_padding_mask_conv(padding_mask, n_dim, conv2d_module)
    assert padding_mask4.shape == (1, 1000 * 8)
    assert padding_mask4.all(), "All elements should be padded"

    # Test case 5: Edge case with very short input
    padding_mask = torch.zeros(
        1, 128, dtype=torch.bool
    )  # Only one stride worth of input
    padding_mask5 = forward_padding_mask_conv(padding_mask, n_dim, conv2d_module)
    assert padding_mask5.shape[1] <= 8 * 8, f"Unexpected shape: {padding_mask5.shape}"

    # Test case 6: Large input tensor (ensure scalability)
    padding_mask = torch.zeros(1, 64000, dtype=torch.bool)
    padding_mask[:, 32000:] = True  # Padding half of a long input
    padding_mask6 = forward_padding_mask_conv(padding_mask, n_dim, conv2d_module)
    assert padding_mask6.shape == (
        1,
        4000 * 8,
    ), f"Expected shape (1, 4000 * 8), got {padding_mask6.shape}"
    assert (~padding_mask6[0, : 2000 * 8]).all(), "First half should be unpadded"
    assert (
        padding_mask6[0, 2000 * 8 :].sum() == 2000 * 8
    ), "Second half should be fully padded"

    # Test case 7: Padding in irregular intervals
    padding_mask = torch.zeros(1, 16000, dtype=torch.bool)
    padding_mask[:, 2000:3000] = True
    padding_mask[:, 5000:7000] = True
    padding_mask[:, 9000:12000] = True
    padding_mask7 = forward_padding_mask_conv(padding_mask, n_dim, conv2d_module)
    assert padding_mask7.shape == (1, 1000 * 8)
    assert (
        padding_mask7[0, 125 * 8 : 187 * 8].sum() == 62 * 8
    ), "First irregular padding segment should be correct"
    assert (
        padding_mask7[0, 312 * 8 : 437 * 8].sum() == 125 * 8
    ), "Second irregular padding segment should be correct"
    assert (
        padding_mask7[0, 562 * 8 : 750 * 8].sum() == 188 * 8
    ), "Third irregular padding segment should be correct"

    # Test case 8: Batch input (multiple examples)
    padding_mask = torch.zeros(2, 16000, dtype=torch.bool)
    padding_mask[0, 8000:] = True
    padding_mask[1, 4000:12000] = True
    padding_mask8 = forward_padding_mask_conv(padding_mask, n_dim, conv2d_module)
    assert padding_mask8.shape == (2, 1000 * 8)
    assert (
        padding_mask8[0, : 500 * 8].sum() == 0
    ), "Batch 0: First half should be unpadded"
    assert (
        padding_mask8[0, 500 * 8 :].sum() == 500 * 8
    ), "Batch 0: Second half should be fully padded"
    assert (
        padding_mask8[1, : 250 * 8].sum() == 0
    ), "Batch 1: First quarter should be unpadded"
    assert (
        padding_mask8[1, 250 * 8 : 750 * 8].sum() == 500 * 8
    ), "Batch 1: Middle half should be fully padded"
    assert (
        padding_mask8[1, 750 * 8 :].sum() == 0
    ), "Batch 1: Last quarter should be unpadded"
