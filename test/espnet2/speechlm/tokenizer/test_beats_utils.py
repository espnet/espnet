import pytest
import torch

from espnet2.speechlm.tokenizer.beats_utils import (
    kmeans,
    sample_vectors,
)


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
