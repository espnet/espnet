from types import SimpleNamespace

import joblib
import numpy as np
import pytest
import torch

from espnet2.tok.quantizer.differentiable_kmeans import (
    DifferentiableKMeans,
)


@pytest.fixture
def centroid_path(tmp_path):
    centers = np.asarray([[0.0, 0.0], [2.0, 2.0], [-2.0, -2.0]], dtype=np.float32)
    path = tmp_path / "km_3.mdl"
    joblib.dump(SimpleNamespace(cluster_centers_=centers), path)
    return path, centers


def test_loads_offline_kmeans_centroids(centroid_path):
    path, centers = centroid_path
    quantizer = DifferentiableKMeans(path)

    assert quantizer.num_clusters == 3
    assert quantizer.feature_dim == 2
    assert quantizer.centroids.requires_grad
    torch.testing.assert_close(quantizer.centroids, torch.from_numpy(centers))


def test_encode_matches_nearest_centroid_and_masks_padding(centroid_path):
    path, _ = centroid_path
    quantizer = DifferentiableKMeans(path)
    features = torch.tensor(
        [[[0.1, 0.1], [1.9, 2.1], [-1.8, -2.2]], [[2.1, 2.0], [8.0, 8.0], [8.0, 8.0]]]
    )
    lengths = torch.tensor([3, 1])

    output = quantizer.encode(features, lengths)

    assert output.token_ids.tolist() == [[0, 1, 2], [1, 0, 0]]
    assert output.assignment.shape == (2, 3, 3)
    assert output.assignment[0].sum(dim=-1).tolist() == [1.0, 1.0, 1.0]
    assert output.assignment[1, 1:].sum() == 0.0
    assert output.soft_assignment[1, 1:].sum() == 0.0


def test_forward_is_hard_and_backpropagates(centroid_path):
    path, _ = centroid_path
    quantizer = DifferentiableKMeans(path, temperature_init=2.0)
    features = torch.randn(2, 4, 2, requires_grad=True)
    lengths = torch.tensor([4, 3])
    embedding = torch.randn(3, 5)

    output = quantizer(features, lengths)
    embedded = torch.matmul(output.assignment, embedding)
    embedded.square().sum().backward()

    valid_assignment = output.assignment[
        torch.arange(4).unsqueeze(0) < lengths.unsqueeze(1)
    ]
    assert torch.all((valid_assignment == 0.0) | (valid_assignment == 1.0))
    assert torch.all(valid_assignment.sum(dim=-1) == 1.0)
    assert features.grad is not None
    assert torch.count_nonzero(features.grad) > 0
    assert quantizer.centroids.grad is not None
    assert torch.count_nonzero(quantizer.centroids.grad) > 0


def test_rejects_incompatible_feature_dimension(centroid_path):
    path, _ = centroid_path
    quantizer = DifferentiableKMeans(path)

    with pytest.raises(ValueError, match="does not match centroid"):
        quantizer.encode(torch.randn(1, 2, 3), torch.tensor([2]))


def test_temperature_must_be_positive(centroid_path):
    path, _ = centroid_path
    quantizer = DifferentiableKMeans(path)

    with pytest.raises(ValueError, match="temperature must be positive"):
        quantizer.set_temperature(0.0)


def test_temperature_anneals_only_during_training(centroid_path):
    path, _ = centroid_path
    quantizer = DifferentiableKMeans(
        path,
        temperature_init=2.0,
        temperature_floor=0.5,
        temperature_decay=0.5,
    )
    features = torch.randn(1, 2, 2)
    lengths = torch.tensor([2])

    quantizer.train()
    quantizer(features, lengths)
    assert quantizer.temperature.item() == pytest.approx(2.0)
    assert quantizer.num_updates.item() == 1
    quantizer(features, lengths)
    assert quantizer.temperature.item() == pytest.approx(1.0)
    assert quantizer.num_updates.item() == 2
    quantizer(features, lengths)
    assert quantizer.temperature.item() == pytest.approx(0.5)
    assert quantizer.num_updates.item() == 3
    quantizer(features, lengths)
    assert quantizer.temperature.item() == pytest.approx(0.5)

    quantizer.eval()
    quantizer(features, lengths)
    assert quantizer.temperature.item() == pytest.approx(0.5)
    assert quantizer.num_updates.item() == 4


@pytest.mark.parametrize("distance_type", ["euclidean", "squared_euclidean"])
def test_distance_types_preserve_nearest_centroid(centroid_path, distance_type):
    path, _ = centroid_path
    quantizer = DifferentiableKMeans(path, distance_type=distance_type)
    features = torch.tensor([[[0.1, 0.1], [1.9, 2.1], [-1.8, -2.2]]])

    output = quantizer.encode(features, torch.tensor([3]))

    assert output.token_ids.tolist() == [[0, 1, 2]]
