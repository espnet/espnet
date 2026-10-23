"""Tests for the ``vector`` kind."""

import numpy as np
import pytest
import torch

from espnet3.contract import KINDS, Field
from espnet3.contract.kinds.vector import VectorKind

FIELD = Field("spembs", "vector")


def test_vector_is_a_registered_kind():
    """A Field can declare the kind, and the registry exports it."""
    assert isinstance(KINDS["vector"], VectorKind)
    assert FIELD.kind == "vector"
    assert FIELD.label == "Spembs"


@pytest.mark.parametrize(
    "value",
    [
        np.ones(192, dtype=np.float32),
        np.arange(3),
        torch.ones(4),
        [0.0, 1.0, 2.0],
        (1, 2),
    ],
)
def test_check_returns_a_1d_numpy_array(value):
    """Arrays, tensors and number sequences come back as a 1-D NumPy array."""
    out = VectorKind().check(value, FIELD, None, output=False)
    assert isinstance(out, np.ndarray)
    assert out.ndim == 1
    assert out.shape[0] == len(value)


def test_check_keeps_the_dtype_of_an_array():
    """A float32 embedding is not silently upcast."""
    out = VectorKind().check(np.ones(2, dtype=np.float32), FIELD, None, output=True)
    assert out.dtype == np.float32


@pytest.mark.parametrize(
    "value, message",
    [
        (np.ones((2, 3)), "2 dimensions"),
        (np.float32(1.0), "0 dimensions"),
        (np.array([], dtype=np.float32), "empty"),
        (np.array([True, False]), "dtype bool"),
        (np.array([1 + 1j]), "dtype complex"),
        (["a", "b"], "dtype"),
        ("abc", "0 dimensions"),
        ([[1.0], [2.0, 3.0]], "must be a 1-D numeric vector"),
        (object(), "0 dimensions"),
    ],
)
def test_check_refuses_what_is_not_a_numeric_vector(value, message):
    """The wrong shape, dtype or type is a TypeError naming the field."""
    with pytest.raises(TypeError, match=f"'spembs'.*{message}"):
        VectorKind().check(value, FIELD, None, output=False)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_check_refuses_non_finite_values(bad):
    """NaN and infinities are a ValueError."""
    with pytest.raises(ValueError, match="NaN or Inf"):
        VectorKind().check(np.array([1.0, bad]), FIELD, None, output=True)


def test_error_names_the_direction():
    """Inputs are 'given', outputs are 'returned'."""
    with pytest.raises(TypeError, match="given"):
        VectorKind().check(np.ones((1, 1)), FIELD, None, output=False)
    with pytest.raises(TypeError, match="returned"):
        VectorKind().check(np.ones((1, 1)), FIELD, None, output=True)


def test_accepts_follows_check_without_a_model():
    """A dataset item's embedding is accepted; a matrix is not."""
    kind = VectorKind()
    assert kind.accepts(np.ones(3, dtype=np.float32), FIELD)
    assert kind.accepts(torch.ones(3), FIELD)
    assert not kind.accepts(np.ones((2, 3)), FIELD)
    assert not kind.accepts("abc", FIELD)


def test_a_list_of_vectors_is_a_batch():
    """One vector is one value; a list of them is one entry per sample."""
    kind = VectorKind()
    assert not kind.is_batch([0.0, 1.0], FIELD, None)
    assert kind.is_batch([np.ones(2), np.ones(2)], FIELD, None)


def test_vector_does_not_join():
    """Embeddings have no stream to join."""
    with pytest.raises(NotImplementedError):
        VectorKind().join(np.ones(2), np.ones(2))
