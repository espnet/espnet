"""Tests for the ``number`` kind."""

import pytest

from espnet3.api.inference import KINDS, Field, NumberKind


def test_number_kind_is_registered():
    assert KINDS["number"] is not None
    assert isinstance(KINDS["number"], NumberKind)


def test_number_kind_accepts_int_and_float():
    field = Field("WER", "number")
    assert NumberKind().check(4, field, model=None, output=True) == 4
    assert NumberKind().check(4.3, field, model=None, output=True) == 4.3


def test_number_kind_rejects_bool():
    field = Field("WER", "number")
    with pytest.raises(TypeError, match="must be int or float"):
        NumberKind().check(True, field, model=None, output=True)


def test_number_kind_rejects_str():
    field = Field("WER", "number")
    with pytest.raises(TypeError, match="must be int or float"):
        NumberKind().check("4.3", field, model=None, output=False)
