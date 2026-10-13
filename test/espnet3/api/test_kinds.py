"""Tests for the ``number`` kind and ``Kind.accepts``."""

import numpy as np
import pytest

from espnet3.api.inference import KINDS, AudioKind, Field, NumberKind, TextKind


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


def test_kind_accepts_defaults_to_check_with_no_model():
    field = Field("text", "text")
    assert TextKind().accepts("hi", field) is True
    assert TextKind().accepts(7, field) is False


@pytest.mark.parametrize(
    "value",
    [
        np.zeros(16000, dtype=np.float32),
        "a.wav",
        (16000, np.zeros(16000, dtype=np.float32)),
    ],
)
def test_audio_kind_accepts_rate_less_forms(value):
    field = Field("speech", "audio")
    assert AudioKind().accepts(value, field) is True


def test_audio_kind_accepts_rejects_unrelated_value():
    field = Field("speech", "audio")
    assert AudioKind().accepts(42, field) is False
