"""Tests for espnet3.components.contract.dataset."""

import numpy as np
import pytest

from espnet3.api.inference import Field
from espnet3.components.contract.dataset import (
    DatasetContractError,
    check_fields,
    check_item,
    require_fields,
)

# ---------------------------------------------------------------------------
# check_fields
# ---------------------------------------------------------------------------


def test_check_fields_returns_well_formed_tuple():
    class Good:
        fields = (Field("speech", "audio"), Field("text", "text"))

    assert check_fields(Good, "fields") == Good.fields


def test_check_fields_returns_none_when_undeclared():
    class Undeclared:
        pass

    assert check_fields(Undeclared, "fields") is None


def test_check_fields_accepts_instance_attribute():
    class Configurable:
        def __init__(self, extra):
            self.fields = (Field("speech", "audio"), Field(extra, "text"))

    obj = Configurable("text")
    assert check_fields(obj, "fields") == obj.fields


def test_check_fields_rejects_non_tuple():
    class Bad:
        fields = [Field("speech", "audio")]

    with pytest.raises(TypeError, match="must be a tuple of Field"):
        check_fields(Bad, "fields")


def test_check_fields_rejects_empty_tuple():
    class Bad:
        fields = ()

    with pytest.raises(TypeError, match="must name at least one field"):
        check_fields(Bad, "fields")


def test_check_fields_rejects_duplicate_names():
    class Bad:
        fields = (Field("speech", "audio"), Field("speech", "text"))

    with pytest.raises(TypeError, match="repeats a name"):
        check_fields(Bad, "fields")


# ---------------------------------------------------------------------------
# require_fields
# ---------------------------------------------------------------------------


def test_require_fields_always_raises():
    class Undeclared:
        pass

    with pytest.raises(TypeError, match="does not declare fields"):
        require_fields(Undeclared, "fields")


# ---------------------------------------------------------------------------
# check_item
# ---------------------------------------------------------------------------

_FIELDS = (Field("speech", "audio"), Field("text", "text"))


def test_check_item_accepts_matching_item():
    check_item(
        _FIELDS, {"speech": np.zeros(16000, dtype=np.float32), "text": "hi"}, "x"
    )


def test_check_item_allows_undeclared_extra_keys():
    check_item(
        _FIELDS,
        {"speech": np.zeros(16000, dtype=np.float32), "text": "hi", "utt_id": "u1"},
        "x",
    )


def test_check_item_rejects_non_mapping():
    with pytest.raises(DatasetContractError, match="item must be a dict"):
        check_item(_FIELDS, ["not", "a", "dict"], "x")


def test_check_item_rejects_missing_field():
    with pytest.raises(DatasetContractError, match="lacks declared field 'text'"):
        check_item(_FIELDS, {"speech": np.zeros(16000, dtype=np.float32)}, "x")


def test_check_item_rejects_wrong_kind():
    with pytest.raises(DatasetContractError, match="declared audio but the item holds"):
        check_item(_FIELDS, {"speech": 42, "text": "hi"}, "x")


def test_check_item_allows_missing_optional_field():
    fields = (Field("speech", "audio"), Field("prompt", "text", optional=True))
    check_item(fields, {"speech": np.zeros(16000, dtype=np.float32)}, "x")
