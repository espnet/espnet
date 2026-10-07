"""Tests for espnet3.components.contract.dataset."""

from pathlib import Path

import numpy as np
import pytest

from espnet3.api.inference import Field
from espnet3.components.contract.dataset import (
    DatasetContractError,
    check_dataset_column_kind,
    check_fields,
    check_item,
    check_manifests,
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


# ---------------------------------------------------------------------------
# check_manifests
# ---------------------------------------------------------------------------


class _FakeBuilder:
    manifest_columns = (
        Field("utt_id", "text"),
        Field("wav", "audio"),
        Field("text", "text"),
    )
    manifest_header = False

    def __init__(self, manifests):
        self._manifests = manifests

    def built_manifests(self, **kwargs):
        return self._manifests


def test_check_manifests_accepts_matching_row(tmp_path: Path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    manifest = tmp_path / "train.tsv"
    manifest.write_text(f"utt1\t{wav}\thello\n", encoding="utf-8")

    check_manifests(_FakeBuilder({"train": manifest}))


def test_check_manifests_rejects_wrong_column_count(tmp_path: Path):
    manifest = tmp_path / "train.tsv"
    manifest.write_text("utt1\thello\n", encoding="utf-8")

    with pytest.raises(DatasetContractError, match="row 1 has 2 columns"):
        check_manifests(_FakeBuilder({"train": manifest}))


def test_check_manifests_rejects_missing_audio_file(tmp_path: Path):
    manifest = tmp_path / "train.tsv"
    manifest.write_text("utt1\t/no/such/file.wav\thello\n", encoding="utf-8")

    with pytest.raises(DatasetContractError, match="points to a missing file"):
        check_manifests(_FakeBuilder({"train": manifest}))


def test_check_manifests_skips_header_row(tmp_path: Path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    manifest = tmp_path / "train.tsv"
    manifest.write_text(f"id\twav\ttext\nutt1\t{wav}\thello\n", encoding="utf-8")

    class HeaderedBuilder(_FakeBuilder):
        manifest_header = True

    check_manifests(HeaderedBuilder({"train": manifest}))


def test_check_manifests_noop_when_builder_writes_no_manifest():
    class NoManifest:
        def built_manifests(self, **kwargs):
            return {}

    check_manifests(NoManifest())


def test_check_manifests_raises_when_manifests_but_undeclared(tmp_path: Path):
    manifest = tmp_path / "train.tsv"
    manifest.write_text("utt1\thello\n", encoding="utf-8")

    class Undeclared:
        def built_manifests(self, **kwargs):
            return {"train": manifest}

    with pytest.raises(DatasetContractError, match="does not declare manifest_columns"):
        check_manifests(Undeclared())


def test_check_manifests_accepts_instance_declaration(tmp_path: Path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    manifest = tmp_path / "train.tsv"
    manifest.write_text(f"utt1\t{wav}\thello\n", encoding="utf-8")

    class Configurable:
        def __init__(self, extra):
            self.manifest_columns = (
                Field("utt_id", "text"),
                Field("wav", "audio"),
                Field(extra, "text"),
            )
            self.manifest_header = False

        def built_manifests(self, **kwargs):
            return {"train": manifest}

    check_manifests(Configurable("text"))


# ---------------------------------------------------------------------------
# check_dataset_column_kind
# ---------------------------------------------------------------------------

_DATASET_FIELDS = (Field("speech", "audio"), Field("text", "text"))


def test_check_dataset_column_kind_accepts_matching_kind():
    check_dataset_column_kind(_DATASET_FIELDS, "text", "text", where="x")


def test_check_dataset_column_kind_rejects_mismatched_kind():
    with pytest.raises(DatasetContractError, match="wants 'audio'"):
        check_dataset_column_kind(_DATASET_FIELDS, "text", "audio", where="x")


def test_check_dataset_column_kind_rejects_no_declared_fields():
    with pytest.raises(DatasetContractError, match="declares no fields"):
        check_dataset_column_kind(None, "text", "text", where="x")


def test_check_dataset_column_kind_rejects_empty_declared_fields():
    with pytest.raises(DatasetContractError, match="declares no fields"):
        check_dataset_column_kind((), "text", "text", where="x")


def test_check_dataset_column_kind_rejects_undeclared_column():
    with pytest.raises(DatasetContractError, match="do not name column 'missing'"):
        check_dataset_column_kind(_DATASET_FIELDS, "missing", "text", where="x")
