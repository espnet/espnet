"""Tests for the manifest contract added to DatasetBuilder."""

from pathlib import Path

import pytest

from espnet3.api.inference import Field
from espnet3.components.contract.dataset import DatasetContractError
from espnet3.components.data.dataset_builder import DatasetBuilder


class _NoopBuilder(DatasetBuilder):
    def is_source_prepared(self, **kwargs):
        return True

    def prepare_source(self, **kwargs):
        pass

    def is_built(self, **kwargs):
        return True

    def build(self, **kwargs):
        pass


def test_builder_without_manifest_columns_is_not_checked():
    _NoopBuilder().build()


def test_builder_with_declared_manifest_passes_matching_row(tmp_path: Path):
    class GoodBuilder(_NoopBuilder):
        manifest_columns = (Field("utt_id", "text"), Field("text", "text"))

        def build(self, **kwargs):
            (tmp_path / "train.tsv").write_text("utt1" + chr(9) + "hello")

        def built_manifests(self, **kwargs):
            return {"train": tmp_path / "train.tsv"}

    GoodBuilder().build()


def test_builder_rejects_manifest_with_wrong_column_count(tmp_path: Path):
    class BadBuilder(_NoopBuilder):
        manifest_columns = (
            Field("utt_id", "text"),
            Field("text", "text"),
            Field("extra", "text"),
        )

        def build(self, **kwargs):
            (tmp_path / "train.tsv").write_text("utt1" + chr(9) + "hello")

        def built_manifests(self, **kwargs):
            return {"train": tmp_path / "train.tsv"}

    with pytest.raises(DatasetContractError, match="row 1 has 2 columns"):
        BadBuilder().build()


def test_builder_raises_when_manifest_written_but_undeclared(tmp_path: Path):
    class UndeclaredBuilder(_NoopBuilder):
        def build(self, **kwargs):
            (tmp_path / "train.tsv").write_text("utt1" + chr(9) + "hello")

        def built_manifests(self, **kwargs):
            return {"train": tmp_path / "train.tsv"}

    with pytest.raises(DatasetContractError, match="does not declare manifest_columns"):
        UndeclaredBuilder().build()


def test_builder_rejects_malformed_class_manifest_columns_at_class_creation():
    with pytest.raises(TypeError, match="must name at least one field"):

        class Bad(_NoopBuilder):
            manifest_columns = ()


def test_builder_accepts_instance_level_manifest_columns(tmp_path: Path):
    class ConfigurableBuilder(_NoopBuilder):
        def __init__(self, extra_column):
            self.manifest_columns = (
                Field("utt_id", "text"),
                Field(extra_column, "text"),
            )

        def build(self, **kwargs):
            (tmp_path / "train.tsv").write_text("utt1" + chr(9) + "hello")

        def built_manifests(self, **kwargs):
            return {"train": tmp_path / "train.tsv"}

    ConfigurableBuilder("speaker").build()


def test_build_return_value_is_passed_through():
    class ReturningBuilder(_NoopBuilder):
        def build(self, **kwargs):
            return "built"

    assert ReturningBuilder().build() == "built"


def test_builder_accepts_a_positional_recipe_dir(tmp_path: Path):
    class GoodBuilder(_NoopBuilder):
        manifest_columns = (Field("utt_id", "text"), Field("text", "text"))

        def build(self, recipe_dir, **kwargs):
            (Path(recipe_dir) / "train.tsv").write_text("utt1" + chr(9) + "hello")

        def built_manifests(self, recipe_dir, **kwargs):
            return {"train": Path(recipe_dir) / "train.tsv"}

    GoodBuilder().build(str(tmp_path))


def test_builder_build_preserves_name_and_docstring():
    class GoodBuilder(_NoopBuilder):
        def build(self, **kwargs):
            """Build it."""

    assert GoodBuilder.build.__name__ == "build"
    assert GoodBuilder.build.__doc__ == "Build it."


def test_builder_checks_manifests_only_once_through_a_grandchild(
    tmp_path: Path, monkeypatch
):
    import espnet3.components.data.dataset_builder as dataset_builder_module

    calls = []
    real_check_manifests = dataset_builder_module.check_manifests

    def spy(*args, **kwargs):
        calls.append(1)
        return real_check_manifests(*args, **kwargs)

    monkeypatch.setattr(dataset_builder_module, "check_manifests", spy)

    class GoodBuilder(_NoopBuilder):
        manifest_columns = (Field("utt_id", "text"), Field("text", "text"))

        def build(self, **kwargs):
            (tmp_path / "train.tsv").write_text("utt1" + chr(9) + "hello")

        def built_manifests(self, **kwargs):
            return {"train": tmp_path / "train.tsv"}

    class Grandchild(GoodBuilder):
        pass

    Grandchild().build()
    assert len(calls) == 1
