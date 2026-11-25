from pathlib import Path

import pytest

from espnet3.components.data import dataset_module as dm
from espnet3.components.data.base_dataset import BaseDataset
from espnet3.components.data.data_organizer import DatasetConfig
from espnet3.contract import Field

_BASE_IMPORT = "from espnet3.components.data.base_dataset import BaseDataset"
_FIELD_IMPORT = "from espnet3.contract import Field"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _create_local_recipe_dataset(tmp_path: Path) -> None:
    _write(
        tmp_path / "dataset" / "__init__.py",
        "\n".join(
            [
                _BASE_IMPORT,
                _FIELD_IMPORT,
                "class DatasetBuilder:",
                "    def is_source_prepared(self, **kwargs):",
                "        return True",
                "",
                "    def prepare_source(self, **kwargs):",
                "        return None",
                "",
                "    def is_built(self, **kwargs):",
                "        return True",
                "",
                "    def build(self, **kwargs):",
                "        return None",
                "",
                "class Dataset(BaseDataset):",
                '    fields = (Field("text", "text"),)',
                "    def __init__(self, **kwargs):",
                "        self.kwargs = kwargs",
                "    def __getitem__(self, index):",
                "        return {'text': 'hi'}",
                "",
                "def get_data_root(recipe_dir=None):",
                "    from pathlib import Path",
                "    return Path(recipe_dir) / 'prepared_root'",
            ]
        ),
    )


def _create_module_path_dataset(tmp_path: Path) -> str:
    module_name = "tests_dataset_module_path"
    _write(
        tmp_path / f"{module_name}.py",
        "\n".join(
            [
                _BASE_IMPORT,
                _FIELD_IMPORT,
                "class DatasetBuilder:",
                "    def is_source_prepared(self, **kwargs):",
                "        return True",
                "",
                "    def prepare_source(self, **kwargs):",
                "        return None",
                "",
                "    def is_built(self, **kwargs):",
                "        return True",
                "",
                "    def build(self, **kwargs):",
                "        return None",
                "",
                "class Dataset(BaseDataset):",
                '    fields = (Field("text", "text"),)',
                "    def __init__(self, **kwargs):",
                "        self.kwargs = kwargs",
                "    def __getitem__(self, index):",
                "        return {'text': 'hi'}",
            ]
        ),
    )
    return module_name


def test_resolve_dataset_module_name_supports_task_scoped_dataset_names():
    assert (
        dm.resolve_dataset_module_name("mini_an4/esp2_asr")
        == "egs3.mini_an4.esp2_asr.dataset"
    )


def test_load_dataset_module_accepts_full_module_path():
    # Full dotted module path must bypass tag resolution and import as-is.
    module = dm.load_dataset_module(data_src="egs3.mini_an4.esp2_asr.dataset")
    assert hasattr(module, "Dataset")
    assert hasattr(module, "DatasetBuilder")


def test_load_dataset_module_rejects_missing_module():
    with pytest.raises(ModuleNotFoundError):
        dm.load_dataset_module(data_src="no.such.dataset.module")


def test_is_tag_distinguishes_slash_from_dotted_path():
    assert dm._is_tag("mini_an4/esp2_asr") is True
    assert dm._is_tag("egs3.mini_an4.esp2_asr.dataset") is False


def test_instantiate_dataset_reference_keeps_explicit_split(monkeypatch):
    called = {}

    class DummyDataset(BaseDataset):
        fields = (Field("text", "text"),)

        def __init__(self, **kwargs):
            called.update(kwargs)

        def __getitem__(self, index):
            return {"text": "hi"}

    class DummyModule:
        Dataset = DummyDataset

    monkeypatch.setattr(
        dm,
        "resolve_data_src",
        lambda data_src=None, recipe_dir=None: (DummyModule(), None),
    )
    dm.instantiate_dataset_reference(
        {"data_src": "mini_an4/esp2_asr", "data_src_args": {"split": "valid"}},
        recipe_dir="/tmp/recipe",
    )

    assert called["split"] == "valid"
    assert "recipe_dir" not in called


def test_load_dataset_module_without_ref_uses_recipe_dir_dataset(tmp_path):
    _create_local_recipe_dataset(tmp_path)

    module = dm.load_dataset_module(data_src=None, recipe_dir=tmp_path)

    assert hasattr(module, "Dataset")
    assert hasattr(module, "DatasetBuilder")


def test_instantiate_dataset_reference_without_ref_keeps_kwargs(tmp_path):
    _create_local_recipe_dataset(tmp_path)

    dataset = dm.instantiate_dataset_reference(
        {"data_src_args": {"custom_arg": 10}},
        recipe_dir=tmp_path,
    )

    assert "split" not in dataset.kwargs
    assert dataset.kwargs["custom_arg"] == 10
    assert "recipe_dir" not in dataset.kwargs


def test_tag_ref_resolves_and_passes_kwargs(monkeypatch):
    captured = {}

    class DummyDataset(BaseDataset):
        fields = (Field("text", "text"),)

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __getitem__(self, index):
            return {"text": "hi"}

    class DummyModule:
        Dataset = DummyDataset

    def fake_import_module(module_name):
        captured["module_name"] = module_name
        return DummyModule()

    monkeypatch.setattr(dm, "import_module", fake_import_module)

    dataset = dm.instantiate_dataset_reference(
        {
            "data_src": "mini_an4/esp2_asr",
            "data_src_args": {"split": "valid", "extra_arg": "ok"},
        },
    )

    assert captured["module_name"] == "egs3.mini_an4.esp2_asr.dataset"
    assert dataset.kwargs["split"] == "valid"
    assert dataset.kwargs["extra_arg"] == "ok"


def test_module_path_ref_resolves_direct_module(tmp_path, monkeypatch):
    module_ref = _create_module_path_dataset(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    dataset = dm.instantiate_dataset_reference(
        {
            "data_src": module_ref,
            "data_src_args": {"split": "test", "sample_rate": 16000},
        },
        recipe_dir=tmp_path,
    )

    assert dataset.kwargs["split"] == "test"
    assert dataset.kwargs["sample_rate"] == 16000


def test_instantiate_dataset_reference_passes_only_kwargs_to_dataset(monkeypatch):
    captured = {}

    class DummyDataset(BaseDataset):
        fields = (Field("text", "text"),)

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __getitem__(self, index):
            return {"text": "hi"}

    class DummyModule:
        Dataset = DummyDataset

    monkeypatch.setattr(
        dm,
        "resolve_data_src",
        lambda data_src=None, recipe_dir=None: (DummyModule(), None),
    )
    dm.instantiate_dataset_reference(
        {
            "data_src": "mini_an4/esp2_asr",
            "name": "train_set",
            "transform": {"_target_": "dummy.Transform"},
            "split": "train",
            "dataset_path": "/tmp/data",
            "data_src_args": {"split": "train", "num_workers": 4},
        },
        recipe_dir="/tmp/recipe",
    )

    assert captured == {
        "split": "train",
        "num_workers": 4,
    }


def test_load_dataset_module_without_ref_raises_if_local_dataset_missing(tmp_path):
    with pytest.raises(ModuleNotFoundError):
        dm.load_dataset_module(data_src=None, recipe_dir=tmp_path)


def test_load_dataset_module_without_ref_and_recipe_dir_raises_assertion():
    with pytest.raises(AssertionError, match="recipe_dir must be set"):
        dm.load_dataset_module(data_src=None, recipe_dir=None)


def test_parse_dataset_reference_config_keeps_data_src_args():
    data_src, data_src_args = dm.parse_dataset_reference_config(
        {
            "data_src": "mini_an4/esp2_asr",
            "name": "train",
            "data_src_args": {"split": "train"},
        }
    )

    assert data_src == "mini_an4/esp2_asr"
    assert data_src_args == {"split": "train"}


def test_parse_dataset_reference_config_omitted_data_src_args_parses():
    """A plain dict omitting data_src_args must not fail dict() conversion."""
    data_src, data_src_args = dm.parse_dataset_reference_config(
        {"data_src": "mini_an4/asr"}
    )

    assert data_src == "mini_an4/asr"
    assert data_src_args == {}


def test_parse_dataset_reference_config_dataset_config_default_parses():
    """DatasetConfig defaults data_src_args to None; parsing must still succeed."""
    data_src, data_src_args = dm.parse_dataset_reference_config(
        DatasetConfig(data_src="mini_an4/asr")
    )

    assert data_src == "mini_an4/asr"
    assert data_src_args == {}


def test_instantiate_dataset_reference_blank_data_src_uses_local_module(tmp_path):
    _create_local_recipe_dataset(tmp_path)

    dataset = dm.instantiate_dataset_reference(
        {"data_src": "   ", "data_src_args": {"custom_arg": 1}},
        recipe_dir=tmp_path,
    )

    assert dataset.kwargs["custom_arg"] == 1


def test_instantiate_dataset_reference_invalid_kwargs_raises_type_error(tmp_path):
    _create_local_recipe_dataset(tmp_path)

    with pytest.raises(ValueError):
        dm.instantiate_dataset_reference(
            {"data_src_args": "not-a-mapping"},
            recipe_dir=tmp_path,
        )


def test_instantiate_dataset_reference_rejects_dataset_outside_base(monkeypatch):
    class NotBaseDataset:
        def __init__(self, **kwargs):
            raise AssertionError("must not be instantiated before the base check")

    class DummyModule:
        Dataset = NotBaseDataset

    monkeypatch.setattr(
        dm,
        "resolve_data_src",
        lambda data_src=None, recipe_dir=None: (DummyModule(), None),
    )
    with pytest.raises(TypeError, match="does not inherit BaseDataset"):
        dm.instantiate_dataset_reference({"data_src_args": {}})


# -------- data_src naming a dataset class --------


def _create_class_path_module(tmp_path: Path, monkeypatch, module_name: str) -> str:
    _write(
        tmp_path / f"{module_name}.py",
        "\n".join(
            [
                _BASE_IMPORT,
                _FIELD_IMPORT,
                "class ExplicitDataset(BaseDataset):",
                '    fields = (Field("text", "text"),)',
                "    def __init__(self, **kwargs):",
                "        self.kwargs = kwargs",
                "    def __getitem__(self, index):",
                "        return {'text': 'explicit'}",
                "",
                "class Dataset(ExplicitDataset):",
                "    pass",
                "",
                "class NotADataset:",
                "    def __init__(self, **kwargs):",
                "        raise AssertionError('must not be instantiated')",
                "",
                "def make_dataset(**kwargs):",
                "    raise AssertionError('must not be called')",
            ]
        ),
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    return module_name


def test_class_path_ref_instantiates_named_class(tmp_path, monkeypatch):
    module_ref = _create_class_path_module(tmp_path, monkeypatch, "tests_dm_explicit")

    dataset = dm.instantiate_dataset_reference(
        {
            "data_src": f"{module_ref}.ExplicitDataset",
            "data_src_args": {"split": "test", "sample_rate": 24000},
        }
    )

    assert type(dataset).__name__ == "ExplicitDataset"
    assert dataset.kwargs == {"split": "test", "sample_rate": 24000}


def test_class_path_ref_loads_defining_module(tmp_path, monkeypatch):
    module_ref = _create_class_path_module(tmp_path, monkeypatch, "tests_dm_defining")

    module = dm.load_dataset_module(data_src=f"{module_ref}.ExplicitDataset")

    assert module.__name__ == module_ref


def test_class_path_ref_resolves_class_in_repository():
    module, dataset_cls = dm.resolve_data_src("egs3.mini_an4.esp2_asr.dataset.Dataset")

    assert module.__name__ == "egs3.mini_an4.esp2_asr.dataset"
    assert dataset_cls is module.Dataset


def test_module_tag_and_local_refs_still_use_module_dataset(tmp_path, monkeypatch):
    module_ref = _create_class_path_module(tmp_path, monkeypatch, "tests_dm_module")
    _create_local_recipe_dataset(tmp_path)

    module_dataset = dm.instantiate_dataset_reference({"data_src": module_ref})
    tag_module, tag_cls = dm.resolve_data_src("mini_an4/esp2_asr")
    local_dataset = dm.instantiate_dataset_reference({}, recipe_dir=tmp_path)

    assert type(module_dataset).__name__ == "Dataset"
    assert tag_module.__name__ == "egs3.mini_an4.esp2_asr.dataset"
    assert tag_cls is None
    assert local_dataset[0] == {"text": "hi"}


@pytest.mark.parametrize(
    "attr_name, match",
    [
        ("NotADataset", "does not inherit BaseDataset"),
        ("make_dataset", "is not a class"),
    ],
)
def test_class_path_ref_rejects_invalid_class(tmp_path, monkeypatch, attr_name, match):
    module_ref = _create_class_path_module(
        tmp_path, monkeypatch, f"tests_dm_invalid_{attr_name.lower()}"
    )

    with pytest.raises(TypeError, match=match):
        dm.instantiate_dataset_reference({"data_src": f"{module_ref}.{attr_name}"})


def test_class_path_ref_reports_missing_class_name(tmp_path, monkeypatch):
    module_ref = _create_class_path_module(tmp_path, monkeypatch, "tests_dm_missing")

    with pytest.raises(
        ImportError, match="no attribute or submodule 'NoSuchDataset'"
    ) as info:
        dm.instantiate_dataset_reference({"data_src": f"{module_ref}.NoSuchDataset"})

    assert type(info.value) is ImportError


def test_class_path_ref_reports_missing_module(tmp_path):
    with pytest.raises(ModuleNotFoundError, match="no module named 'no_such_pkg'"):
        dm.instantiate_dataset_reference({"data_src": "no_such_pkg.sub.SomeDataset"})


@pytest.mark.parametrize("suffix", ["", ".ExplicitDataset"])
def test_missing_dependency_of_dataset_module_is_not_hidden(
    tmp_path, monkeypatch, suffix
):
    _write(tmp_path / "tests_dm_pkg" / "__init__.py", "")
    _write(
        tmp_path / "tests_dm_pkg" / "broken.py",
        "\n".join(
            [
                "import tests_dm_absent_dependency  # noqa: F401",
                _BASE_IMPORT,
                "class ExplicitDataset(BaseDataset):",
                "    pass",
            ]
        ),
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ModuleNotFoundError) as info:
        dm.instantiate_dataset_reference({"data_src": f"tests_dm_pkg.broken{suffix}"})

    assert info.value.name == "tests_dm_absent_dependency"


def test_import_time_error_of_dataset_module_is_not_wrapped(tmp_path, monkeypatch):
    _write(tmp_path / "tests_dm_raising.py", "raise RuntimeError('corpus root unset')")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(RuntimeError, match="corpus root unset"):
        dm.instantiate_dataset_reference(
            {"data_src": "tests_dm_raising.ExplicitDataset"}
        )
