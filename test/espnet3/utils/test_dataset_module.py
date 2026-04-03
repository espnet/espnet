from pathlib import Path

import pytest

from espnet3.components.data import dataset_module as dm


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _create_local_recipe_dataset(tmp_path: Path) -> None:
    _write(
        tmp_path / "dataset" / "__init__.py",
        "\n".join(
            [
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
                "class Dataset:",
                "    def __init__(self, **kwargs):",
                "        self.kwargs = kwargs",
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
                "class Dataset:",
                "    def __init__(self, **kwargs):",
                "        self.kwargs = kwargs",
            ]
        ),
    )
    return module_name


def test_resolve_dataset_module_name_supports_task_scoped_dataset_names():
    assert dm.resolve_dataset_module_name("mini_an4/asr") == "egs3.mini_an4.asr.dataset"


def test_load_dataset_module_accepts_full_module_path():
    # Full dotted module path must bypass tag resolution and import as-is.
    module = dm.load_dataset_module(ref="egs3.mini_an4.asr.dataset")
    assert hasattr(module, "Dataset")
    assert hasattr(module, "DatasetBuilder")


def test_is_tag_distinguishes_slash_from_dotted_path():
    assert dm._is_tag("mini_an4/asr") is True
    assert dm._is_tag("egs3.mini_an4.asr.dataset") is False


def test_instantiate_dataset_reference_keeps_explicit_split(monkeypatch):
    called = {}

    class DummyDataset:
        def __init__(self, **kwargs):
            called.update(kwargs)

    class DummyModule:
        Dataset = DummyDataset

    monkeypatch.setattr(
        dm,
        "load_dataset_module",
        lambda ref=None, recipe_dir=None: DummyModule(),
    )
    dm.instantiate_dataset_reference(
        {"ref": "mini_an4/asr", "kwargs": {"split": "valid"}},
        recipe_dir="/tmp/recipe",
    )

    assert called["split"] == "valid"
    assert "recipe_dir" not in called


def test_load_dataset_module_without_ref_uses_recipe_dir_dataset(tmp_path):
    _create_local_recipe_dataset(tmp_path)

    module = dm.load_dataset_module(ref=None, recipe_dir=tmp_path)

    assert hasattr(module, "Dataset")
    assert hasattr(module, "DatasetBuilder")


def test_instantiate_dataset_reference_without_ref_keeps_kwargs(tmp_path):
    _create_local_recipe_dataset(tmp_path)

    dataset = dm.instantiate_dataset_reference(
        {"kwargs": {"custom_arg": 10}},
        recipe_dir=tmp_path,
    )

    assert "split" not in dataset.kwargs
    assert dataset.kwargs["custom_arg"] == 10
    assert "recipe_dir" not in dataset.kwargs


def test_tag_ref_resolves_and_passes_kwargs(monkeypatch):
    captured = {}

    class DummyDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyModule:
        Dataset = DummyDataset

    def fake_import_module(module_name):
        captured["module_name"] = module_name
        return DummyModule()

    monkeypatch.setattr(dm, "import_module", fake_import_module)

    dataset = dm.instantiate_dataset_reference(
        {"ref": "mini_an4/asr", "kwargs": {"split": "valid", "extra_arg": "ok"}},
    )

    assert captured["module_name"] == "egs3.mini_an4.asr.dataset"
    assert dataset.kwargs["split"] == "valid"
    assert dataset.kwargs["extra_arg"] == "ok"


def test_module_path_ref_resolves_direct_module(tmp_path, monkeypatch):
    module_ref = _create_module_path_dataset(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    dataset = dm.instantiate_dataset_reference(
        {"ref": module_ref, "kwargs": {"split": "test", "sample_rate": 16000}},
        recipe_dir=tmp_path,
    )

    assert dataset.kwargs["split"] == "test"
    assert dataset.kwargs["sample_rate"] == 16000


def test_instantiate_dataset_reference_passes_only_kwargs_to_dataset(monkeypatch):
    captured = {}

    class DummyDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class DummyModule:
        Dataset = DummyDataset

    monkeypatch.setattr(
        dm,
        "load_dataset_module",
        lambda ref=None, recipe_dir=None: DummyModule(),
    )
    dm.instantiate_dataset_reference(
        {
            "ref": "mini_an4/asr",
            "name": "train_set",
            "transform": {"_target_": "dummy.Transform"},
            "split": "train",
            "dataset_path": "/tmp/data",
            "kwargs": {"split": "train", "num_workers": 4},
        },
        recipe_dir="/tmp/recipe",
    )

    assert captured == {
        "split": "train",
        "num_workers": 4,
    }


def test_load_dataset_module_without_ref_raises_if_local_dataset_missing(tmp_path):
    with pytest.raises(ModuleNotFoundError):
        dm.load_dataset_module(ref=None, recipe_dir=tmp_path)


def test_load_dataset_module_without_ref_and_recipe_dir_raises_assertion():
    with pytest.raises(AssertionError, match="recipe_dir must be set"):
        dm.load_dataset_module(ref=None, recipe_dir=None)


def test_instantiate_dataset_reference_blank_ref_uses_local_module(tmp_path):
    _create_local_recipe_dataset(tmp_path)

    dataset = dm.instantiate_dataset_reference(
        {"ref": "   ", "kwargs": {"custom_arg": 1}},
        recipe_dir=tmp_path,
    )

    assert dataset.kwargs["custom_arg"] == 1


def test_instantiate_dataset_reference_invalid_kwargs_raises_type_error(tmp_path):
    _create_local_recipe_dataset(tmp_path)

    with pytest.raises(ValueError):
        dm.instantiate_dataset_reference(
            {"kwargs": "not-a-mapping"},
            recipe_dir=tmp_path,
        )
