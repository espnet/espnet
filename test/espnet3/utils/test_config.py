import os

import pytest
from yaml.parser import ParserError

from espnet3.utils.config_utils import load_config_with_defaults, load_line

# ===============================================================
# Test Case Summary for Config Utilities
# ===============================================================
#
# Tests for `load_line(path)`
# | Test Name                  | Description                      |
# |----------------------------|----------------------------------|
# | test_load_line_basic       | Reads multiline like "a\nb\nc" |
# | test_load_line_with_whitespace | Strips surrounding whitespace   |
# | test_load_line_empty_file  | Empty file → empty list          |
# | test_load_line_single_line | Handles single-line file         |
# | test_load_line_trailing_newline | Handles trailing newline        |
#
# Simple Tests for `load_config_with_defaults`
# | Test Name                      | Description                  |
# |--------------------------------|------------------------------|
# | test_config_without_defaults   | Loads config as-is           |
# | test_config_with_self_only     | defaults: [_self_] merges    |
# | test_config_with_one_include   | Includes one external file   |
# | test_config_with_key_value     | defaults: [{opt: adam}]      |
# | test_config_update_with_key_value | Merges extra nested fields   |
# | test_config_ignore_none_entry  | Skips null defaults entries  |
# | test_defaults_removed_after_merge | Drops defaults key           |
#
# Complex/Recursive Merging
# | Test Name                      | Description                  |
# |--------------------------------|------------------------------|
# | test_config_with_nested_defaults | Resolves nested defaults      |
# | test_config_with_self_in_middle | `_self_` in middle respected |
#
# Dataset target behavior
# | Test Name                               | Description            |
# |-----------------------------------------|------------------------|
# | test_config_does_not_infer_dataset_target | No implicit targets |
#
# Error Cases
# | Test Name                 | Expected Exception        |
# |---------------------------|---------------------------|
# | test_missing_file_raises  | FileNotFoundError         |
# | test_invalid_yaml_raises  | ParserError               |


@pytest.fixture
def tmp_txt_file(tmp_path):
    def _create(content: str, filename: str = "test.txt"):
        file_path = tmp_path / filename
        file_path.write_text(content)
        return file_path

    return _create


def test_load_line_basic(tmp_txt_file):
    path = tmp_txt_file("a\nb\nc\n")
    result = load_line(path)
    assert result == ["a", "b", "c"]


def test_load_line_with_whitespace(tmp_txt_file):
    path = tmp_txt_file("  x  \n y\n")
    result = load_line(path)
    assert result == ["x", "y"]


def test_load_line_empty_file(tmp_txt_file):
    path = tmp_txt_file("")
    result = load_line(path)
    assert result == []


def test_load_line_single_line(tmp_txt_file):
    path = tmp_txt_file("hello")
    result = load_line(path)
    assert result == ["hello"]


def test_load_line_trailing_newline(tmp_txt_file):
    path = tmp_txt_file("last\n")
    result = load_line(path)
    assert result == ["last"]


@pytest.fixture
def write_yaml(tmp_path):
    """Fixture to create a YAML file under tmp_path and return its path.

    Usage: path = write_yaml("filename.yaml", content_as_str)
    """

    def _write(filename: str, content: str):
        path = tmp_path / filename
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))

        path.write_text(content)
        return path

    return _write


def test_config_without_defaults(write_yaml):
    path = write_yaml("config.yaml", "foo: bar\n")
    cfg = load_config_with_defaults(str(path))
    assert cfg.foo == "bar"


def test_config_with_self_only(write_yaml):
    content = """
defaults:
  - _self_
bar: 123
"""
    path = write_yaml("config.yaml", content)
    cfg = load_config_with_defaults(str(path))
    assert cfg.bar == 123
    assert "defaults" not in cfg


def test_config_with_one_include(write_yaml):
    write_yaml("base.yaml", "a: 1\n")
    main = write_yaml(
        "main.yaml",
        """
defaults:
  - base
b: 2
""",
    )
    cfg = load_config_with_defaults(str(main))
    assert cfg.a == 1
    assert cfg.b == 2
    assert "defaults" not in cfg


def test_config_with_key_value(write_yaml):
    write_yaml("optimizer/adam.yaml", "lr: 0.001\n")
    path = write_yaml(
        "config.yaml",
        """
defaults:
  - optimizer: adam
weight_decay: 0.01
""",
    )
    cfg = load_config_with_defaults(str(path))
    assert cfg.optimizer.lr == 0.001
    assert cfg.weight_decay == 0.01


def test_config_update_with_key_value(write_yaml):
    write_yaml("optimizer/adam.yaml", "lr: 0.001\n")
    path = write_yaml(
        "config.yaml",
        """
defaults:
  - optimizer: adam
optimizer:
  weight_decay: 0.01
""",
    )
    cfg = load_config_with_defaults(str(path))
    assert cfg.optimizer.lr == 0.001
    assert cfg.optimizer.weight_decay == 0.01


def test_config_with_nested_defaults(write_yaml):
    write_yaml(
        "base.yaml",
        """
defaults:
  - _self_
foo: 42
""",
    )
    write_yaml(
        "model.yaml",
        """
defaults:
  - base
bar: 99
""",
    )
    path = write_yaml(
        "main.yaml",
        """
defaults:
  - model
baz: 7
""",
    )
    cfg = load_config_with_defaults(str(path))
    assert cfg.foo == 42
    assert cfg.bar == 99
    assert cfg.baz == 7


def test_config_with_self_in_middle(write_yaml):
    write_yaml("a.yaml", "val1: A\n")
    write_yaml("b.yaml", "val2: B\n")
    path = write_yaml(
        "main.yaml",
        """
defaults:
  - a
  - _self_
  - b
val_main: MAIN
""",
    )
    cfg = load_config_with_defaults(str(path))
    assert cfg.val1 == "A"
    assert cfg.val2 == "B"
    assert cfg.val_main == "MAIN"


def test_config_ignore_none_entry(write_yaml):
    path = write_yaml(
        "main.yaml",
        """
defaults:
  - {"opt": null}
foo: bar
""",
    )
    cfg = load_config_with_defaults(str(path))
    assert cfg.foo == "bar"


def test_defaults_removed_after_merge(write_yaml):
    path = write_yaml(
        "main.yaml",
        """
defaults:
  - _self_
foo: bar
""",
    )
    cfg = load_config_with_defaults(str(path))
    assert "defaults" not in cfg


def test_missing_file_raises(tmp_path):
    """Test that loading a config with a missing file in `defaults`

    raises a FileNotFoundError.
    """
    config_path = tmp_path / "main.yaml"
    config_path.write_text(
        """
defaults:
  - nonexistent_config
"""
    )

    with pytest.raises(FileNotFoundError):
        load_config_with_defaults(str(config_path))


def test_invalid_yaml_raises(tmp_path):
    """Test that loading a config that points to a YAML file

    with syntax errors raises an OmegaConf parsing exception.
    """
    bad_yaml = tmp_path / "bad.yaml"
    # Intentionally malformed YAML (unclosed list)
    bad_yaml.write_text("foo: [unclosed_list\n")

    main_path = tmp_path / "main.yaml"
    main_path.write_text(
        """
defaults:
  - bad
"""
    )

    with pytest.raises(ParserError):
        load_config_with_defaults(str(main_path))


def test_config_does_not_infer_dataset_target(tmp_path):
    recipe_root = tmp_path / "recipe"
    conf_dir = recipe_root / "conf"
    src_dir = recipe_root / "src"
    conf_dir.mkdir(parents=True)
    src_dir.mkdir()
    (src_dir / "dataset.py").write_text(
        """
class MyDataset:
    pass
"""
    )

    config_path = conf_dir / "config.yaml"
    config_path.write_text(
        """
defaults:
  - _self_
dataset:
  train:
    - name: train
      dataset:
        split: train
"""
    )

    cfg = load_config_with_defaults(str(config_path))
    assert "_target_" not in cfg.dataset
    assert "dataset_module" not in cfg.dataset
    assert "_target_" not in cfg.dataset.train[0].dataset
