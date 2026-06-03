import os
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from omegaconf import OmegaConf
from yaml.parser import ParserError

from espnet3.utils.config_utils import (
    _build_config_path,
    _ensure_target_convert_all,
    _infer_default_package_from_config_path,
    _rewrite_relative_resolver_paths,
    load_and_merge_config,
    load_config_with_defaults,
    load_default_config,
    load_line,
)

# ===============================================================
# Test Case Summary for Config Utilities
# ===============================================================
#
# Tests for `load_line(path)`
# | Test Name                         | Description                                                    | # noqa: E501
# |----------------------------------|----------------------------------------------------------------| # noqa: E501
# | test_load_line_basic             | Reads a file with multiple lines like `"a\\nb\\nc"`            | # noqa: E501
# | test_load_line_with_whitespace   | Strips leading/trailing whitespace from each line              | # noqa: E501
# | test_load_line_empty_file        | Returns an empty list for an empty file                        | # noqa: E501
# | test_load_line_single_line       | Handles file with only one line, no newline                    | # noqa: E501
# | test_load_line_trailing_newline  | Handles file ending with a newline character                   | # noqa: E501
#
# Simple Tests for `load_config_with_defaults`
# | Test Name                          | Description                                                    | # noqa: E501
# |-----------------------------------|----------------------------------------------------------------| # noqa: E501
# | test_config_without_defaults       | Config with no `defaults` key, should load as-is               | # noqa: E501
# | test_config_with_self_only         | `defaults: [_self_]` merges config with itself                 | # noqa: E501
# | test_config_with_one_include       | Includes one external config file via `defaults`              | # noqa: E501
# | test_config_with_key_value         | Loads config with `defaults: [{opt: adam}]` → opt/adam.yaml    | # noqa: E501
# | test_config_update_with_key_value  | Adds extra fields to a nested dict loaded from `defaults`      | # noqa: E501
# | test_config_ignore_none_entry      | Skips null entries in the `defaults` list                      | # noqa: E501
# | test_defaults_removed_after_merge  | Ensures `defaults` is not present in the final config          | # noqa: E501
#
# Complex/Recursive Merging
# | Test Name                          | Description                                                    | # noqa: E501
# |-----------------------------------|----------------------------------------------------------------| # noqa: E501
# | test_config_with_nested_defaults   | Nested includes (config includes another config that has its own defaults) | # noqa: E501
# | test_config_with_self_in_middle    | `_self_` appears in the middle of the `defaults` list          | # noqa: E501
#
# Error Cases
# | Test Name                          | Description                                                    | Expected Exception       | # noqa: E501
# |-----------------------------------|----------------------------------------------------------------|--------------------------| # noqa: E501
# | test_missing_file_raises           | Missing file in `defaults`                                     | FileNotFoundError        | # noqa: E501
# | test_invalid_yaml_raises           | YAML parse error in included file                              | ParserError              | # noqa: E501


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


def test_load_line_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_line(tmp_path / "missing.txt")


def test_load_line_permission_error_raises():
    with patch("builtins.open", mock_open()) as mocked_open:
        mocked_open.side_effect = PermissionError("no access")
        with pytest.raises(PermissionError, match="no access"):
            load_line("forbidden.txt")


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


def test_self_name_resolver(write_yaml):
    path = write_yaml("main.yaml", "name: ${self_name:}\n")
    cfg = load_config_with_defaults(str(path))
    assert cfg.name == "main"


def test_self_name_resolver_in_defaults(write_yaml):
    write_yaml("base.yaml", "base_name: ${self_name:}\n")
    path = write_yaml("main.yaml", "defaults:\n  - base\nname: ${self_name:}\n")
    cfg = load_config_with_defaults(str(path))
    assert cfg.base_name == "base"
    assert cfg.name == "main"


def test_missing_file_raises(tmp_path):
    """Test that loading a config with a missing file in `defaults`

    raises a FileNotFoundError.
    """
    config_path = tmp_path / "main.yaml"
    config_path.write_text("""
defaults:
  - nonexistent_config
""")

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
    main_path.write_text("""
defaults:
  - bad
""")

    with pytest.raises(ParserError):
        load_config_with_defaults(str(main_path))


def test_load_default_config_train():
    cfg = load_default_config("training.yaml", "egs3.TEMPLATE.asr")
    assert "dataset" in cfg
    assert "exp_dir" in cfg


def test_load_and_merge_config_none():
    assert load_and_merge_config(None, "metrics.yaml") is None


def test_load_and_merge_config_requires_inferable_default_package(tmp_path):
    config_path = tmp_path / "training.yaml"
    config_path.write_text("exp_dir: ./exp/test\n")

    with pytest.raises(
        ValueError,
        match="default_package is required when it cannot be inferred",
    ):
        load_and_merge_config(config_path, "training.yaml")


def test_load_and_merge_config_resolve_false_preserves_interpolations(
    write_yaml, monkeypatch
):
    template = write_yaml(
        "template.yaml",
        """
exp_dir: ./exp/from_template
custom_dir: ${exp_dir}/custom
""",
    )
    user = write_yaml(
        "user.yaml",
        """
custom_path: ${exp_dir}/artifacts
""",
    )

    monkeypatch.setattr(
        "espnet3.utils.config_utils._load_default_config",
        lambda _, __, bind_self_name=True: (
            load_config_with_defaults(str(template), resolve=False)
            if bind_self_name
            else OmegaConf.load(str(template))
        ),
    )

    cfg = load_and_merge_config(
        user,
        "training.yaml",
        default_package="dummy.package",
        resolve=False,
    )

    unresolved = OmegaConf.to_container(cfg, resolve=False)
    assert unresolved["custom_dir"] == "${exp_dir}/custom"
    assert unresolved["custom_path"] == "${exp_dir}/artifacts"


def test_load_and_merge_config_resolves_user_reference_to_template_value(
    write_yaml, monkeypatch
):
    template = write_yaml(
        "template.yaml",
        """
exp_dir: ./exp/from_template
inference_dir: ${exp_dir}/inference
""",
    )
    user = write_yaml(
        "user.yaml",
        """
custom_dir: ${exp_dir}/custom
""",
    )

    monkeypatch.setattr(
        "espnet3.utils.config_utils._load_default_config",
        lambda _, __, bind_self_name=True: (
            load_config_with_defaults(str(template), resolve=False)
            if bind_self_name
            else OmegaConf.load(str(template))
        ),
    )

    cfg = load_and_merge_config(
        user,
        "training.yaml",
        default_package="dummy.package",
    )

    assert cfg.exp_dir == "./exp/from_template"
    assert cfg.inference_dir == "./exp/from_template/inference"
    assert cfg.custom_dir == "./exp/from_template/custom"


def test_load_and_merge_config_resolves_template_reference_to_user_override(
    write_yaml, monkeypatch
):
    template = write_yaml(
        "template.yaml",
        """
exp_dir: ./exp/from_template
custom_dir: ${exp_dir}/custom
""",
    )
    user = write_yaml(
        "user.yaml",
        """
exp_dir: ./exp/from_user
""",
    )

    monkeypatch.setattr(
        "espnet3.utils.config_utils._load_default_config",
        lambda _, __, bind_self_name=True: (
            load_config_with_defaults(str(template), resolve=False)
            if bind_self_name
            else OmegaConf.load(str(template))
        ),
    )

    cfg = load_and_merge_config(
        user,
        "training.yaml",
        default_package="dummy.package",
    )

    assert cfg.exp_dir == "./exp/from_user"
    assert cfg.custom_dir == "./exp/from_user/custom"


def test_load_and_merge_config_binds_self_name_to_user_config(write_yaml, monkeypatch):
    template = write_yaml(
        "template.yaml",
        """
exp_name: ${self_name:}
""",
    )
    user = write_yaml("user.yaml", "{}\n")

    monkeypatch.setattr(
        "espnet3.utils.config_utils._load_default_config",
        lambda _, __, bind_self_name=True: (
            load_config_with_defaults(str(template), resolve=False)
            if bind_self_name
            else OmegaConf.load(str(template))
        ),
    )

    cfg = load_and_merge_config(
        user,
        "training.yaml",
        default_package="dummy.package",
    )

    assert cfg.exp_name == "user"


def test_load_and_merge_config_resolve_false_binds_self_name_to_user_config(
    write_yaml, monkeypatch
):
    template = write_yaml(
        "template.yaml",
        """
exp_name: ${self_name:}
custom_dir: ${exp_dir}/custom
""",
    )
    user = write_yaml(
        "user.yaml",
        """
exp_dir: ./exp/from_user
""",
    )

    monkeypatch.setattr(
        "espnet3.utils.config_utils._load_default_config",
        lambda _, __, bind_self_name=True: (
            load_config_with_defaults(str(template), resolve=False)
            if bind_self_name
            else OmegaConf.load(str(template))
        ),
    )

    cfg = load_and_merge_config(
        user,
        "training.yaml",
        default_package="dummy.package",
        resolve=False,
    )

    unresolved = OmegaConf.to_container(cfg, resolve=False)
    assert unresolved["exp_name"] == "user"
    assert unresolved["custom_dir"] == "${exp_dir}/custom"


def test_ensure_target_convert_all_nested():
    cfg = OmegaConf.create(
        {
            "a": {"_target_": "package.A", "x": 1},
            "b": [{"_target_": "package.B", "y": 2}, {"z": 3}],
            "c": {"nested": {"_target_": "package.C"}},
        }
    )
    _ensure_target_convert_all(cfg)
    assert cfg.a._convert_ == "all"
    assert cfg.b[0]._convert_ == "all"
    assert cfg.c.nested._convert_ == "all"
    assert "_convert_" not in cfg.b[1]


def test_build_config_path_adds_yaml(tmp_path):
    path = _build_config_path(tmp_path, "train")
    assert path == tmp_path / "train.yaml"


def test_build_config_path_keeps_yaml(tmp_path):
    path = _build_config_path(tmp_path, "train.yaml")
    assert path == tmp_path / "train.yaml"


def test_build_config_path_rejects_suffix(tmp_path):
    with pytest.raises(ValueError):
        _build_config_path(tmp_path, "train.txt")


def test_infer_default_package_returns_none_without_egs3():
    path = Path("/home/user/project/conf/train.yaml")
    assert _infer_default_package_from_config_path(path) is None


def test_infer_default_package_returns_none_when_conf_too_shallow(tmp_path):
    # egs3/conf/train.yaml — conf is only 1 level deep under egs3
    path = tmp_path / "egs3" / "conf" / "train.yaml"
    assert _infer_default_package_from_config_path(path) is None


def test_infer_default_package_infers_valid_task(tmp_path):
    path = tmp_path / "egs3" / "mini_an4" / "asr" / "conf" / "train.yaml"
    assert _infer_default_package_from_config_path(path) == "egs3.TEMPLATE.asr"


def test_rewrite_preserves_absolute_resolver_path(tmp_path):
    value = "${load_line:/absolute/tokens.txt}"
    assert _rewrite_relative_resolver_paths(value, tmp_path) == value


def test_rewrite_preserves_dynamic_resolver_path(tmp_path):
    value = "${load_line:${base_dir}/tokens.txt}"
    assert _rewrite_relative_resolver_paths(value, tmp_path) == value


def test_rewrite_handles_quoted_resolver_path(tmp_path):
    value = "${load_line:'my tokens.txt'}"
    result = _rewrite_relative_resolver_paths(value, tmp_path)
    assert result.startswith("${load_line:'")
    assert result.endswith("'}")
    assert "my tokens.txt" in result
