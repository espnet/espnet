import pytest
from pathlib import Path
from omegaconf import OmegaConf
from espnet3.utils.config import load_line, load_config_with_defaults  
from omegaconf.errors import OmegaConfBaseException


# test for load_line
# | Test Case Name                    | Description                                                     | Notes                                | # noqa: E501
# | --------------------------------- | --------------------------------------------------------------- | ------------------------------------ | # noqa: E501
# | `test_load_line_basic`            | Reads a file with multiple lines like `"a\\nb\\nc\\n"`          | Basic multi-line case                | # noqa: E501
# | `test_load_line_with_whitespace`  | Trims leading/trailing spaces from lines like `"  x  \\n y\\n"` | Ensures `.strip()` works as expected | # noqa: E501
# | `test_load_line_empty_file`       | Reads an empty file (`""`)                                      | Should return an empty list `[]`     | # noqa: E501
# | `test_load_line_single_line`      | Reads a file with no newline, e.g., `"hello"`                   | Should return `["hello"]`            | # noqa: E501
# | `test_load_line_trailing_newline` | Reads a file ending with a newline, e.g., `"last\\n"`           | Should return `["last"]`             | # noqa: E501

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


# test for load_config_with_defaults
# Simple Cases
# | Test Case Name                      | Description                                                   | Notes                                         | # noqa: E501
# | ----------------------------------- | ------------------------------------------------------------- | --------------------------------------------- | # noqa: E501
# | `test_config_without_defaults`      | Config does not contain a `defaults` key                      | Returns the config as-is                      | # noqa: E501
# | `test_config_with_self_only`        | `defaults` contains only `_self_`                             | Merges the config with itself                 | # noqa: E501
# | `test_config_with_one_include`      | Includes another config file with `defaults: ["other"]`       | Basic one-level inclusion                     | # noqa: E501
# | `test_config_with_key_value`        | Includes config using key/value format like `{"opt": "adam"}` | Loads from `opt/adam.yaml`                    | # noqa: E501
# | `test_config_ignore_none_entry`     | Skips an entry with `{"something": null}`                     | Null entries in `defaults` are safely ignored | # noqa: E501
# | `test_defaults_removed_after_merge` | Ensures final config does not contain a `defaults` key        | Important for runtime compatibility           | # noqa: E501

# Complex / Recursive Cases
# | Test Case Name                     | Description                                                       | Notes                                           | # noqa: E501
# | ---------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------- | # noqa: E501
# | `test_config_with_nested_defaults` | A config includes another, which itself contains `defaults`       | Recursively merges all involved configs         | # noqa: E501
# | `test_config_with_self_in_middle`  | `_self_` appears in between other includes in the `defaults` list | Validates correct merge order: `A -> self -> B` | # noqa: E501

# Error Cases
# | Test Case Name             | Description                                                           | Expected Exception       | # noqa: E501
# | -------------------------- | --------------------------------------------------------------------- | ------------------------ | # noqa: E501
# | `test_missing_file_raises` | Includes a missing file in `defaults`                                 | `FileNotFoundError`      | # noqa: E501
# | `test_invalid_yaml_raises` | Refers to a YAML file with malformed syntax (e.g., unclosed brackets) | `OmegaConfBaseException` | # noqa: E501



@pytest.fixture
def write_yaml(tmp_path):
    """
    Fixture to create a YAML file under tmp_path and return its path.
    Usage: path = write_yaml("filename.yaml", content_as_str)
    """
    def _write(filename: str, content: str):
        path = tmp_path / filename
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
    base = write_yaml("base.yaml", "a: 1\n")
    main = write_yaml("main.yaml", """
defaults:
  - base
b: 2
""")
    cfg = load_config_with_defaults(str(main))
    assert cfg.a == 1
    assert cfg.b == 2
    assert "defaults" not in cfg


def test_config_with_key_value(write_yaml):
    write_yaml("optimizer/adam.yaml", "lr: 0.001\n")
    path = write_yaml("config.yaml", """
defaults:
  - optimizer: adam
weight_decay: 0.01
""")
    cfg = load_config_with_defaults(str(path))
    assert cfg.optimizer.lr == 0.001
    assert cfg.weight_decay == 0.01


def test_config_with_nested_defaults(write_yaml):
    write_yaml("base.yaml", """
defaults:
  - _self_
foo: 42
""")
    write_yaml("model.yaml", """
defaults:
  - base
bar: 99
""")
    path = write_yaml("main.yaml", """
defaults:
  - model
baz: 7
""")
    cfg = load_config_with_defaults(str(path))
    assert cfg.foo == 42
    assert cfg.bar == 99
    assert cfg.baz == 7


def test_config_with_self_in_middle(write_yaml):
    write_yaml("a.yaml", "val1: A\n")
    write_yaml("b.yaml", "val2: B\n")
    path = write_yaml("main.yaml", """
defaults:
  - a
  - _self_
  - b
val_main: MAIN
""")
    cfg = load_config_with_defaults(str(path))
    assert cfg.val1 == "A"
    assert cfg.val2 == "B"
    assert cfg.val_main == "MAIN"


def test_config_ignore_none_entry(write_yaml):
    path = write_yaml("main.yaml", """
defaults:
  - {"opt": null}
foo: bar
""")
    cfg = load_config_with_defaults(str(path))
    assert cfg.foo == "bar"


def test_defaults_removed_after_merge(write_yaml):
    path = write_yaml("main.yaml", """
defaults:
  - _self_
foo: bar
""")
    cfg = load_config_with_defaults(str(path))
    assert "defaults" not in cfg


def test_missing_file_raises(tmp_path):
    """
    Test that loading a config with a missing file in `defaults`
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
    """
    Test that loading a config that points to a YAML file
    with syntax errors raises an OmegaConf parsing exception.
    """
    bad_yaml = tmp_path / "bad.yaml"
    # Intentionally malformed YAML (unclosed list)
    bad_yaml.write_text("foo: [unclosed_list\n")

    main_path = tmp_path / "main.yaml"
    main_path.write_text(f"""
defaults:
  - bad
""")

    with pytest.raises(OmegaConfBaseException):
        load_config_with_defaults(str(main_path))