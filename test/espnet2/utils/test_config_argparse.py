import pytest
import yaml

from espnet2.utils import config_argparse


@pytest.fixture()
def parser():
    _parser = config_argparse.ArgumentParser("test")
    _parser.add_argument("--foo")
    _parser.add_argument("--bar")
    _parser.add_argument("--baz", action="store_true")
    _parser.add_argument("--count", action="count")
    return _parser


def test_config_argparse(tmpdir, parser):
    config = tmpdir / "a.yaml"
    with config.open("w") as f:
        yaml.safe_dump({"foo": "2", "baz": True, "count": 3}, f)

    args = parser.parse_args(["--config", str(config), "--bar", "4"])
    assert args.foo == "2"
    assert args.bar == "4"
    assert args.baz
    assert args.count == 3


def test_config_argparse_config_not_found(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(["--config", "dummy.yaml", "--bar", "4"])


def test_config_argparse_not_dict(tmpdir, parser):
    config = tmpdir / "a.yaml"
    with config.open("w") as f:
        yaml.safe_dump([1, 2, 3], f)

    with pytest.raises(SystemExit):
        parser.parse_args(["--config", str(config), "--bar", "4"])


def test_config_argparse_invalid_key(tmpdir, parser):
    config = tmpdir / "a.yaml"
    with config.open("w") as f:
        yaml.safe_dump({"foo": "2", "dummy": "aaa"}, f)

    with pytest.raises(SystemExit):
        parser.parse_args(["--config", str(config), "--bar", "4"])
