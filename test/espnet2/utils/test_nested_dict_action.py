import argparse
from argparse import Namespace

import pytest

from espnet2.utils.nested_dict_action import NestedDictAction


def test_NestedDictAction():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", action=NestedDictAction, default=3)

    assert parser.parse_args(["--conf", "a=3", "--conf", "c=4"]) == Namespace(
        conf={"a": 3, "c": 4}
    )
    assert parser.parse_args(["--conf", "c.d=4"]) == Namespace(conf={"c": {"d": 4}})
    assert parser.parse_args(["--conf", "c.d=4", "--conf", "c=2"]) == Namespace(
        conf={"c": 2}
    )
    assert parser.parse_args(["--conf", "{d: 5, e: 9}"]) == Namespace(
        conf={"d": 5, "e": 9}
    )
    assert parser.parse_args(["--conf", '{"d": 5, "e": 9}']) == Namespace(
        conf={"d": 5, "e": 9}
    )
    assert parser.parse_args(
        ["--conf", '{"d": 5, "e": 9}', "--conf", "d.e=3"]
    ) == Namespace(conf={"d": {"e": 3}, "e": 9})


def test_NestedDictAction_exception():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", action=NestedDictAction, default={"a": 4})
    with pytest.raises(SystemExit):
        parser.parse_args(["--aa", "{d: 5, e: 9}"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--conf", "aaa"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--conf", "[0, 1, 2]"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--conf", "[cd, e, aaa]"])
