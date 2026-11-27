"""Check that every espnet2/bin parser has a description.

doc/argparse2rst.py lists each espnet2/bin module with its
get_parser().description and asserts that the description is not None.
The documentation build runs on master pushes and on pull requests that carry
the Documentation label, so this test catches a missing description in the
ordinary pull request CI.
"""

import importlib
from pathlib import Path

import pytest

import espnet2.bin

BIN_DIR = Path(espnet2.bin.__file__).parent
MODULES = sorted(p.stem for p in BIN_DIR.glob("*.py") if p.stem != "__init__")


@pytest.mark.parametrize("name", MODULES)
def test_get_parser_has_description(name):
    try:
        module = importlib.import_module(f"espnet2.bin.{name}")
    except ModuleNotFoundError as e:
        # Optional dependencies such as k2 or mir_eval; argparse2rst skips
        # these modules in the same way.
        if e.name is not None and e.name.split(".")[0] == "espnet2":
            raise
        pytest.skip(f"optional dependency {e.name} is not installed")
    if not hasattr(module, "get_parser"):
        pytest.skip(f"espnet2.bin.{name} has no get_parser()")

    description = module.get_parser().description
    assert description is not None and description.strip(), (
        f"espnet2.bin.{name}.get_parser() has no description; "
        "doc/argparse2rst.py requires one"
    )
