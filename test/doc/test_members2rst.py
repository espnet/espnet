"""Regressions for public API documentation discovery."""

import ast
import importlib.util
from pathlib import Path

# doc/ is not a package; importing it as one works only when the repository
# root happens to be on sys.path, which plain `pytest` does not arrange.
_SCRIPT = Path(__file__).resolve().parents[2] / "doc" / "members2rst.py"
_spec = importlib.util.spec_from_file_location("members2rst", _SCRIPT)
_members2rst = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_members2rst)
top_level_classes = _members2rst.top_level_classes
top_level_functions = _members2rst.top_level_functions


def test_private_members_do_not_create_colliding_pages():
    """Private classes follow the same exclusion rule as private functions."""
    body = ast.parse(
        "class _Hypothesis: pass\n"
        "class Hypothesis(_Hypothesis): pass\n"
        "def _helper(): pass\n"
        "def helper(): pass\n"
    ).body
    assert [node.name for node in top_level_classes(body)] == ["Hypothesis"]
    assert [node.name for node in top_level_functions(body)] == ["helper"]


def test_beam_search_keeps_public_hypothesis_documentation():
    """The real tuple implementation must not generate a second Hypothesis page."""
    source = Path(__file__).resolve().parents[2] / "espnet2/legacy/nets/beam_search.py"
    names = {
        node.name for node in top_level_classes(ast.parse(source.read_text()).body)
    }
    assert "Hypothesis" in names
    assert "_Hypothesis" not in names
