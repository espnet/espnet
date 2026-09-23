"""Tests for the normalizer resolver in egs3/ami/s2t/dataset/text_norm.py.

The CHiME-8 normalizer it can return is an optional dependency, so the tests
that need it skip when it is absent. Everything else runs anywhere.
"""

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[4]
_RECIPE = _REPO / "egs3" / "ami" / "s2t"


def _load():
    spec = importlib.util.spec_from_file_location(
        "ami_s2t_text_norm", _RECIPE / "dataset" / "text_norm.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tn = _load()


def test_none_means_no_normalizer_at_all():
    """None rather than an identity function, so a caller can skip the call."""
    assert tn.get_text_norm(None) is None
    assert tn.get_text_norm("none") is None


def test_an_unknown_name_is_rejected_by_name():
    with pytest.raises(ValueError, match="Unknown text_norm"):
        tn.get_text_norm("whisper_en")


def test_the_missing_dependency_is_named_with_its_install_command():
    """chime-utils is not on PyPI, so pip's own error would not help."""
    pytest.importorskip  # keep the import list honest
    try:
        import chime_utils  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="chime-utils"):
            tn.get_text_norm("chime8_keep_fillers")
        with pytest.raises(ImportError, match="github.com/chimechallenge"):
            tn.get_text_norm("chime8_keep_fillers")
    else:
        pytest.skip("chime-utils is installed, so the error path cannot run")


def test_the_chime8_normalizer_keeps_fillers_and_expands_contractions():
    """What the published checkpoint's targets were written with.

    Fillers are the point of the name: chime_utils' own
    get_txt_norm("chime8") takes no arguments and deletes them, which would
    strip most of AMI's backchannels out of the targets.
    """
    pytest.importorskip("chime_utils")
    norm = tn.get_text_norm("chime8_keep_fillers")
    assert norm("yeah well i don't know if it's usable") == (
        "yeah well i do not know if it is usable"
    )
    assert norm("um apparently") == "hmm apparently"
