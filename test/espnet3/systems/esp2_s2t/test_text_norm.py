"""Tests for the normalizer resolver in espnet3/systems/esp2_s2t/text_norm.py.

The CHiME-8 normalizer it can return is an optional dependency, so the tests
that need it skip when it is absent. Everything else runs anywhere.
"""

from pathlib import Path

import pytest

import espnet3.systems.esp2_s2t.text_norm as tn  # noqa: E402


def test_none_means_no_normalizer_at_all():
    """None rather than an identity function, so a caller can skip the call."""
    assert tn.get_text_norm(None) is None
    assert tn.get_text_norm("none") is None


def test_an_unknown_name_is_rejected_by_name():
    with pytest.raises(ValueError, match="Unknown text_norm"):
        tn.get_text_norm("whisper_en")


def test_the_missing_dependency_is_named_with_its_install_command():
    """chime-utils is not on PyPI, so pip's own error would not help.

    It cannot be an espnet extra either -- pip rejects a URL dependency in a
    published package's metadata -- so the message has to name the installer
    that does have the URL.
    """
    try:
        import chime_utils  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="chime-utils"):
            tn.get_text_norm("chime8_keep_fillers")
        with pytest.raises(ImportError, match="make chime_utils.done"):
            tn.get_text_norm("chime8_keep_fillers")
    else:
        pytest.skip("chime-utils is installed, so the error path cannot run")


def test_the_named_installer_exists_and_carries_the_repository():
    """The message points at a real target, and that target installs the thing."""
    repo = Path(__file__).resolve().parents[4]
    installer = repo / "tools" / "installers" / "install_chime_utils.sh"
    assert installer.is_file()
    assert "github.com/chimechallenge/chime-utils" in installer.read_text()
    makefile = (repo / "tools" / "Makefile").read_text()
    assert "chime_utils.done:" in makefile
    assert "install_chime_utils.sh" in makefile


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
