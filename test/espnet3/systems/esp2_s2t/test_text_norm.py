"""Tests for the normalizer resolver in espnet3/systems/esp2_s2t/src/sot/text_norm.py.

The CHiME-8 normalizer it can return is an optional dependency, so the tests
that need it skip when it is absent. The other tests run in any environment.
"""

import pytest

import espnet3.systems.esp2_s2t.src.sot.text_norm as tn  # noqa: E402


def test_none_means_no_normalizer_at_all():
    """get_text_norm returns None for None and for "none".

    It does not return an identity function, so a caller can skip the call.
    """
    assert tn.get_text_norm(None) is None
    assert tn.get_text_norm("none") is None


def test_an_unknown_name_is_rejected_by_name():
    with pytest.raises(ValueError, match="Unknown text_norm"):
        tn.get_text_norm("whisper_en")


def test_the_missing_dependency_is_named_with_its_install_command():
    """The error message gives the pip command that installs chime-utils.

    chime-utils is not on PyPI, and PyPI does not accept a package that lists
    a URL dependency, so espnet cannot install it through an extra. The
    message therefore gives the pip command with the repository URL.
    """
    try:
        import chime_utils  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="chime-utils"):
            tn.get_text_norm("chime8_keep_fillers")
        with pytest.raises(ImportError, match="github.com/chimechallenge/chime-utils"):
            tn.get_text_norm("chime8_keep_fillers")
    else:
        pytest.skip("chime-utils is installed, so the error path cannot run")


def test_the_chime8_normalizer_keeps_fillers_and_expands_contractions():
    """chime8_keep_fillers keeps fillers and expands contractions.

    The published checkpoint's targets were written with this normalizer. As its
    name says, it keeps fillers. chime_utils' own get_txt_norm("chime8") takes no
    arguments and deletes fillers, which would remove most of AMI's backchannels
    from the targets.
    """
    pytest.importorskip("chime_utils")
    norm = tn.get_text_norm("chime8_keep_fillers")
    assert norm("yeah well i don't know if it's usable") == (
        "yeah well i do not know if it is usable"
    )
    assert norm("um apparently") == "hmm apparently"
