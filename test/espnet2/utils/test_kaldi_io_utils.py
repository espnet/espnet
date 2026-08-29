import sys

import pytest

from espnet2.utils.kaldi_io_utils import KALDI_IO_INSTALL_MESSAGE, import_kaldi_io


class _BlockOmniio:
    """Meta path finder that makes ``omniio`` look uninstalled."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "omniio" or fullname.startswith("omniio."):
            raise ImportError("No module named 'omniio'")
        return None


def test_import_kaldi_io_returns_module():
    omniio_kaldi = pytest.importorskip("omniio.kaldi")
    assert import_kaldi_io() is omniio_kaldi


def test_import_kaldi_io_error_message(monkeypatch):
    for name in [n for n in sys.modules if n == "omniio" or n.startswith("omniio.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [_BlockOmniio()] + sys.meta_path)

    with pytest.raises(ImportError, match="omniio"):
        import_kaldi_io()


def test_install_message_mentions_the_repository():
    assert "github.com/wavlab-speech/omniio" in KALDI_IO_INSTALL_MESSAGE
