import sys

import pytest

from espnet2.utils.kaldiio_utils import KALDIIO_INSTALL_MESSAGE, import_kaldiio


class _BlockKaldiio:
    """Meta path finder that makes ``kaldiio`` look uninstalled."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "kaldiio" or fullname.startswith("kaldiio."):
            raise ImportError("No module named 'kaldiio'")
        return None


def test_import_kaldiio_returns_module():
    kaldiio = pytest.importorskip("kaldiio")
    assert import_kaldiio() is kaldiio


def test_import_kaldiio_error_message(monkeypatch):
    for name in [n for n in sys.modules if n == "kaldiio" or n.startswith("kaldiio.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [_BlockKaldiio()] + sys.meta_path)

    with pytest.raises(ImportError, match=r"espnet\[kaldiio\]"):
        import_kaldiio()
    assert "optional dependency" in KALDIIO_INSTALL_MESSAGE
