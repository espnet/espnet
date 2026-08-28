import builtins
import sys

import pytest

from espnet2.utils.kaldiio_utils import KALDIIO_INSTALL_MESSAGE, import_kaldiio


def test_import_kaldiio_returns_module():
    kaldiio = pytest.importorskip("kaldiio")
    assert import_kaldiio() is kaldiio


def test_import_kaldiio_error_message(monkeypatch):
    monkeypatch.delitem(sys.modules, "kaldiio", raising=False)
    real_import = builtins.__import__

    def _no_kaldiio(name, *args, **kwargs):
        if name == "kaldiio":
            raise ImportError("No module named 'kaldiio'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_kaldiio)
    with pytest.raises(ImportError, match="espnet\\[kaldi\\]"):
        import_kaldiio()
    assert "optional dependency" in KALDIIO_INSTALL_MESSAGE
