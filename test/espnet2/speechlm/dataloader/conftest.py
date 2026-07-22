"""Pytest configuration with lightweight stubs for optional heavy dependencies.

Modeled after test/espnet3/conftest.py. Injects stubs for omniio, duckdb,
pyarrow, lhotse, and soundfile so that TextReader, DialogueReader,
SingleDataset, etc. can be imported without the real packages.
"""

import importlib.machinery
import os
import sys
import types


def _install_stub(name: str, module: types.ModuleType) -> None:
    sys.modules.setdefault(name, module)


def pytest_configure():
    """Inject lightweight stubs for optional heavy dependencies."""
    # Force CPU-only tests
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    # ---- omniio stubs ----
    # Mirror the real omniio layout used by the loaders:
    #   audio_loader -> ``from omniio.interface import audio_read``
    #   text_loader  -> ``from omniio.text.read import text_read_local``
    if "omniio" not in sys.modules:
        omniio = types.ModuleType("omniio")
        omniio.__spec__ = importlib.machinery.ModuleSpec("omniio", loader=None)
        omniio.__path__ = []
        _install_stub("omniio", omniio)

        # omniio.interface.audio_read used by audio_loader
        omniio_interface = types.ModuleType("omniio.interface")
        omniio_interface.__spec__ = importlib.machinery.ModuleSpec(
            "omniio.interface", loader=None
        )

        def _audio_read(*args, **kwargs):
            raise NotImplementedError("stub: omniio not installed")

        omniio_interface.audio_read = _audio_read
        omniio.interface = omniio_interface
        _install_stub("omniio.interface", omniio_interface)

        # omniio.text.read.text_read_local used by text_loader
        omniio_text = types.ModuleType("omniio.text")
        omniio_text.__spec__ = importlib.machinery.ModuleSpec(
            "omniio.text", loader=None
        )
        omniio_text.__path__ = []
        _install_stub("omniio.text", omniio_text)

        omniio_text_read = types.ModuleType("omniio.text.read")
        omniio_text_read.__spec__ = importlib.machinery.ModuleSpec(
            "omniio.text.read", loader=None
        )

        def _text_read_local(*args, **kwargs):
            raise NotImplementedError("stub: omniio not installed")

        omniio_text_read.text_read_local = _text_read_local
        omniio_text.read = omniio_text_read
        _install_stub("omniio.text.read", omniio_text_read)

    # ---- duckdb stub ----
    if "duckdb" not in sys.modules:
        duckdb = types.ModuleType("duckdb")
        duckdb.__spec__ = importlib.machinery.ModuleSpec("duckdb", loader=None)

        class DummyDuckDBConnection:
            def execute(self, query):
                return self

            def register(self, name, table):
                pass

            def pl(self):
                raise NotImplementedError("stub: duckdb not installed")

            def close(self):
                pass

        def connect(*args, **kwargs):
            return DummyDuckDBConnection()

        duckdb.connect = connect
        _install_stub("duckdb", duckdb)

    # ---- pyarrow stub ----
    if "pyarrow" not in sys.modules:
        pa = types.ModuleType("pyarrow")
        pa.__spec__ = importlib.machinery.ModuleSpec("pyarrow", loader=None)

        def table(data):
            return data

        def array(data):
            return data

        pa.table = table
        pa.array = array
        _install_stub("pyarrow", pa)

    # ---- lhotse stub ----
    if "lhotse" not in sys.modules:
        lhotse = types.ModuleType("lhotse")
        lhotse.__spec__ = importlib.machinery.ModuleSpec("lhotse", loader=None)

        class DummyCutSet:
            @classmethod
            def from_file(cls, path):
                raise NotImplementedError("stub: lhotse not installed")

            @classmethod
            def from_cuts(cls, cuts):
                raise NotImplementedError("stub: lhotse not installed")

        class DummyRecordingSet:
            @classmethod
            def from_file(cls, path):
                raise NotImplementedError("stub: lhotse not installed")

            @classmethod
            def from_recordings(cls, recordings):
                raise NotImplementedError("stub: lhotse not installed")

        lhotse.CutSet = DummyCutSet
        lhotse.RecordingSet = DummyRecordingSet
        _install_stub("lhotse", lhotse)

    # ---- soundfile stub ----
    if "soundfile" not in sys.modules:
        sf = types.ModuleType("soundfile")
        sf.__spec__ = importlib.machinery.ModuleSpec("soundfile", loader=None)

        def read(*args, **kwargs):
            raise NotImplementedError("stub: soundfile not installed")

        sf.read = read
        _install_stub("soundfile", sf)

    # ---- kaldiio stub ----
    if "kaldiio" not in sys.modules:
        kaldiio = types.ModuleType("kaldiio")
        kaldiio.__spec__ = importlib.machinery.ModuleSpec("kaldiio", loader=None)

        def load_mat(*args, **kwargs):
            raise NotImplementedError("stub: kaldiio not installed")

        kaldiio.load_mat = load_mat
        _install_stub("kaldiio", kaldiio)
