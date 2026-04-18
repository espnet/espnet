"""Pytest configuration for espnet2/speechlm/bin tests.

``prepare_length_stats.py`` transitively imports ``espnet2.speechlm.model`` →
``speechlm_job`` → ``multimodal_io`` (needs transformers, joblib, humanfriendly)
and ``espnet2.speechlm.dataloader.multimodal_loader`` (needs arkive, duckdb,
pyarrow, lhotse, soundfile, kaldiio). This conftest stubs all of those so the
test module imports without the real packages.
"""

import importlib.machinery
import os
import sys
import types

import torch.nn as nn


def _install_stub(name: str, module: types.ModuleType) -> None:
    sys.modules.setdefault(name, module)


def pytest_configure():
    """Inject stubs before any test module is collected."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    # ---- arkive stubs ----
    if "arkive" not in sys.modules:
        arkive = types.ModuleType("arkive")
        arkive.__spec__ = importlib.machinery.ModuleSpec("arkive", loader=None)
        arkive.__path__ = []

        arkive_text = types.ModuleType("arkive.text")
        arkive_text.__spec__ = importlib.machinery.ModuleSpec(
            "arkive.text", loader=None
        )
        arkive_text.__path__ = []

        arkive_text_write_utils = types.ModuleType("arkive.text.write_utils")
        arkive_text_write_utils.__spec__ = importlib.machinery.ModuleSpec(
            "arkive.text.write_utils", loader=None
        )

        def _decompress_text_data(data_bytes):
            raise NotImplementedError("stub: arkive not installed")

        arkive_text_write_utils._decompress_text_data = _decompress_text_data

        def _audio_read(*args, **kwargs):
            raise NotImplementedError("stub: arkive not installed")

        arkive.audio_read = _audio_read
        _install_stub("arkive", arkive)
        _install_stub("arkive.text", arkive_text)
        _install_stub("arkive.text.write_utils", arkive_text_write_utils)

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

        duckdb.connect = lambda *args, **kwargs: DummyDuckDBConnection()
        _install_stub("duckdb", duckdb)

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

        def _sf_read(*args, **kwargs):
            raise NotImplementedError("stub: soundfile not installed")

        sf.read = _sf_read
        _install_stub("soundfile", sf)

    # ---- kaldiio stub ----
    if "kaldiio" not in sys.modules:
        kaldiio = types.ModuleType("kaldiio")
        kaldiio.__spec__ = importlib.machinery.ModuleSpec("kaldiio", loader=None)

        def load_mat(*args, **kwargs):
            raise NotImplementedError("stub: kaldiio not installed")

        kaldiio.load_mat = load_mat
        _install_stub("kaldiio", kaldiio)

    # ---- transformers stub ----
    if "transformers" not in sys.modules:
        transformers = types.ModuleType("transformers")
        transformers.__spec__ = importlib.machinery.ModuleSpec(
            "transformers", loader=None
        )

        class _MockConfig:
            architectures = ["MockModel"]
            vocab_size = 100
            hidden_size = 64

        class AutoConfig:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return _MockConfig()

        transformers.AutoConfig = AutoConfig

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return AutoTokenizer()

            def __call__(self, text, **kwargs):
                return {"input_ids": [0]}

        transformers.AutoTokenizer = AutoTokenizer

        cache_utils = types.ModuleType("transformers.cache_utils")
        cache_utils.__spec__ = importlib.machinery.ModuleSpec(
            "transformers.cache_utils", loader=None
        )

        class DynamicCache:
            def __init__(self):
                self.layers = []

        cache_utils.DynamicCache = DynamicCache
        transformers.cache_utils = cache_utils

        class _MockInnerModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        class MockModel(nn.Module):
            config_class = _MockConfig

            def __init__(self, config=None):
                super().__init__()
                if config is None:
                    config = _MockConfig()
                self.config = config
                self.model = _MockInnerModel(config)
                self.lm_head = nn.Linear(
                    config.hidden_size, config.vocab_size, bias=False
                )

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls(_MockConfig())

        transformers.MockModel = MockModel

        _install_stub("transformers", transformers)
        _install_stub("transformers.cache_utils", cache_utils)

    # ---- joblib stub ----
    if "joblib" not in sys.modules:
        joblib = types.ModuleType("joblib")
        joblib.__spec__ = importlib.machinery.ModuleSpec("joblib", loader=None)

        class _MockKMeans:
            def predict(self, x):
                import numpy as np

                return np.zeros(len(x), dtype=int)

        joblib.load = lambda *args, **kwargs: _MockKMeans()
        _install_stub("joblib", joblib)

    # ---- humanfriendly stub ----
    if "humanfriendly" not in sys.modules:
        humanfriendly = types.ModuleType("humanfriendly")
        humanfriendly.__spec__ = importlib.machinery.ModuleSpec(
            "humanfriendly", loader=None
        )

        humanfriendly.format_size = lambda num_bytes, **kwargs: f"{num_bytes} bytes"
        _install_stub("humanfriendly", humanfriendly)
