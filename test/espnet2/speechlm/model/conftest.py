"""Dependency stubs for espnet2/speechlm/model tests.

Injects lightweight stubs for heavy optional dependencies (transformers,
joblib, humanfriendly) so that tests can run in CI without those packages.
"""

import importlib.machinery
import os
import sys
import types

import torch.nn as nn


def _install_stub(name, module):
    sys.modules.setdefault(name, module)


def pytest_configure():
    """Inject stubs before any test module is collected."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    # ---- transformers stub ----
    if "transformers" not in sys.modules:
        transformers = types.ModuleType("transformers")
        transformers.__spec__ = importlib.machinery.ModuleSpec(
            "transformers", loader=None
        )

        # AutoConfig
        class _MockConfig:
            architectures = ["MockModel"]
            vocab_size = 100
            hidden_size = 64

        class AutoConfig:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return _MockConfig()

        transformers.AutoConfig = AutoConfig

        # AutoTokenizer
        class AutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return AutoTokenizer()

            def __call__(self, text, **kwargs):
                return {"input_ids": [0]}

        transformers.AutoTokenizer = AutoTokenizer

        # cache_utils
        cache_utils = types.ModuleType("transformers.cache_utils")
        cache_utils.__spec__ = importlib.machinery.ModuleSpec(
            "transformers.cache_utils", loader=None
        )

        class DynamicCache:
            def __init__(self):
                self.layers = []

        cache_utils.DynamicCache = DynamicCache
        transformers.cache_utils = cache_utils

        # MockModel (minimal HF-style nn.Module)
        class _MockInnerModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

            def forward(self, inputs_embeds=None, position_ids=None, **kwargs):
                class _Out:
                    pass

                out = _Out()
                out.last_hidden_state = inputs_embeds
                out.past_key_values = None

                def get(key, default=None):
                    return default

                out.get = get
                return out

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
                config = _MockConfig()
                return cls(config)

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

        def load(*args, **kwargs):
            return _MockKMeans()

        joblib.load = load
        _install_stub("joblib", joblib)

    # ---- humanfriendly stub ----
    if "humanfriendly" not in sys.modules:
        humanfriendly = types.ModuleType("humanfriendly")
        humanfriendly.__spec__ = importlib.machinery.ModuleSpec(
            "humanfriendly", loader=None
        )

        def format_size(num_bytes, **kwargs):
            return f"{num_bytes} bytes"

        humanfriendly.format_size = format_size
        _install_stub("humanfriendly", humanfriendly)
