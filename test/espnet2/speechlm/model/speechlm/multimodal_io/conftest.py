"""Dependency stubs for multimodal_io tests.

Adds a torchaudio stub (only when the real package is unavailable) so that
audio.py can be imported at module level. Test files handle their own mocking
via unittest.mock.patch for everything else (transformers, espnet_model_zoo).

IMPORTANT: Stubs injected via sys.modules in pytest_configure pollute the
entire test session. Only stub packages that (1) are imported at module level
by source code AND (2) are genuinely unavailable. Never stub packages that
other tests in the same session may need.
"""

import importlib
import importlib.machinery
import os
import sys
import types


def pytest_configure():
    """Install stubs and force CPU before any test collection."""

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # ---- torchaudio stub ----
    # audio.py does `import torchaudio` at module level, so we need this
    # stub when the real package is not installed. Only inject if a real
    # import would fail — never shadow an installed package.
    if "torchaudio" not in sys.modules:
        try:
            importlib.import_module("torchaudio")
        except ImportError:
            torchaudio = types.ModuleType("torchaudio")
            torchaudio.__spec__ = importlib.machinery.ModuleSpec(
                "torchaudio", loader=None
            )

            functional = types.ModuleType("torchaudio.functional")
            functional.__spec__ = importlib.machinery.ModuleSpec(
                "torchaudio.functional", loader=None
            )

            def resample(waveform, orig_freq, new_freq):
                return waveform

            functional.resample = resample
            torchaudio.functional = functional

            sys.modules["torchaudio"] = torchaudio
            sys.modules["torchaudio.functional"] = functional
