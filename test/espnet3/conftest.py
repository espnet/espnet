import contextlib
import importlib.machinery
import os
import sys
import tempfile
import types


def _install_stub(name: str, module: types.ModuleType) -> None:
    sys.modules.setdefault(name, module)


def pytest_configure():
    """Inject lightweight stubs for optional heavy dependencies.

    These stubs keep imports working in CI environments that lack the actual
    packages while keeping the tests focused on ESPnet3 logic.
    """
    # Force tests to stay on CPU to avoid CUDA initialization on machines without GPUs.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    # Disable numba JIT to prevent cache issues from site-packages
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    # ---- g2p_en stub (avoids pulling in scipy/sklearn stacks) ----
    if "g2p_en" not in sys.modules:
        g2p_en = types.ModuleType("g2p_en")

        class DummyG2p:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, text):
                # Return an empty sequence to satisfy callers that iterate
                return []

        g2p_en.G2p = DummyG2p
        g2p_en.__spec__ = importlib.machinery.ModuleSpec("g2p_en", loader=None)
        _install_stub("g2p_en", g2p_en)

    # ---- vietnamese_cleaner stub (avoids underthesea/nltk + heavy deps) ----
    if "vietnamese_cleaner" not in sys.modules:
        vnm = types.ModuleType("vietnamese_cleaner")

        def vietnamese_cleaner(text: str, *args, **kwargs):
            return text

        def vietnamese_cleaners(text: str, *args, **kwargs):
            return text

        vnm.vietnamese_cleaner = vietnamese_cleaner
        vnm.vietnamese_cleaners = vietnamese_cleaners
        vnm.__spec__ = importlib.machinery.ModuleSpec("vietnamese_cleaner", loader=None)
        _install_stub("vietnamese_cleaner", vnm)

    # ---- dask.utils.tmpfile fallback (used by async runner path) ----
    try:
        import dask.utils  # type: ignore  # noqa: F401
    except Exception:
        dask_mod = types.ModuleType("dask")
        utils_mod = types.ModuleType("dask.utils")

        @contextlib.contextmanager
        def tmpfile(extension: str = ""):
            fd, path = tempfile.mkstemp(suffix=extension)
            os.close(fd)
            try:
                yield path
            finally:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass

        utils_mod.tmpfile = tmpfile  # type: ignore[attr-defined]
        dask_mod.utils = utils_mod  # type: ignore[attr-defined]
        _install_stub("dask", dask_mod)
        _install_stub("dask.utils", utils_mod)
