"""Pytest configuration for espnet2/speechlm/bin tests.

Imports and runs the sibling conftests' stub installation functions so
`espnet2.speechlm.bin.train` can be imported on CI without any of its
heavy optional deps (arkive, duckdb, pyarrow, lhotse, soundfile, kaldiio,
deepspeed, wandb, humanfriendly, torchtitan, parallel_utils, transformers
extras).

Keeping the stub logic in the sibling conftests avoids duplication — this
conftest is a thin composer.
"""

import importlib.util
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SPEECHLM_TEST = _HERE.parent


def _load_and_configure(sibling_dir: str) -> None:
    """Load a sibling conftest.py and run its pytest_configure()."""
    path = _SPEECHLM_TEST / sibling_dir / "conftest.py"
    spec = importlib.util.spec_from_file_location(
        f"_speechlm_test_{sibling_dir}_conftest", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "pytest_configure"):
        module.pytest_configure()


def pytest_configure():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    # Dataloader stubs: arkive, duckdb, pyarrow, lhotse, soundfile, kaldiio
    _load_and_configure("dataloader")
    # Model stubs: transformers / joblib / humanfriendly extras
    _load_and_configure("model")
    # Trainer stubs: deepspeed, wandb, humanfriendly, torchtitan, parallel_utils
    _load_and_configure("trainer")
