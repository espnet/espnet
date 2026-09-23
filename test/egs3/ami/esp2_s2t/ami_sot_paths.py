"""Shared corpus locations for the AMI SOT recipe tests.

Imported as ``import ami_sot_paths``: pytest prepends this directory to sys.path, and
the name is unique in the repo, so it cannot collide the way a second ``conftest``
would.

Every path these tests need comes from an environment variable, so the suite carries no
location belonging to one machine. An unset variable leaves its path ``None`` and the
tests that need it skip, which is what upstream CI does: it has neither the AMI corpus
nor a trained checkpoint.

The variable names are the ones the recipe itself reads, so pointing a run at a corpus
configures the recipe and its tests together:

``AMI_SOT_DATA_ROOT`` Root the prepared split directories live under, as in
``dataset/config.yaml``. ``AMI_SOT_CUTSET_DIR`` Directory holding the Lhotse CutSet
manifests, as in ``dataset/config.yaml``. ``AMI_SOT_REFERENCE_DECODE`` Directory holding
a recorded ``1best_recog`` decode (``text`` and ``text_sot``). Only the byte-exactness
regressions read it; it has no recipe-side counterpart because nothing but a test
consumes it.
"""

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
RECIPE = REPO / "egs3" / "ami" / "esp2_s2t"


def _env_path(name: str):
    """Return the path in ``name``, or None when it is unset or empty."""
    value = os.environ.get(name)
    return Path(value) if value else None


DATA_ROOT = _env_path("AMI_SOT_DATA_ROOT")
CUTSET_DIR = _env_path("AMI_SOT_CUTSET_DIR")
REFERENCE_DECODE = _env_path("AMI_SOT_REFERENCE_DECODE")

# The prepared test split's reference transcript, the file most regressions
# compare against.
TEST_TEXT = (DATA_ROOT / "data" / "test" / "text") if DATA_ROOT else None


def _have(path) -> bool:
    return path is not None and path.exists()


needs_corpus = pytest.mark.skipif(
    not _have(TEST_TEXT),
    reason="set AMI_SOT_DATA_ROOT to a prepared AMI SOT directory",
)


def _have_cutsets() -> bool:
    """Report whether the manifests the recipe names are all present.

    Gating on the directory alone turns a directory holding other filenames
    into a hard error inside the test rather than a clean skip.
    """
    if not _have(CUTSET_DIR):
        return False
    import yaml

    config = yaml.safe_load((RECIPE / "dataset" / "config.yaml").read_text())
    return all(
        (CUTSET_DIR / split["cutset"]).is_file()
        for split in config["builder"]["sot"].values()
    )


needs_cutsets = pytest.mark.skipif(
    not _have_cutsets(),
    reason="set AMI_SOT_CUTSET_DIR to a directory with the recipe's CutSet manifests",
)

needs_reference_decode = pytest.mark.skipif(
    not _have(REFERENCE_DECODE),
    reason="set AMI_SOT_REFERENCE_DECODE to a recorded 1best_recog directory",
)
