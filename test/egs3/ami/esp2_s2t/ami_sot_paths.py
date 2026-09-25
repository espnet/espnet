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
manifests, as in ``dataset/config.yaml``. ``AMI_SOT_REFERENCE_DECODE``
Directory holding a recorded ``1best_recog`` decode (``text`` and ``text_sot``).
``AMI_SOT_REFERENCE_TEXT`` A recorded reference transcript in ``utt_id text`` form.

The last two have no recipe-side counterpart because nothing but a test consumes
them. They are recordings of what the reference run produced, so comparing the
recipe's own output against them says something a comparison against the recipe's
own corpus could not.
"""

import functools
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
REFERENCE_TEXT = _env_path("AMI_SOT_REFERENCE_TEXT")

# The prepared test split's cuts, the one file a prepared split consists of.
TEST_CUTS = (DATA_ROOT / "data" / "test" / "cuts.jsonl.gz") if DATA_ROOT else None


def _have(path) -> bool:
    return path is not None and path.exists()


needs_corpus = pytest.mark.skipif(
    not _have(TEST_CUTS),
    reason="set AMI_SOT_DATA_ROOT to a prepared AMI SOT directory",
)


@functools.lru_cache(maxsize=1)
def dataset_module():
    """Load the recipe's dataset module by file path.

    Loading it by path rather than by dotted name is what every test here does, so
    the helpers below read the same constants the tests assert against.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ami_s2t_dataset_shared", RECIPE / "dataset" / "dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_test_reference(path) -> None:
    """Write the prepared test split as ``utt_id text`` lines to ``path``.

    The regressions that compare a whole-test-set score against a recorded decode
    need the reference in the same two-column shape as the decode. A prepared split
    is a CutSet, so the text comes from the recipe's own reader rather than from a
    file the recipe no longer writes.

    Args:
        path: File to write.
    """
    texts = dataset_module().load_reference_texts("test")
    Path(path).write_text(
        "".join(f"{utt} {text}\n" for utt, text in texts.items()), encoding="utf-8"
    )


def write_cuts(split_dir, ids) -> None:
    """Write a split of ``ids`` under ``split_dir`` whose audio no test reads.

    Building the recordings by hand rather than from real files keeps a test that
    only needs a split to exist down to the manifest.

    Args:
        split_dir: Directory to write the split into.
        ids: Utterance ids, one cut each, in order.
    """
    from lhotse import CutSet
    from lhotse.audio import AudioSource, Recording
    from lhotse.cut import MonoCut

    module = dataset_module()
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    cuts = []
    for utt in ids:
        recording = Recording(
            id=utt,
            sources=[AudioSource(type="file", channels=[0], source=f"{utt}.wav")],
            sampling_rate=16000,
            num_samples=16000,
            duration=1.0,
        )
        cut = MonoCut(id=utt, start=0.0, duration=1.0, channel=0, recording=recording)
        cut.custom = {module._SOT_TEXT_FIELD: utt}
        cuts.append(cut)
    CutSet.from_cuts(cuts).to_file(str(split_dir / module._CUTS_FILENAME))


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

needs_reference_text = pytest.mark.skipif(
    not _have(REFERENCE_TEXT),
    reason="set AMI_SOT_REFERENCE_TEXT to a recorded reference transcript",
)
