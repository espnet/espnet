"""Shared corpus paths for the AMI SOT recipe tests.

The tests import this module as ``import ami_sot_paths``. pytest adds this directory
to sys.path, and no other module in the repo has this name. So the import cannot
clash with another module. A second ``conftest`` module could clash with the first.

Every path that these tests need is read from an environment variable, so the tests
contain no path from one machine. When a variable is unset, its path is ``None`` and
the tests that need it are skipped. This happens in upstream CI, which has neither
the AMI corpus nor a trained checkpoint.

The recipe reads the first two variables too, so one setting configures both the
recipe and its tests.

``AMI_SOT_DATA_ROOT`` is the root directory of the prepared split directories, as in
``dataset/config.yaml``. ``AMI_SOT_CUTSET_DIR`` is the directory with the Lhotse
CutSet manifests, as in ``dataset/config.yaml``. ``AMI_SOT_REFERENCE_DECODE`` is the
directory of a recorded ``1best_recog`` decode (``text`` and ``text_sot``).
``AMI_SOT_REFERENCE_TEXT`` is a recorded reference transcript in ``utt_id text``
form.

The recipe has no setting for the last two, because only the tests use them. They
contain outputs recorded from the reference run. A comparison with these recordings
can find problems that a comparison with the recipe's own corpus cannot find.
"""

import functools
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
RECIPE = REPO / "egs3" / "ami" / "esp2_s2t"

# The recipe's configs are merged over the conf/ directory of this package. run.py
# calls launch(ESP2S2TSystem), and default_conf_package in launch.py derives this
# package name from the system module. The tests use this one constant, so they
# load the same defaults as run.py.
TEMPLATE_PACKAGE = "egs3.TEMPLATE.esp2_s2t"


def _env_path(name: str):
    """Return the path in ``name``, or None when it is unset or empty."""
    value = os.environ.get(name)
    return Path(value) if value else None


DATA_ROOT = _env_path("AMI_SOT_DATA_ROOT")
CUTSET_DIR = _env_path("AMI_SOT_CUTSET_DIR")
REFERENCE_DECODE = _env_path("AMI_SOT_REFERENCE_DECODE")
REFERENCE_TEXT = _env_path("AMI_SOT_REFERENCE_TEXT")

# The cuts file of the prepared test split. A prepared split has only this file.
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

    Every test here loads the module by its file path. The helpers below load it
    the same way, so they read the same constants that the tests check.
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

    Some regression tests compare a score on the whole test set with a recorded
    decode. They need the reference in the same two-column format as the decode. A
    prepared split is a CutSet, so this function reads the text with the recipe's
    own reader. The recipe no longer writes a separate text file.

    Args:
        path: File to write.
    """
    cuts = dataset_module().load_cuts("test")
    Path(path).write_text(
        "".join(f"{cut.id} {cut.custom['sot_text']}\n" for cut in cuts),
        encoding="utf-8",
    )


def write_cuts(split_dir, ids) -> None:
    """Write a split of ``ids`` to ``split_dir`` with audio that no test reads.

    The recordings are built by hand, and their audio files do not exist. A test
    that only needs a split to exist then needs only the manifest.

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
    """Return True when every manifest listed in the recipe config exists.

    If only the directory were checked, a directory with other file names would make
    the test fail with an error. This check makes the test skip in that case.
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
