"""The two OWSM demos must not drift apart.

Each demo is uploaded as a Hugging Face Space, and a Space is exactly one
directory: `hf upload espnet/owsm-v4 egs2/owsm_v4/s2t1/demo .` sends that
directory and nothing else. So the parts both apps share - the language
table, the helpers, the interface - are copied rather than imported, and a
fix applied to one could be forgotten in the other. These tests fail when
that happens.
"""

import re
from pathlib import Path

import pytest

DEMOS = {
    "ctc": Path(__file__).parents[3] / "egs2/owsm_ctc_v4/s2t1/demo",
    "attention": Path(__file__).parents[3] / "egs2/owsm_v4/s2t1/demo",
}
# Everything from the ZeroGPU shim to the model section is common ground.
SHARED_FROM = "# The ZeroGPU package patches torch"
SHARED_TO = "TITLE = "


def _source(name):
    return (DEMOS[name] / "app.py").read_text()


def _shared(name):
    text = _source(name)
    start, end = text.index(SHARED_FROM), text.index(SHARED_TO)
    # the model tag and the espnet imports differ by design; everything else
    # in this region is the same text in both files
    body = text[start:end]
    body = re.sub(r"^from espnet2\.bin\..*$", "", body, flags=re.M)
    body = re.sub(r"^MODEL_TAG = .*$", "", body, flags=re.M)
    # one app imports one class and the other two, which leaves a different
    # number of blank lines behind; that is not drift
    return [line for line in body.splitlines() if line.strip()]


def test_both_demos_exist_with_the_three_files_a_space_needs():
    for name, directory in DEMOS.items():
        for filename in ("app.py", "README.md", "requirements.txt"):
            assert (directory / filename).is_file(), f"{name}: {filename}"


def test_the_shared_half_of_the_two_apps_is_identical():
    assert _shared("ctc") == _shared("attention")


def _predict_signature(name):
    return re.search(r"^def predict\(([^)]*)\)", _source(name), re.M).group(1)


def test_both_offer_the_same_tasks():
    for name in DEMOS:
        source = _source(name)
        assert "long_form" in source, name
        assert "st_" in source, name  # translation targets
        assert "DETECT" in source, name  # language identification
    # only the autoregressive one takes a prompt: OWSM-CTC ignores one
    assert "prompt" not in _predict_signature("ctc")
    assert "prompt" in _predict_signature("attention")


@pytest.mark.parametrize("name", list(DEMOS))
def test_each_demo_asks_for_the_gpu_time_it_limits_itself_to(name):
    source = _source(name)
    assert "@spaces.GPU(duration=GPU_SECONDS)" in source, name
    limit = int(re.search(r"^MAX_SECS = (\d+)", source, re.M).group(1))
    granted = int(re.search(r"^GPU_SECONDS = (\d+)", source, re.M).group(1))
    assert limit <= granted, f"{name}: refuses {limit}s but asks for {granted}s"


@pytest.mark.parametrize("name", list(DEMOS))
def test_spaces_is_imported_before_torch(name):
    """ZeroGPU patches torch as it is imported, so its package has to be first."""
    source = _source(name)
    assert source.index("import spaces") < source.index("import torch"), name


@pytest.mark.parametrize("name", list(DEMOS))
def test_the_card_names_the_app_and_the_model_the_app_loads(name):
    card = (DEMOS[name] / "README.md").read_text()
    assert card.startswith("---"), name
    front = card.split("---", 2)[1]
    assert "app_file: app.py" in front, name
    tag = re.search(
        r'^MODEL_TAG = os\.environ\.get\("OWSM_MODEL_TAG", "([^"]+)"',
        _source(name),
        re.M,
    ).group(1)
    assert tag in front, f"{name}: the card does not list {tag}"
