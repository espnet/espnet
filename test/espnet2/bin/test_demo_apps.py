"""The demo apps must not drift apart.

Each demo is uploaded as a Hugging Face Space, and a Space is exactly one
directory: `hf upload espnet/owsm-v4 egs2/owsm_v4/s2t1/demo .` sends that
directory and nothing else. So everything the apps have in common - the
ZeroGPU shim and the device rule in all five, and in the two OWSM demos the
language table, the helpers and the interface as well - is copied rather than
imported, and a fix applied to one could be forgotten in the others. These
tests fail when that happens.

They read the apps as text. Importing one would want gradio and would
download a checkpoint, and it is the source the Hub receives that these
checks are about.
"""

import ast
import re
from pathlib import Path

import pytest
import yaml

EGS2 = Path(__file__).parents[3] / "egs2"
DEMOS = {
    "ctc": EGS2 / "owsm_ctc_v4/s2t1/demo",
    "attention": EGS2 / "owsm_v4/s2t1/demo",
    "tts": EGS2 / "ljspeech/tts1/demo",
    "enh": EGS2 / "universal_se_v1/enh1/demo",
    "spk": EGS2 / "voxceleb/spk1/demo",
}
# The two that decode speech with OWSM, and so share far more than the rest.
OWSM = ("ctc", "attention")
# Everything from the ZeroGPU shim to the model section is common ground
# between the two OWSM apps.
SHARED_FROM = "# The ZeroGPU package patches torch"
SHARED_TO = "TITLE = "
# These two blocks are common ground between all five.
SHIM_TO = "import os  # noqa: E402"
DEVICE_FROM = "# ZeroGPU attaches the GPU only while"
DEVICE_TO = '    DEVICE = "cpu"'
OTHERS = [name for name in DEMOS if name not in OWSM]


def _source(name):
    return (DEMOS[name] / "app.py").read_text()


def _block(name, start, end, keep_end=False):
    """The lines of one region of an app, blank lines dropped."""
    text = _source(name)
    stop = text.index(end) + (len(end) if keep_end else 0)
    return [
        line for line in text[text.index(start) : stop].splitlines() if line.strip()
    ]


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


def _front_matter(name):
    card = (DEMOS[name] / "README.md").read_text()
    assert card.startswith("---"), name
    return yaml.safe_load(card.split("---", 2)[1])


def _caps(name):
    """The input cap each app declares, as (constant name, value) pairs."""
    return [
        (m.group(1), int(m.group(2)))
        for m in re.finditer(r"^(MAX_[A-Z]+) = (\d+)$", _source(name), re.M)
    ]


def _named_in_messages(name):
    """Every identifier interpolated into a gr.Warning or gr.Error message.

    Read from the syntax tree rather than by matching text, so that a message
    split across lines or rewrapped by black still counts.
    """
    named = set()
    for node in ast.walk(ast.parse(_source(name))):
        if not isinstance(node, ast.Call):
            continue
        called = node.func
        if not isinstance(called, ast.Attribute) or called.attr not in (
            "Warning",
            "Error",
        ):
            continue
        named.update(
            child.id for child in ast.walk(node) if isinstance(child, ast.Name)
        )
    return named


@pytest.mark.parametrize("name", list(DEMOS))
def test_every_demo_has_the_three_files_a_space_needs(name):
    for filename in ("app.py", "README.md", "requirements.txt"):
        assert (DEMOS[name] / filename).is_file(), f"{name}: {filename}"


def test_the_shared_half_of_the_two_owsm_apps_is_identical():
    assert _shared("ctc") == _shared("attention")


@pytest.mark.parametrize("name", OTHERS)
def test_every_app_keeps_the_same_zerogpu_shim(name):
    """ZeroGPU's package is imported the same way everywhere, or not at all."""
    assert _block(name, SHARED_FROM, SHIM_TO) == _block("ctc", SHARED_FROM, SHIM_TO)


@pytest.mark.parametrize("name", OTHERS)
def test_every_app_keeps_the_same_device_rule(name):
    """DEVICE, then SPACES_ZERO_GPU, then torch - in every app, word for word.

    Getting this one wrong is silent: the app runs on the CPU of a machine
    rented for its GPU, and only the clock says so.
    """
    mine = _block(name, DEVICE_FROM, DEVICE_TO, keep_end=True)
    assert mine == _block("ctc", DEVICE_FROM, DEVICE_TO, keep_end=True)


def _predict_signature(name):
    return re.search(r"^def predict\(([^)]*)\)", _source(name), re.M).group(1)


def test_both_owsm_demos_offer_the_same_tasks():
    for name in OWSM:
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
    granted = int(re.search(r"^GPU_SECONDS = (\d+)", source, re.M).group(1))
    for constant, limit in _caps(name):
        # a cap counted in characters says nothing about seconds
        if constant == "MAX_SECS":
            assert limit <= granted, f"{name}: takes {limit}s but asks for {granted}s"


@pytest.mark.parametrize("name", list(DEMOS))
def test_each_demo_caps_its_input_and_says_so(name):
    """A cap nobody is told about reads as the model losing the tail."""
    caps = _caps(name)
    assert len(caps) == 1, f"{name}: expected one MAX_* cap, found {caps}"
    constant, _ = caps[0]
    assert constant in _named_in_messages(name), (
        f"{name}: {constant} trims the input without a gr.Warning or gr.Error "
        "naming it"
    )


@pytest.mark.parametrize("name", list(DEMOS))
def test_spaces_is_imported_before_torch(name):
    """ZeroGPU patches torch as it is imported, so its package has to be first."""
    source = _source(name)
    assert source.index("import spaces") < source.index("import torch"), name


@pytest.mark.parametrize("name", list(DEMOS))
def test_the_card_names_the_app_and_the_model_the_app_loads(name):
    front = _front_matter(name)
    assert front.get("app_file") == "app.py", name
    tag = re.search(
        r'MODEL_TAG = os\.environ\.get\(\s*"[A-Z_]+_MODEL_TAG",\s*"([^"]+)"',
        _source(name),
    ).group(1)
    assert tag in (front.get("models") or []), f"{name}: the card does not list {tag}"


@pytest.mark.parametrize("name", list(DEMOS))
def test_the_card_carries_what_the_hub_and_espnet_need(name):
    """What the Hub refuses, and the one thing it accepts that espnet cannot.

    A Space image built on the runtime's default Python installs no espnet at
    all - pip reports "No matching distribution found" - so the interpreter is
    pinned, and pinned as a string: YAML reads an unquoted 3.12 as a float,
    and 3.1 is not what anyone meant.
    """
    front = _front_matter(name)
    for key in ("title", "sdk", "sdk_version", "app_file", "short_description"):
        assert isinstance(front.get(key), str) and front[key].strip(), f"{name}: {key}"
    assert len(front["short_description"]) <= 60, name
    assert front.get("python_version") == "3.12", name
