"""The demo apps must not drift apart.

Each demo is uploaded as a Hugging Face Space, and a Space is exactly one
directory: `hf upload espnet/owsm-v4 egs2/owsm_v4/s2t1/demo .` sends that
directory and nothing else. So what the apps have in common - the ZeroGPU
shim, the device rule, the shape of the card - is copied rather than
imported, and a fix applied to one could be forgotten in the others. These
tests fail when that happens.

They read the apps as text. Importing one would want gradio and would
download a checkpoint, and it is the source the Hub receives that these
checks are about.

The page itself is no longer copied. Every speech-to-text demo imports
`build_app` from espnet2.bin.demo - the module `espnet demo` serves - which
builds the page from the checkpoint it is handed. What used to be here to
hold three copies of that page together is gone with the copies; what is
left checks that the imports stay imports, and that everything around them
is the same in every Space.
"""

import ast
import re
from pathlib import Path

import pytest
import yaml
from packaging.version import Version

from espnet2.bin import demo

EGS2 = Path(__file__).parents[3] / "egs2"
DEMOS = {
    "ctc": EGS2 / "owsm_ctc_v4/s2t1/demo",
    "attention": EGS2 / "owsm_v4/s2t1/demo",
    "powsm": EGS2 / "powsm_ctc/s2t1/demo",
    "align": EGS2 / "owsm_ctc_v4/s2t1/demo_align",
    "tts": EGS2 / "ljspeech/tts1/demo",
    "enh": EGS2 / "universal_se_v1/enh1/demo",
    "spk": EGS2 / "voxceleb/spk1/demo",
}
# The apps with no page of their own: they import the one `espnet demo`
# serves, so the checks about an interface do not apply to them, and one
# about the import does. Every speech-to-text demo is now one of these -
# which is what retired the comparisons that used to hold the two OWSM apps
# against espnet2.bin.demo, since there is nothing left to drift.
IMPORTED = ("ctc", "attention", "powsm")
# The ZeroGPU shim, which every app repeats because a Space is one directory.
SHARED_FROM = "# The ZeroGPU package patches torch"
SHIM_TO = "import os  # noqa: E402"
DEVICE_FROM = "# ZeroGPU attaches the GPU only while"
DEVICE_TO = '    DEVICE = "cpu"'
OTHERS = [name for name in DEMOS if name != "ctc"]
# the apps that draw their own page, and so answer for what is on it
OWN_PAGE = [name for name in DEMOS if name not in IMPORTED]


def _source(name):
    return (DEMOS[name] / "app.py").read_text()


def _block(name, start, end, keep_end=False):
    """The lines of one region of an app, blank lines dropped."""
    text = _source(name)
    stop = text.index(end) + (len(end) if keep_end else 0)
    return [
        line for line in text[text.index(start) : stop].splitlines() if line.strip()
    ]


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
    split across lines or rewrapped by black still counts. Only the arguments
    of the call are read, which for these apps is the message itself.
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
        for argument in node.args:
            named.update(
                child.id for child in ast.walk(argument) if isinstance(child, ast.Name)
            )
    return named


def _compared_against(name):
    """Every identifier one side of a comparison is measured against.

    An app that declares a cap and mentions it in a warning but never tests
    the input against it would pass a check on the message alone - it would
    announce a limit it does not apply. Whether the branch then trims or
    refuses is left to the app: the alignment demo refuses, the rest trim.
    """
    named = set()
    for node in ast.walk(ast.parse(_source(name))):
        if not isinstance(node, ast.Compare):
            continue
        for side in [node.left, *node.comparators]:
            named.update(
                child.id for child in ast.walk(side) if isinstance(child, ast.Name)
            )
    return named


@pytest.mark.parametrize("name", list(DEMOS))
def test_every_demo_has_the_three_files_a_space_needs(name):
    for filename in ("app.py", "README.md", "requirements.txt"):
        assert (DEMOS[name] / filename).is_file(), f"{name}: {filename}"


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


@pytest.mark.parametrize("name", list(DEMOS))
def test_each_demo_asks_for_the_gpu_time_it_limits_itself_to(name):
    source = _source(name)
    # as a decorator on the app's own predict, or handed to build_app as the
    # wrapper for the one it builds
    assert "spaces.GPU(duration=GPU_SECONDS)" in source, name
    granted = int(re.search(r"^GPU_SECONDS = (\d+)", source, re.M).group(1))
    for constant, limit in _caps(name):
        # a cap counted in characters says nothing about seconds
        if constant == "MAX_SECS":
            assert limit <= granted, f"{name}: takes {limit}s but asks for {granted}s"


@pytest.mark.parametrize("name", OWN_PAGE)
def test_each_demo_caps_its_input_and_says_so(name):
    """One cap, applied to the input, and named in what the user is told.

    Either half alone is a demo that misleads: a cap nobody is told about
    reads as the model losing the tail, and a cap announced but never tested
    against the input is a promise the app does not keep.
    """
    caps = _caps(name)
    assert len(caps) == 1, f"{name}: expected one MAX_* cap, found {caps}"
    constant, _ = caps[0]
    assert constant in _compared_against(name), (
        f"{name}: {constant} is declared but the input is never measured " "against it"
    )
    assert constant in _named_in_messages(name), (
        f"{name}: {constant} limits the input without a gr.Warning or "
        "gr.Error naming it"
    )


@pytest.mark.parametrize("name", IMPORTED)
def test_the_imported_page_is_the_one_espnet_demo_serves(name):
    """An app with no interface has to have no interface.

    The point of importing `build_app` is that the Space and `espnet demo`
    cannot drift, which only holds while the app adds nothing of its own. Its
    input cap is then espnet2.bin.demo's, and that is what the ZeroGPU slice
    has to cover.
    """
    source = _source(name)
    assert "from espnet2.bin.demo import build_app" in source, name
    assert "wrap=spaces.GPU(duration=GPU_SECONDS)" in source, name
    assert "gr.Blocks" not in source, f"{name}: builds a page of its own"
    assert "def predict" not in source, f"{name}: decodes on its own"

    granted = int(re.search(r"^GPU_SECONDS = (\d+)", source, re.M).group(1))
    assert demo.MAX_SECS <= granted, (
        f"{name}: the page takes {demo.MAX_SECS}s of audio and the app asks "
        f"for {granted}s of GPU"
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


# --- what a Space installs, against what its app needs ---

# Names that no released espnet has yet. A Space installs espnet from PyPI,
# so an app calling one of these is a Space that builds and then fails to
# start; its requirements.txt has to ask for a release that carries it.
# 2026-09-20: uploading the CTC app with `espnet>=202609.post2` took the live
# owsm-ctc-v4 Space down that way.
UNRELEASED = {
    "best_path": "202610.post1",
    # build_app took its `wrap` argument, and offered a phone page for a
    # checkpoint that has <pr>, in 202610.post2
    "build_app": "202610.post2",
    # espnet2.bin.align, and with it `espnet align`, arrived in 202610.post2
    "ForcedAligner": "202610.post2",
}
# The extra each front-end needs, by the import that gives it away. RawNet3's
# asteroid_frontend imports asteroid_filterbanks, which only espnet[spk] has;
# leaving it out is the same failure, one package lower down.
EXTRA_FOR_TASK = {"spk": "spk", "enh": "enh", "tts": "tts"}


def _names_used(source):
    """Every name the app calls or imports, method or plain.

    Read from the syntax tree rather than by matching text: `demo =
    build_app(...)` is how every app calls the builder, and a check for
    `build_app(` at the start of a line sees none of them - which is how the
    release floor for an imported name went unchecked.
    """
    used = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            called = node.func
            used.add(getattr(called, "attr", None) or getattr(called, "id", None))
        elif isinstance(node, ast.ImportFrom):
            used.update(alias.name for alias in node.names)
    return used - {None}


def _requirements(name):
    return (DEMOS[name] / "requirements.txt").read_text()


def _espnet_requirement(name):
    for line in _requirements(name).splitlines():
        if line.strip().startswith("espnet") and "model_zoo" not in line:
            return line.strip()
    raise AssertionError(f"{name}: requirements.txt does not ask for espnet")


@pytest.mark.parametrize("name", list(DEMOS))
def test_every_demo_pins_a_lowest_espnet_it_works_with(name):
    assert ">=" in _espnet_requirement(name), name


@pytest.mark.parametrize("name", list(DEMOS))
def test_an_app_using_a_new_api_asks_for_the_release_that_has_it(name):
    source = _source(name)
    requirement = _espnet_requirement(name)
    called = _names_used(source)
    for attribute, since in UNRELEASED.items():
        if attribute not in called:
            continue
        floor = requirement.split(">=")[1].strip()
        # parsed, not compared as text: "202612rc1" sorts after "202612" as a
        # string and before it as a version, and so does "202610.post2"
        # against "202610.post10"
        assert Version(floor) >= Version(since), (
            f"{name}: the app calls {attribute}(), which no espnet before "
            f"{since} has, but requirements.txt asks for {requirement}"
        )


@pytest.mark.parametrize("name, extra", sorted(EXTRA_FOR_TASK.items()))
def test_the_task_demos_ask_for_their_extra(name, extra):
    assert f"espnet[{extra}]" in _espnet_requirement(name), (
        f"{name}: the checkpoint's front-end or criteria come from "
        f"espnet[{extra}], and a Space installs nothing else"
    )
