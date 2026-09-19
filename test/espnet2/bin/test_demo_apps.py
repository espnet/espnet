"""The two OWSM demos must not drift apart, from each other or from espnet.

Each demo is uploaded as a Hugging Face Space, and a Space is exactly one
directory: `hf upload espnet/owsm-v4 egs2/owsm_v4/s2t1/demo .` sends that
directory and nothing else. So the parts both apps share - the language
table, the helpers, the interface - are copied rather than imported, and a
fix applied to one could be forgotten in the other. These tests fail when
that happens.

`espnet demo` (espnet2/bin/demo.py) is a third copy of the same pieces, and
the only one a Space could import. It does not: a Space installs espnet from
PyPI, so importing a module newer than every release would break both live
demos the next time one of them is uploaded. The apps switch to importing it
once a release carries it; until then the comparisons at the bottom of this
file are what hold the three together.
"""

import ast
import re
import types
from pathlib import Path

import librosa
import numpy as np
import pytest

from espnet2.bin import demo

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


# --- and neither may drift from espnet2.bin.demo, which `espnet demo` runs ---

# A token list shaped like OWSM's, so the menus can be built from something
# smaller than a 1B checkpoint.
TOKENS = ["<unk>", "<nolang>", "<eng>", "<deu>", "<asr>", "<st_deu>", "<sos>"]
# The names each app defines that espnet2.bin.demo also defines. Everything
# else in an app is its own: the ZeroGPU shim, the model, the page.
SHARED = (
    "SAMPLE_RATE",
    "WINDOW_SECS",
    "MAX_SECS",
    "DETECT",
    "ASR_LABEL",
    "LANGUAGE_NAMES",
)
HELPERS = ("_names", "_language_codes", "_target_codes", "_pad", "_split_tokens")
MENUS = ("LANGUAGES", "TARGETS", "LANGUAGE_CODES")


def _app(name):
    """The app's own constants and helpers, run without importing the app.

    Importing app.py downloads a 1B checkpoint, needs gradio and builds a
    page, so the definitions these tests compare are lifted out of its syntax
    tree and executed on their own, with the loaded model stood in for.
    """
    wanted = set(SHARED) | set(HELPERS) | set(MENUS)
    body = []
    for node in ast.parse(_source(name)).body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            body.append(node)
        elif isinstance(node, ast.Assign):
            named = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if named & wanted:
                body.append(node)
    namespace = {
        "re": re,
        "librosa": librosa,
        # the app reads nothing else off the model at module level
        "s2t": types.SimpleNamespace(
            s2t_model=types.SimpleNamespace(token_list=TOKENS)
        ),
    }
    code = compile(ast.Module(body=body, type_ignores=[]), str(name), "exec")
    exec(code, namespace)
    return namespace


@pytest.mark.parametrize("name", list(DEMOS))
@pytest.mark.parametrize("constant", SHARED)
def test_the_apps_constants_are_the_ones_espnet_demo_uses(name, constant):
    assert _app(name)[constant] == getattr(demo, constant), f"{name}: {constant}"


@pytest.mark.parametrize("name", list(DEMOS))
def test_the_apps_menus_are_the_ones_espnet_demo_builds(name):
    app = _app(name)

    # both read the dropdowns off the checkpoint; they must read them alike
    assert (app["LANGUAGES"], app["TARGETS"]) == demo.menus(TOKENS)


@pytest.mark.parametrize("name", list(DEMOS))
def test_the_apps_padding_is_what_espnet_demo_pads_to(name):
    # librosa.util reaches for scipy.ndimage, which a mismatched numpy breaks
    pytest.importorskip("scipy.ndimage")
    speech = np.random.default_rng(0).standard_normal(16000, dtype="float32")

    # np.pad in espnet2.bin.demo, librosa.util.fix_length here: same array
    assert np.array_equal(_app(name)["_pad"](speech), demo.pad(speech))


def test_the_ctc_apps_symbol_splitting_is_what_espnet_demo_splits():
    app = _app("ctc")  # only the CTC app reads symbols back out of its output

    for decoded in ("<eng><asr><0.00> hello there", "<asr> hello", " plain "):
        assert app["_split_tokens"](decoded) == demo.split_tokens(
            decoded, app["LANGUAGE_CODES"]
        ), decoded
