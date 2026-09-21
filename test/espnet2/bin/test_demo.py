"""The pieces `espnet demo` and the two OWSM Spaces share.

Everything here is checked against a token list rather than a checkpoint: the
menus a user sees are the model's own tokens, and that is the part that would
silently go wrong when a new OWSM spells its symbols differently.
"""

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from espnet2.bin import demo

# A token list shaped like OWSM's: languages first, then <asr>, then the
# translation targets and the symbols a decoder adds.
TOKENS = [
    "<unk>",
    "<nolang>",
    "<eng>",
    "<deu>",
    "<jpn>",
    "<asr>",
    "<st_deu>",
    "<st_eng>",
    "<sos>",
    "<eos>",
    "<sop>",
    "<0.00>",
    "hello",
]


def test_the_language_menu_is_the_checkpoints_own_symbols():
    languages, _ = demo.menus(TOKENS)

    # sorted by label, named where a name is known; <nolang> and <unk> are
    # not languages a user picks
    assert languages == [
        ("English (eng)", "eng"),
        ("German (deu)", "deu"),
        ("Japanese (jpn)", "jpn"),
    ]


def test_language_codes_stop_at_asr_and_skip_unk():
    assert demo.language_codes(TOKENS) == ["eng", "deu", "jpn"]


def test_the_translation_targets_are_the_st_symbols():
    assert demo.target_codes(TOKENS) == ["deu", "eng"]

    _, targets = demo.menus(TOKENS)
    assert targets == [
        ("Translate to English (eng)", "eng"),
        ("Translate to German (deu)", "deu"),
    ]


def test_a_language_without_an_english_name_keeps_its_code():
    languages, _ = demo.menus(["<xyz>", "<asr>"])
    assert languages == [("xyz (xyz)", "xyz")]


def test_split_tokens_takes_the_language_and_leaves_the_text():
    codes = demo.language_codes(TOKENS)

    assert demo.split_tokens("<eng><asr><0.00> hello there", codes) == (
        "eng",
        "hello there",
    )
    # <asr> is three lowercase letters too, and is not a language
    assert demo.split_tokens("<asr> hello", codes) == ("", "hello")
    assert demo.split_tokens("  plain text ", codes) == ("", "plain text")


def test_padding_fills_the_window_a_short_clip_leaves_empty():
    padded = demo.pad(np.ones(demo.SAMPLE_RATE, dtype="float32"))

    assert len(padded) == demo.SAMPLE_RATE * demo.WINDOW_SECS
    assert padded[: demo.SAMPLE_RATE].all() and not padded[demo.SAMPLE_RATE :].any()


def test_padding_never_returns_more_than_one_window():
    long_clip = np.ones(demo.SAMPLE_RATE * (demo.WINDOW_SECS + 10), dtype="float32")

    assert len(demo.pad(long_clip)) == demo.SAMPLE_RATE * demo.WINDOW_SECS


def test_padding_can_be_told_a_different_window():
    # POWSM's window is 20 s, and padding it to OWSM's 30 would be ten
    # seconds of silence for the model to hallucinate over
    padded = demo.pad(np.ones(demo.SAMPLE_RATE, dtype="float32"), 20)

    assert len(padded) == demo.SAMPLE_RATE * 20


def test_the_window_is_the_one_the_checkpoint_was_trained_on():
    class _Model:
        preprocessor_conf = {"speech_length": 20}

    assert demo.window_secs(_Model()) == 20

    class _Silent:
        preprocessor_conf = {}

    # a config that does not say keeps OWSM's, which is what this page meant
    # for as long as it served one model
    assert demo.window_secs(_Silent()) == demo.WINDOW_SECS


def test_the_phone_option_appears_only_for_a_checkpoint_that_has_it():
    assert demo.phone_task(["<eng>", "<asr>", "<pr>"])
    assert not demo.phone_task(["<eng>", "<asr>", "<st_deu>"])


def test_a_checkpoint_that_cannot_detect_a_language_opens_on_one(monkeypatch):
    """A model with no "work it out yourself" symbol must not break the page.

    espnet/powsm records none and has no <nolang> in its vocabulary at all.
    The menu then has no Detect entry and opens on a language, rather than
    every Run raising.
    """
    gr = pytest.importorskip("gradio")
    assert gr  # the page is built below

    class _Cannot:
        preprocessor_conf = {"speech_length": 20}
        s2t_model = types.SimpleNamespace(
            token_list=["<unk>", "<eng>", "<deu>", "<asr>", "<pr>", "<sos>"]
        )

        def no_language(self):
            raise ValueError("this model has no symbol for an unknown language")

    app = demo.build_app(_Cannot(), device="cpu", model_tag="espnet/a-model")
    menus = {
        block.label: [
            c[0] if isinstance(c, (list, tuple)) else c for c in block.choices
        ]
        for block in app.blocks.values()
        if getattr(block, "label", None) in ("Spoken language", "Task")
    }
    assert demo.DETECT not in menus["Spoken language"]
    assert menus["Spoken language"][0] == "English (eng)"
    # and the phone option is there, since this checkpoint has <pr>
    assert demo.PHONES_LABEL in menus["Task"]


def test_a_phonetic_checkpoint_says_what_it_is_for(monkeypatch):
    """The page offers Transcribe on every model; some are not built for it.

    Asked for by POWSM's author on #6776: its English ASR is weak, and a page
    that offers the button without saying so invites the wrong reading.
    """
    pytest.importorskip("gradio")

    class _Phonetic:
        preprocessor_conf = {"speech_length": 20, "nolang_symbol": "<unk>"}
        ctc_only = True
        s2t_model = types.SimpleNamespace(
            token_list=["<unk>", "<eng>", "<asr>", "<pr>", "<sos>"]
        )

        def no_language(self):
            return "<unk>"

    class _NotPhonetic(_Phonetic):
        s2t_model = types.SimpleNamespace(
            token_list=["<nolang>", "<eng>", "<asr>", "<st_deu>", "<sos>"]
        )

        def no_language(self):
            return "<nolang>"

    def markdown(model):
        app = demo.build_app(model, device="cpu", model_tag="espnet/a-model")
        return "\n".join(
            str(getattr(block, "value", "")) for block in app.blocks.values()
        )

    assert demo.PHONE_MODEL_NOTE in markdown(_Phonetic())
    assert demo.PHONE_MODEL_NOTE not in markdown(_NotPhonetic())


def test_the_device_rule_answers_a_device_torch_accepts():
    assert demo.default_device() in ("cpu", "cuda")


def test_importing_this_module_does_not_import_gradio():
    # the whole point of the [demo] extra: a bare install must import this
    # module, and ci/check_inference_imports.py fails if it stops doing so
    probe = "import espnet2.bin.demo, sys; print('gradio' in sys.modules)"
    r = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=Path(demo.__file__).parents[2],
    )

    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"


def _import_fails_with(monkeypatch, error):
    """Make `import gradio` raise, whether or not gradio is installed."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "gradio":
            raise error
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "gradio", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_a_missing_gradio_is_the_answer_none(monkeypatch):
    _import_fails_with(
        monkeypatch, ModuleNotFoundError("No module named 'gradio'", name="gradio")
    )

    assert demo.load_gradio() is None


def test_a_gradio_that_is_installed_but_broken_is_not_called_missing(monkeypatch):
    # "pip install espnet[demo]" would not fix this one, and naming gradio
    # instead of the package that is actually absent would send a user there
    _import_fails_with(
        monkeypatch, ModuleNotFoundError("No module named 'pandas'", name="pandas")
    )

    with pytest.raises(ModuleNotFoundError, match="pandas"):
        demo.load_gradio()
