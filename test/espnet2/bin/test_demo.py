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


def test_the_written_input_tasks_appear_only_for_a_checkpoint_that_has_them():
    """POWSM answers <g2p> and <p2g>; OWSM has neither and its page is unchanged."""
    assert demo.prompt_tasks(["<eng>", "<asr>", "<pr>", "<g2p>", "<p2g>"]) == [
        demo.G2P_LABEL,
        demo.P2G_LABEL,
    ]
    assert demo.prompt_tasks(["<eng>", "<asr>", "<p2g>"]) == [demo.P2G_LABEL]
    assert demo.prompt_tasks(["<nolang>", "<eng>", "<asr>", "<st_deu>"]) == []


def test_phones_are_taken_in_either_spelling():
    """The page prints them spaced; POWSM reads them between slashes.

    Someone typing phones into the box will copy what the page showed them,
    and someone who knows the training data will type its own spelling. Both
    have to work, and neither may be turned into the other twice.
    """
    assert demo.as_phones("ð ə s e ɪ l") == "/ð//ə//s//e//ɪ//l/"
    assert demo.as_phones("/ð//ə//s/") == "/ð//ə//s/"
    assert demo.as_phones("  ") == ""


def test_the_written_input_box_appears_for_the_tasks_that_read_it(monkeypatch):
    """One box: hidden on a CTC checkpoint until a task asks for it.

    A checkpoint with a decoder has it open from the start, since anything
    typed there primes the search. One with none has nothing to prime, so
    the box appears only for `<g2p>` and `<p2g>`, whose input is written
    whether or not there is a decoder.
    """
    pytest.importorskip("gradio")

    class _Powsm:
        preprocessor_conf = {"speech_length": 20, "nolang_symbol": "<unk>"}
        ctc_only = True
        s2t_model = types.SimpleNamespace(
            token_list=["<unk>", "<eng>", "<asr>", "<pr>", "<g2p>", "<p2g>", "<sos>"]
        )

        def no_language(self):
            return "<unk>"

    app = demo.build_app(_Powsm(), device="cpu", model_tag="espnet/a-model")
    blocks = list(app.blocks.values())
    tasks = [
        c[0] if isinstance(c, (list, tuple)) else c
        for block in blocks
        if getattr(block, "label", None) == "Task"
        for c in block.choices
    ]
    assert demo.G2P_LABEL in tasks and demo.P2G_LABEL in tasks

    box = [b for b in blocks if getattr(b, "label", None) == demo.PROMPT_LABEL]
    assert len(box) == 1, "the box is one box"
    assert box[0].visible is False, "and hidden until a task asks for it"

    # the two notes: why there is nothing to prime, and what the prompted
    # tasks are worth on a checkpoint of this kind
    markdown = "\n".join(str(getattr(b, "value", "")) for b in blocks)
    assert demo.NO_PROMPT_NOTE in markdown
    assert demo.PROMPT_TASK_NOTE in markdown


def test_the_page_says_which_model_it_is_serving(monkeypatch):
    """The page serves any checkpoint, so its heading is not OWSM's.

    A Space passes its own title and description; `espnet demo` passes the
    tag it was given and gets a heading that names it.
    """
    pytest.importorskip("gradio")

    class _Any:
        preprocessor_conf = {"speech_length": 30, "nolang_symbol": "<nolang>"}
        ctc_only = True
        s2t_model = types.SimpleNamespace(
            token_list=["<nolang>", "<eng>", "<asr>", "<sos>"]
        )

        def no_language(self):
            return "<nolang>"

    def markdown(app):
        return "\n".join(str(getattr(b, "value", "")) for b in app.blocks.values())

    given = demo.build_app(_Any(), device="cpu", model_tag="espnet/a-model")
    assert "espnet/a-model" in markdown(given)

    own = demo.build_app(
        _Any(),
        device="cpu",
        model_tag="espnet/a-model",
        title="A Model",
        description="# A Model\n\nWhat this one is for.",
    )
    assert "What this one is for." in markdown(own)
    assert "Speech in, text out" not in markdown(own)


def test_a_checkpoint_with_a_decoder_can_be_prompted(monkeypatch):
    """There the box is open from the start, and no note says otherwise."""
    pytest.importorskip("gradio")

    class _Searches:
        preprocessor_conf = {"speech_length": 30, "nolang_symbol": "<nolang>"}
        ctc_only = False
        s2t_model = types.SimpleNamespace(
            token_list=["<nolang>", "<eng>", "<asr>", "<st_deu>", "<sos>"]
        )

        def no_language(self):
            return "<nolang>"

    app = demo.build_app(_Searches(), device="cpu", model_tag="espnet/a-model")
    blocks = list(app.blocks.values())
    box = [b for b in blocks if getattr(b, "label", None) == demo.PROMPT_LABEL]
    assert len(box) == 1 and box[0].visible is True

    markdown = "\n".join(str(getattr(b, "value", "")) for b in blocks)
    assert demo.NO_PROMPT_NOTE not in markdown


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


def test_the_rate_comes_from_the_checkpoint_too():
    class _Fast:
        sample_rate = 24000

    class _Silent:
        preprocessor_conf = {}

    assert demo.sample_rate(_Fast()) == 24000
    # nothing said: the rate this page was written for, which is what the two
    # Space apps pass
    assert demo.sample_rate(_Silent()) == demo.SAMPLE_RATE


def test_padding_uses_the_rate_it_is_given():
    at_8k = demo.pad(np.ones(8000, dtype="float32"), 2, 8000)

    assert len(at_8k) == 16000
    assert demo.pad(np.ones(10, dtype="float32"), 1).shape == (demo.SAMPLE_RATE,)


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
