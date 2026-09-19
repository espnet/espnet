"""The pieces `espnet demo` and the two OWSM Spaces share.

Everything here is checked against a token list rather than a checkpoint: the
menus a user sees are the model's own tokens, and that is the part that would
silently go wrong when a new OWSM spells its symbols differently.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

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
