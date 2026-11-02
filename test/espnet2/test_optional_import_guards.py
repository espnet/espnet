"""An absent optional package must not disable an unrelated one.

Each of these modules guards two independent optional packages. Written as one
try around both - which is how they started - importing the module with either
one absent sets both to None, so the feature that *is* installed reports itself
as missing, and the error names the extra rather than the package that is gone.
The espnet[tts] case is the misleading one: jamo is Korean, g2p_en is English,
and a user who has the tts extra is told to install the tts extra.
"""

import builtins
import contextlib
import importlib

import pytest


@contextlib.contextmanager
def reloaded_without(monkeypatch, module_name, missing):
    """Reload module_name with `missing` unimportable, then put it back."""
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == missing or name.startswith(f"{missing}."):
            raise ImportError(f"No module named {missing!r}")
        return real_import(name, *args, **kwargs)

    module = importlib.import_module(module_name)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    try:
        yield importlib.reload(module)
    finally:
        monkeypatch.undo()
        importlib.reload(module)  # leave the real one behind for other tests


# (module, package to hide, its attribute, package that must survive, its attribute)
CASES = [
    ("espnet2.text.phoneme_tokenizer", "jamo", "jamo", "g2p_en", "g2p_en"),
    ("espnet2.text.phoneme_tokenizer", "g2p_en", "g2p_en", "jamo", "jamo"),
    (
        "espnet2.text.cleaner",
        "jaconv",
        "jaconv",
        "tacotron_cleaner",
        "tacotron_cleaners",
    ),
    (
        "espnet2.text.cleaner",
        "tacotron_cleaner",
        "tacotron_cleaners",
        "jaconv",
        "jaconv",
    ),
    (
        "espnet2.enh.loss.criterions.time_domain",
        "ci_sdr",
        "ci_sdr",
        "fast_bss_eval",
        "fast_bss_eval",
    ),
    (
        "espnet2.enh.loss.criterions.time_domain",
        "fast_bss_eval",
        "fast_bss_eval",
        "ci_sdr",
        "ci_sdr",
    ),
]


@pytest.mark.parametrize("module_name, hide, hidden_attr, keep, kept_attr", CASES)
def test_one_missing_package_does_not_disable_the_other(
    monkeypatch, module_name, hide, hidden_attr, keep, kept_attr
):
    pytest.importorskip(keep)
    with reloaded_without(monkeypatch, module_name, hide) as module:
        assert getattr(module, hidden_attr) is None
        assert getattr(module, kept_attr) is not None


# (module, package to hide, what to call, the name the error must contain)
MESSAGES = [
    ("espnet2.text.phoneme_tokenizer", "jamo", lambda m: m.Jaso(), "jamo"),
    ("espnet2.text.phoneme_tokenizer", "g2p_en", lambda m: m.G2p_en(), "g2p_en"),
    (
        "espnet2.text.cleaner",
        "jaconv",
        lambda m: m.TextCleaner(["jaconv"])("hello"),
        "jaconv",
    ),
    (
        "espnet2.text.cleaner",
        "tacotron_cleaner",
        lambda m: m.TextCleaner(["tacotron"])("hello"),
        "tacotron_cleaner",
    ),
    (
        "espnet2.enh.loss.criterions.time_domain",
        "ci_sdr",
        lambda m: m.CISDRLoss(),
        "ci_sdr",
    ),
    (
        "espnet2.enh.loss.criterions.time_domain",
        "fast_bss_eval",
        lambda m: m.SDRLoss(),
        "fast_bss_eval",
    ),
    (
        "espnet2.enh.loss.criterions.time_domain",
        "fast_bss_eval",
        lambda m: m.SISNRLoss(),
        "fast_bss_eval",
    ),
]


@pytest.mark.parametrize("module_name, hide, call, named", MESSAGES)
def test_the_error_names_the_package_that_is_missing(
    monkeypatch, module_name, hide, call, named
):
    with reloaded_without(monkeypatch, module_name, hide) as module:
        with pytest.raises(RuntimeError, match=named):
            call(module)


def test_english_still_works_without_korean(monkeypatch):
    pytest.importorskip("g2p_en")
    with reloaded_without(monkeypatch, "espnet2.text.phoneme_tokenizer", "jamo") as m:
        m.G2p_en()  # the point of the change: this used to raise
