"""Tests for egs3/ami/s2t/src/inference.py."""

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[4]
_RECIPE = _REPO / "egs3" / "ami" / "s2t"

pytest.importorskip("whisper")


def _load_inference_module():
    sys.path.insert(0, str(_REPO))
    sys.path.insert(0, str(_RECIPE))
    spec = importlib.util.spec_from_file_location(
        "ami_s2t_inference", _RECIPE / "src" / "inference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inf = _load_inference_module()

# inference.py imports dataset.py through the real, dotted package path
# (``from egs3.ami.s2t.dataset.dataset import ...``), not the flat,
# file-path loading _load_inference_module() itself uses. Importing it the
# same way here means constructing an AmiSotDataset below registers itself
# on the exact module object inf.build_output reads from, same as a real run.
import egs3.ami.s2t.dataset.dataset as ds_mod  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_module_globals(monkeypatch):
    """Keep every test off the real corpus and the real Whisper assets.

    build_output resolves its utterance id through the split whichever AmiSotDataset
    this process most recently built was recorded with (in os.environ, see dataset.py),
    and renders through lazily built tokenizers. Resetting the split record and its
    cache before every test means no test observes a split an earlier test's dataset
    recorded, and seeding the tokenizers keeps every test off the real Whisper assets.
    """
    monkeypatch.delenv(ds_mod._CURRENT_SPLIT_ENV, raising=False)
    monkeypatch.setattr(ds_mod, "_CACHED_SPLIT", None)
    monkeypatch.setattr(ds_mod, "_CACHED_UTT_IDS", None)
    saved = (inf._CONVERTER, inf._TOKENIZER)
    yield
    inf._CONVERTER, inf._TOKENIZER = saved


def _build_active_dataset(monkeypatch, split, ids):
    """Construct a real AmiSotDataset for ``split`` without touching disk.

    Returns the constructed instance. Construction records ``split`` in os.environ,
    which is what build_output resolves ids through.
    """
    monkeypatch.setattr(ds_mod, "load_utt_ids", lambda s: list(ids))
    monkeypatch.setattr(
        ds_mod, "_load_wav_paths", lambda s: [Path(f"{utt}.wav") for utt in ids]
    )
    # filename= is how AmiSotDataset reads text.prev and text.ctc. A stub
    # without it only survives against a corpus that lacks those two files.
    monkeypatch.setattr(
        ds_mod,
        "load_reference_texts",
        lambda s, filename="text": {utt: "" for utt in ids},
    )
    return ds_mod.AmiSotDataset(split=split)


def test_strip_timestamps_removes_timestamp_tokens():
    text = "<|0.00|> hello world<|1.20|> <sc> <|0.58|> good morning<|2.40|>"
    assert inf.strip_timestamps(text) == "hello world <sc> good morning"


def test_strip_timestamps_collapses_repeated_whitespace():
    assert inf.strip_timestamps("<|0.00|>  a   b <|1.00|>") == "a b"


def test_build_output_emits_the_four_scp_fields(monkeypatch):
    """The measure stage reads four SCP files, two per metric."""
    _build_active_dataset(monkeypatch, "test", ["utt-0", "utt-1"])
    out = inf.build_output(
        data={"text": "<|0.00|> ref words<|1.00|>"},
        model_output=_fake_model_output("<|0.00|> hyp words<|1.00|>"),
        idx=0,
    )
    assert sorted(out) == ["hyp", "hyp_sot", "ref", "ref_sot", "utt_id"]
    assert out["utt_id"] == "utt-0"
    assert out["hyp_sot"] == "<|0.00|> hyp words<|1.00|>"
    assert out["hyp"] == "hyp words"
    assert out["ref_sot"] == "<|0.00|> ref words<|1.00|>"
    assert out["ref"] == "ref words"


def test_build_output_uses_the_real_ami_utterance_id(monkeypatch):
    """A sample cannot carry utt_id, so the id comes from the built dataset."""
    _build_active_dataset(
        monkeypatch, "test", ["EN2002a-0.37-11.83", "EN2002a-1032.74-1034.73"]
    )
    out = inf.build_output(
        data={"text": ""}, model_output=_fake_model_output(""), idx=1
    )
    assert out["utt_id"] == "EN2002a-1032.74-1034.73"


def test_build_output_attaches_ids_from_whatever_split_was_actually_built(
    monkeypatch,
):
    """Build output attaches ids from whatever split was actually built.

    The real bug: conf/inference.yaml (or a conf/tuning/ variant reached through
    --inference_config) can iterate a split other than "test".

    build_output must never hold its own, separately loaded idea of which split that is
    -- whichever AmiSotDataset the framework actually constructed for this run is the
    one its ids must come from. Proven here by building "test" and then "valid": "valid"
    is the one actually being iterated, so its id -- not test's -- must be what comes
    back.
    """
    _build_active_dataset(monkeypatch, "test", ["t-0", "t-1"])
    _build_active_dataset(monkeypatch, "valid", ["v-0", "v-1", "v-2"])
    out = inf.build_output(
        data={"text": ""}, model_output=_fake_model_output(""), idx=1
    )
    assert out["utt_id"] == "v-1"
    assert out["utt_id"] not in ("t-0", "t-1")


def test_build_output_survives_the_frameworks_own_module_loading(monkeypatch, tmp_path):
    """Build output survives the frameworks own module loading.

    The strongest form of the property, and the actual shape of the bug: ESPnet3 does
    not build AmiSotDataset through the same, stable, dotted
    `egs3.ami.s2t.dataset.dataset` path this test file's `ds_mod` (and src/inference.py)
    import by.

    It loads a recipe's own dataset/__init__.py through
    espnet3.components.data.dataset_module._load_local_dataset_module, under a fresh,
    uniquely named module spec, so the AmiSotDataset the framework actually constructs
    is a different class object than ds_mod.AmiSotDataset -- a plain Python global set
    by one is invisible to the other. This test goes through the real framework loader
    (not _build_active_dataset's same-module shortcut) to build "valid", the split that
    disagrees with the old hardcoded "test", and checks that inf.build_output -- loaded
    yet a third way, via _load_inference_module's flat file path -- still resolves the
    id that split actually assigned.
    """
    for split, rel_dir, ids in (
        ("test", "data/test", ["t-0"]),
        ("valid", "data/dev", ["v-0", "v-1"]),
    ):
        split_dir = tmp_path / rel_dir
        split_dir.mkdir(parents=True)
        (split_dir / "wav.scp").write_text(
            "".join(f"{utt} {utt}.wav\n" for utt in ids), encoding="utf-8"
        )
        (split_dir / "text").write_text(
            "".join(f"{utt}\n" for utt in ids), encoding="utf-8"
        )
    monkeypatch.setenv("AMI_SOT_DATA_ROOT", str(tmp_path))

    from espnet3.components.data.dataset_module import load_dataset_module

    module = load_dataset_module(data_src=None, recipe_dir=str(_RECIPE))
    assert module.Dataset is not ds_mod.AmiSotDataset  # the wall this proves
    module.Dataset(split="valid")

    out = inf.build_output(
        data={"text": ""}, model_output=_fake_model_output(""), idx=1
    )
    assert out["utt_id"] == "v-1"


def test_build_output_raises_when_no_dataset_has_been_built_yet():
    """Build output raises when no dataset has been built yet.

    Silently defaulting to "test" would mislabel a run whose dataset was never built (or
    failed to build) without raising anywhere.

    build_output must refuse instead.
    """
    with pytest.raises(RuntimeError, match="AmiSotDataset"):
        inf.build_output(data={"text": ""}, model_output=_fake_model_output(""), idx=0)


def test_build_output_rewrites_the_separator(monkeypatch):
    _build_active_dataset(monkeypatch, "test", ["utt-0"])
    out = inf.build_output(
        data={"text": ""},
        model_output=_fake_model_output("<|0.00|> a????b<|1.00|>"),
        idx=0,
    )
    assert " <sc> " in out["hyp_sot"]
    assert "????" not in out["hyp_sot"]


def _fake_model_output(rendered: str):
    """Mimic Speech2Text's return shape for one hypothesis.

    Speech2Text returns a list of tuples whose third element is the token id sequence.
    The recipe re-renders from those ids so that timestamps survive, so the test injects
    a tokenizer that returns a fixed rendering.
    """

    class _Tokenizer:
        def ids2tokens(self, ids):
            return list(ids)

        def tokens2text(self, tokens):
            return "".join(tokens)

    inf._CONVERTER = _Tokenizer()
    inf._TOKENIZER = _Tokenizer()
    ids = ["<|en|>", "<|transcribe|>"] + ([rendered] if rendered else [])
    return [("ignored", [], ids, None, None)]


def test_separator_defaults_to_the_released_checkpoints_symbol():
    """The released model separates speakers with four question marks."""
    assert inf.SPEAKER_CHANGE_SYMBOL == "????"


def test_separator_follows_the_environment_override(monkeypatch):
    """A checkpoint trained with another symbol needs no code edit."""
    monkeypatch.setenv("AMI_SOT_SPEAKER_CHANGE_SYMBOL", "@@")
    reloaded = _load_inference_module()
    assert reloaded.SPEAKER_CHANGE_SYMBOL == "@@"


def test_build_output_rewrites_whatever_symbol_is_configured(monkeypatch):
    """build_output must rewrite the configured symbol, not a literal."""
    monkeypatch.setenv("AMI_SOT_SPEAKER_CHANGE_SYMBOL", "@@")
    reloaded = _load_inference_module()
    _build_active_dataset(monkeypatch, "test", ["utt-0"])

    class _Tokenizer:
        def ids2tokens(self, ids):
            return list(ids)

        def tokens2text(self, tokens):
            return "".join(tokens)

    reloaded._CONVERTER = _Tokenizer()
    reloaded._TOKENIZER = _Tokenizer()
    out = reloaded.build_output(
        data={"text": "<|0.00|> r@@s<|1.00|>"},
        model_output=[
            ("x", [], ["<|en|>", "<|transcribe|>", "<|0.00|> a@@b<|1.00|>"], None, None)
        ],
        idx=0,
    )
    assert " <sc> " in out["hyp_sot"] and "@@" not in out["hyp_sot"]
    assert " <sc> " in out["ref_sot"] and "@@" not in out["ref_sot"]
