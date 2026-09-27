"""Differential tests for the OWSM utterance layer.

``egs3/owsm_v4/owsm/dataset/utils.py`` is a port of
``egs2/owsm_v1/s2t1/local/utils.py``. These load the egs2 file at runtime and
compare against it, rather than asserting golden values, so a change on either
side surfaces instead of the two silently diverging.

This is the one part of the recipe where a regression would be invisible
downstream: every other step is caught by the dump comparison, but a rounding
change in ``time2token`` produces plausible-looking wrong data.
"""

from __future__ import annotations

import ast
import importlib.util
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPE_DIR = REPO_ROOT / "egs3" / "owsm_v4" / "owsm"
EGS2_UTILS = REPO_ROOT / "egs2" / "owsm_v1" / "s2t1" / "local" / "utils.py"

PACKING_CASES = 200
PACKING_SEED = 0

# The symbol inventory v4 actually emits. No script in the repo produces it:
# generate_nlsyms.py spells languages with the two-letter Whisper keys, and the
# ISO 639-3 spellings only appear after owsm_v3's local/filter_lang_id.py
# rewrites the prepared data. The published model is the only complete oracle.
OWSM_BPE_URL = (
    "https://huggingface.co/espnet/owsm_v4_medium_1B/resolve/main"
    "/data/token_list/bpe_unigram50000/bpe.model"
)
OWSM_BPE_CACHE = Path.home() / ".cache" / "espnet3" / "owsm_v4_bpe.model"

# Everything the wired-up sub-datasets can emit: SPGISpeech is English ASR,
# MuST-C is English ASR plus its 14 translation directions.
REQUIRED_LANGS = ["en"]
REQUIRED_ST_TARGETS = [
    "ar", "cs", "de", "es", "fa", "fr", "it", "nl", "pt", "ro", "ru", "tr",
    "vi", "zh",
]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ours():
    return _load(RECIPE_DIR / "dataset" / "utils.py", "_owsm_ours")


@pytest.fixture(scope="module")
def egs2():
    if not EGS2_UTILS.is_file():
        pytest.skip(f"egs2 reference not present: {EGS2_UTILS}")
    return _load(EGS2_UTILS, "_owsm_egs2")


@pytest.fixture(scope="module")
def released_symbols():
    """Special symbols of the published OWSM v4 model, cached between runs."""
    spm = pytest.importorskip("sentencepiece")
    if not OWSM_BPE_CACHE.is_file():
        import urllib.error
        import urllib.request

        OWSM_BPE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(OWSM_BPE_URL, OWSM_BPE_CACHE)
        except (urllib.error.URLError, OSError) as error:
            OWSM_BPE_CACHE.unlink(missing_ok=True)
            pytest.skip(f"cannot fetch {OWSM_BPE_URL}: {error}")

    processor = spm.SentencePieceProcessor(model_file=str(OWSM_BPE_CACHE))
    return {
        piece
        for piece in (
            processor.id_to_piece(i) for i in range(processor.get_piece_size())
        )
        if piece.startswith("<") and piece.endswith(">")
    }


def _recording(rng: random.Random, count: int) -> list[dict]:
    """One recording's segments, every seventh one over the 30 s limit.

    The long ones are the point: they are what make the packer drop a segment
    and emit the None that resets the next span's prev_text.
    """
    utts = []
    cursor = 0.0
    for i in range(count):
        duration = rng.uniform(31.0, 40.0) if i % 7 == 6 else rng.uniform(0.3, 9.0)
        start = cursor + rng.uniform(0.0, 0.4)
        utts.append(
            dict(
                utt_id="",
                wav_id="REC_talk_1",
                wav_path="/dev/null",
                start_time=round(start, 6),
                end_time=round(start + duration, 6),
                lang="<en>",
                task="<asr>",
                text=f"segment {i} text",
                asr_text=f"segment {i} text",
            )
        )
        cursor = start + duration
    return utts


def test_constants_match(ours, egs2):
    assert ours.SYMBOL_NA == egs2.SYMBOL_NA
    assert ours.SYMBOL_NOSPEECH == egs2.SYMBOL_NOSPEECH
    assert ours.SPEECH_MAX_LEN == egs2.SPEECH_MAX_LEN
    assert ours.SPEECH_RESOLUTION == egs2.SPEECH_RESOLUTION
    assert ours.SYMBOLS_TIME == egs2.SYMBOLS_TIME


def test_language_keys_and_order_match(ours, egs2):
    # Order matters: generate_nlsyms.py emits tokens in this order, which fixes
    # the token-id layout.
    assert list(ours.LANGUAGES) == list(egs2.LANGUAGES)


def test_time2token_matches_over_a_dense_grid(ours, egs2):
    for step in range(0, 30_001):
        value = step * 0.001
        assert ours.time2token(value) == egs2.time2token(value), value


def test_nlsyms_whisper_matches_the_generator(tmp_path, monkeypatch):
    """Run egs2's generate_nlsyms.py and diff its output against ours.

    The script writes data/nlsyms.txt relative to the cwd and guards on
    __main__, so it is executed rather than imported. This covers the
    whisper-spelled list exactly; the ISO 639-3 spellings v4 actually emits come
    from filter_lang_id.py applied to the data, are not produced by any script
    in the repo, and are checked by local/check_iso_table.py against a released
    model instead.
    """
    generator = EGS2_UTILS.parent / "generate_nlsyms.py"
    if not generator.is_file():
        pytest.skip(f"egs2 generator not present: {generator}")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.syspath_prepend(str(generator.parent))
    code = compile(generator.read_text(encoding="utf-8"), str(generator), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(generator)})

    generated = (tmp_path / "data" / "nlsyms.txt").read_text(
        encoding="utf-8"
    ).splitlines()

    ours_module = _load(RECIPE_DIR / "dataset" / "utils.py", "_owsm_ours_nlsyms")
    assert ours_module.nlsyms(scheme="whisper") == generated
    assert len(generated) == 1703


def test_nlsyms_iso_layout(ours):
    symbols = ours.nlsyms()
    assert len(symbols) == 1703
    assert symbols[:2] == [ours.SYMBOL_NA, ours.SYMBOL_NOSPEECH]
    assert symbols[2] == "<eng>"
    assert "<asr>" in symbols
    assert "<st_deu>" in symbols
    assert symbols[-1] == "<30.00>"


def test_lang_and_task_tokens_use_iso639_3(ours):
    assert ours.lang_token("en") == "<eng>"
    assert ours.lang_token("en", scheme="whisper") == "<en>"
    assert ours.task_token("asr") == "<asr>"
    assert ours.task_token("st", "de") == "<st_deu>"
    assert ours.task_token("st", "de", scheme="whisper") == "<st_de>"


def test_unknown_language_is_rejected(ours):
    with pytest.raises(ValueError):
        ours.lang_token("qq")


def test_packing_matches_including_dropped_segments(ours, egs2):
    rng = random.Random(PACKING_SEED)
    saw_a_drop = False
    for case in range(PACKING_CASES):
        fields = _recording(rng, rng.randint(1, 40))
        mine = ours.generate_long_utterances([ours.Utterance(**f) for f in fields])
        ref = egs2.generate_long_utterances([egs2.Utterance(**f) for f in fields])
        assert len(mine) == len(ref), f"case {case}"
        for a, b in zip(mine, ref):
            for field in (
                "utt_id",
                "text",
                "asr_text",
                "prev_text",
                "text_with_time",
                "start_time",
                "end_time",
            ):
                assert getattr(a, field) == getattr(b, field), f"case {case} {field}"
        if any(u.prev_text == ours.SYMBOL_NA for u in mine[1:]):
            saw_a_drop = True
    assert saw_a_drop, "no case exercised the drop sentinel; the fixture is wrong"


def test_fixed_symbols_and_timestamps_are_in_the_release(ours, released_symbols):
    for symbol in (ours.SYMBOL_NA, ours.SYMBOL_NOSPEECH, "<asr>"):
        assert symbol in released_symbols, symbol
    missing = [s for s in ours.SYMBOLS_TIME if s not in released_symbols]
    assert not missing, f"{len(missing)} timestamp tokens absent, e.g. {missing[:5]}"


def test_emitted_language_tokens_are_in_the_release(ours, released_symbols):
    missing = [
        code for code in REQUIRED_LANGS if ours.lang_token(code) not in released_symbols
    ]
    assert not missing, [f"{c} -> {ours.lang_token(c)}" for c in missing]


def test_emitted_task_tokens_are_in_the_release(ours, released_symbols):
    missing = [
        code
        for code in REQUIRED_ST_TARGETS
        if ours.task_token("st", code) not in released_symbols
    ]
    assert not missing, [f"{c} -> {ours.task_token('st', c)}" for c in missing]


def test_iso_table_agrees_with_the_release_where_it_overlaps(ours, released_symbols):
    """The 99-language table against v4's inventory.

    Codes absent from the release are reported, not failed: v4 carries the
    expanded 151-language set, and a code missing from it means that release has
    no such language rather than that the mapping is wrong. Anything this recipe
    actually emits is covered by the two tests above.
    """
    absent = [
        code for code in ours.LANGUAGES if ours.lang_token(code) not in released_symbols
    ]
    confirmed = len(ours.LANGUAGES) - len(absent)
    assert confirmed >= 90, (
        f"only {confirmed}/{len(ours.LANGUAGES)} codes confirmed; "
        f"absent: {' '.join(absent)}"
    )
    # Norwegian is the one the release actually settles: it ships <nob> and
    # <nno> but not the macrolanguage <nor>.
    assert ours.lang_token("no") == "<nob>"
    assert "<nor>" not in released_symbols


def test_no_builder_imports_its_dataset():
    """A builder must stay self-contained.

    egs3/spgispeech/asr reaches the reader from its builder at runtime via
    importlib and then reads the reader's private _examples; that cycle is what
    this forbids.
    """
    offenders = []
    for builder in (RECIPE_DIR / "dataset").rglob("builder.py"):
        tree = ast.parse(builder.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            for name in names:
                if name.split(".")[-1] == "dataset":
                    offenders.append(f"{builder.relative_to(REPO_ROOT)} -> {name}")
    assert not offenders, offenders
