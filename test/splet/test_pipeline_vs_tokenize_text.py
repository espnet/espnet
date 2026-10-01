#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""A SPLET pipeline against espnet2's tokenize_text, composition included.

The individual steps are checked elsewhere -- the Whisper normalizer against
openai-whisper string for string, nlsyms removal against espnet2's own
tokenizers. What those tests do not check is the *order*, and order is
semantic: ``asr.sh`` collapses whitespace, then cleans, then tokenizes, and
nlsyms removal happens inside that last step. A pipeline that removes symbols
before cleaning gets a different answer, because the Whisper cleaners delete
anything in angle brackets themselves.

So this runs the real thing. ``espnet2.bin.tokenize_text.tokenize`` is what
``asr.sh:1697-1721`` pipes both sides through, called here with the arguments
that stage passes, and its output is compared token for token against the
equivalent SPLET pipeline.

splet/ may not import espnet2 (ci/check_splet_independence.py). A test may:
the rule is about the package, and comparing against what SPLET replaces is
the point of this one.
"""

import pathlib

import pytest

from splet.normalizers import build_normalizer
from splet.utterance_metrics.error_rate import TOKENIZER_CHOICES

# Deliberately awkward: runs of spaces, mixed case, nlsyms both standalone and
# glued to a word, punctuation, numbers and a contraction for the Whisper
# cleaners to chew on.
CORPUS = [
    "utt1 the  quick   <noise> brown fox",
    "utt2 Mr. O'Brien paid $1,250.50 in 1990",
    "utt3 <noise>",
    "utt4 a<noise>b glued to a word",
    "utt5 nothing to remove here",
    "utt6 [laughter] he said don't do that",
    "utt7    leading and trailing   ",
]

NLSYMS = ["<noise>", "[laughter]"]


def run_tokenize_text(tmp_path, token_type, cleaner, nlsyms_path):
    """Run asr.sh's scoring front-end and return one token list per line."""
    tokenize = pytest.importorskip("espnet2.bin.tokenize_text").tokenize

    source = tmp_path / f"in_{token_type}_{cleaner}.txt"
    source.write_text("\n".join(CORPUS) + "\n", encoding="utf-8")
    destination = tmp_path / f"out_{token_type}_{cleaner}.txt"

    # The arguments asr.sh:1699-1706 passes, and no others.
    tokenize(
        input=str(source),
        output=str(destination),
        field="2-",
        delimiter=None,
        token_type=token_type,
        space_symbol="<space>",
        non_linguistic_symbols=str(nlsyms_path),
        bpemodel=None,
        log_level="ERROR",
        write_vocabulary=False,
        vocabulary_size=0,
        remove_non_linguistic_symbols=True,
        cutoff=0,
        add_symbol=[],
        cleaner=cleaner,
        g2p=None,
        add_nonsplit_symbol=[],
    )
    return [
        line.split() for line in destination.read_text(encoding="utf-8").splitlines()
    ]


def run_splet(token_type, cleaner, nlsyms_path):
    """The SPLET pipeline that is supposed to mean the same thing."""
    steps = [{"name": "whitespace"}]
    if cleaner == "whisper_en":
        steps.append({"name": "whisper", "language": "en"})
    elif cleaner == "whisper_basic":
        steps.append({"name": "whisper", "language": "basic"})
    steps.append(
        {
            "name": "remove_tokens",
            "tokens_file": str(nlsyms_path),
            # WordTokenizer drops whole tokens; CharTokenizer consumes the
            # symbol wherever it starts.
            "match": "token" if token_type == "word" else "substring",
        }
    )
    normalize = build_normalizer(steps)
    tokenize = (
        TOKENIZER_CHOICES["word"]()
        if token_type == "word"
        else TOKENIZER_CHOICES["char"](space_symbol="<space>")
    )
    # The utterance id is stripped by the caller, as `-f 2-` strips it.
    return [tokenize(normalize(line.split(maxsplit=1)[1])) for line in CORPUS]


@pytest.fixture
def nlsyms_file(tmp_path) -> pathlib.Path:
    path = tmp_path / "nlsyms.txt"
    path.write_text("\n".join(NLSYMS) + "\n", encoding="utf-8")
    return path


@pytest.mark.execution_timeout(120.0)
@pytest.mark.parametrize("token_type", ["word", "char"])
@pytest.mark.parametrize("cleaner", [None, "whisper_en", "whisper_basic"])
def test_pipeline_matches_tokenize_text(tmp_path, nlsyms_file, token_type, cleaner):
    """Token for token, per utterance, so a mismatch names the line."""
    if cleaner is not None:
        pytest.importorskip("whisper.normalizers")

    expected = run_tokenize_text(tmp_path, token_type, cleaner, nlsyms_file)
    actual = run_splet(token_type, cleaner, nlsyms_file)

    assert len(actual) == len(expected)
    for line, (mine, theirs) in enumerate(zip(actual, expected)):
        assert mine == theirs, f"{CORPUS[line].split()[0]}: {mine} != {theirs}"


def test_punctuation_removal_must_not_precede_symbol_removal():
    """Order is semantic here, and getting it wrong is silent.

    remove_punctuation strips the angle brackets, so "<noise>" survives as the
    ordinary word "noise" and remove_tokens can no longer match it -- leaving
    a non-speech marker in the reference to be scored as a real word. The
    shipped config in splet/egs/asr.yaml orders these correctly; this is what
    stops that being tidied into the wrong order later.
    """
    text = "the <noise> quick brown fox"
    symbols = {"name": "remove_tokens", "tokens": ["<noise>"], "match": "substring"}
    punctuation = {"name": "remove_punctuation"}

    assert build_normalizer([symbols, punctuation])(text) == "the  quick brown fox"
    assert build_normalizer([punctuation, symbols])(text) == "the noise quick brown fox"
