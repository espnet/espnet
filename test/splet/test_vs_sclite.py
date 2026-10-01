#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""SPLET's error rates against sclite's, which are the ones egs2 publishes.

Two layers, because each catches what the other cannot:

* the golden file, `test_utils/splet/sclite_golden.json`, holds counts sclite
  produced. It runs everywhere, including the CI job that does not build
  SCTK, and it is what stops a change to the cost model passing unnoticed.
* the live comparison shells out to `sclite` when the binary is there. It is
  what stops the golden file from quietly becoming fiction.

The live layer includes a differential fuzz, because one hand-picked
transposition proves that a cost model was implemented, while a few hundred
random pairs prove it was implemented correctly.
"""

import json
import pathlib
import random
import shutil
import subprocess
import tempfile

import pytest

from splet import list_scoring, load_score_modules, load_summary

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "test_utils" / "splet"
GOLDEN = json.loads((FIXTURES / "sclite_golden.json").read_text(encoding="utf-8"))
CASES = sorted(name for name in GOLDEN if not name.startswith("_"))


def find_sclite():
    """Return the sclite binary, or None.

    PATH first, then the in-tree build, because `tools/extra_path.sh` is
    sourced by the egs2 recipes and not by the Python CI job.
    """
    found = shutil.which("sclite")
    if found:
        return found
    in_tree = REPO_ROOT / "tools" / "sctk" / "bin" / "sclite"
    return str(in_tree) if in_tree.exists() else None


SCLITE = find_sclite()
needs_sclite = pytest.mark.skipif(
    SCLITE is None, reason="sclite not built; run make -C tools sctk"
)


def read_scp(name):
    """Read a fixture as an ordered list of (key, text)."""
    pairs = []
    for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        pairs.append((fields[0], fields[1] if len(fields) > 1 else ""))
    return pairs


def splet_config(spec):
    """Translate a golden case's sclite flags into the metric's options."""
    flags = spec["sclite_flags"]
    entry = {"name": "wer", "case": "sensitive" if "-s" in flags else "fold"}
    if "NOASCII" in flags:
        entry["tokenizer"] = "noascii"
    elif spec["tokenization"] == "char":
        entry["tokenizer"] = "char"
        # asr.sh scores CER on CharTokenizer output, where a space is the
        # token "<space>". One token either way, so this is legibility.
        entry["tokenizer_conf"] = {"space_symbol": "<space>"}
    else:
        entry["tokenizer"] = "word"
    return entry


def score_with_splet(spec):
    """Score a golden case through the public API and pool the counts."""
    ref, hyp = read_scp(spec["ref"]), read_scp(spec["hyp"])
    modules = load_score_modules([splet_config(spec)])
    summary = load_summary(list_scoring(dict(hyp), modules, dict(ref)))
    return {
        "errors": summary["wer_errors"],
        "hits": summary["wer_hit"],
        "substitutions": summary["wer_sub"],
        "deletions": summary["wer_del"],
        "insertions": summary["wer_ins"],
        "ref_len": summary["wer_ref_len"],
        "hyp_len": summary["wer_hyp_len"],
    }


def run_sclite(ref_tokens, hyp_tokens, flags=()):
    """Score token lists with the real binary and return its counts."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for name, corpus in (("ref", ref_tokens), ("hyp", hyp_tokens)):
            with (tmp / f"{name}.trn").open("w", encoding="utf-8") as handle:
                for index, tokens in enumerate(corpus):
                    # -i rm reads the speaker as the id up to the first "-",
                    # so a bare utterance id needs a speaker prefix.
                    handle.write(f"{' '.join(tokens)} (spk-u{index:04d})\n")
        subprocess.run(
            [
                SCLITE,
                *flags,
                "-r",
                str(tmp / "ref.trn"),
                "trn",
                "-h",
                str(tmp / "hyp.trn"),
                "trn",
                "-i",
                "rm",
                "-o",
                "dtl",
                "-O",
                str(tmp),
            ],
            check=True,
            capture_output=True,
        )
        report = (tmp / "hyp.trn.dtl").read_text(encoding="utf-8")

    labels = {
        "Percent Total Error": "errors",
        "Percent Correct": "hits",
        "Percent Substitution": "substitutions",
        "Percent Deletions": "deletions",
        "Percent Insertions": "insertions",
    }
    counts = {}
    for line in report.splitlines():
        for label, key in labels.items():
            if line.startswith(label):
                counts[key] = int(line.split("(")[1].split(")")[0])
    return counts


@pytest.mark.parametrize("case", CASES)
def test_matches_the_sclite_golden(case):
    """Every count, not just the rate.

    A rate can match while the breakdown is wrong -- that is exactly the bug
    this package was written to fix -- so the substitutions, deletions,
    insertions and hits are all compared.
    """
    spec = GOLDEN[case]
    assert score_with_splet(spec) == pytest.approx(spec["counts"])


@needs_sclite
@pytest.mark.parametrize("case", CASES)
def test_the_golden_still_matches_the_binary(case):
    """Guard the goldens themselves against drifting from sclite."""
    spec = GOLDEN[case]
    entry = splet_config(spec)
    modules = load_score_modules([entry])
    tokenize = modules["wer"]["scorer"]["tokenizer"]
    fold = entry["case"] == "fold"

    def prepare(pairs):
        return [tokenize(text.lower() if fold else text) for _, text in pairs]

    # The fixtures are tokenized here and handed to sclite already split, so
    # NOASCII has nothing left to do; its flags stay off to avoid splitting
    # twice.
    live = run_sclite(
        prepare(read_scp(spec["ref"])),
        prepare(read_scp(spec["hyp"])),
        flags=[f for f in spec["sclite_flags"] if f not in ("-c", "NOASCII")],
    )
    for key, value in live.items():
        assert value == spec["counts"][key], f"{case}: {key} drifted"


@needs_sclite
@pytest.mark.execution_timeout(120.0)
def test_differential_fuzz_against_sclite():
    """Random pairs, compared against the binary on every count.

    The generator deliberately produces transpositions, repeats and heavy
    reordering, which is where a weighted cost model and a unit-cost one part
    company.
    """
    random.seed(20260929)
    vocabulary = "alpha bravo charlie delta echo foxtrot golf hotel".split()
    references, hypotheses = [], []
    for _ in range(200):
        ref = [random.choice(vocabulary) for _ in range(random.randint(1, 12))]
        hyp = [w for w in ref if random.random() > 0.25]
        hyp += [random.choice(vocabulary) for _ in range(random.randint(0, 3))]
        if hyp and random.random() < 0.5:
            random.shuffle(hyp)
        references.append(ref)
        # sclite rejects an empty hypothesis line, so those are covered by
        # the golden fixtures rather than here.
        hypotheses.append(hyp or [random.choice(vocabulary)])

    expected = run_sclite(references, hypotheses)

    modules = load_score_modules([{"name": "wer", "tokenizer": "word"}])
    summary = load_summary(
        list_scoring(
            {str(i): " ".join(h) for i, h in enumerate(hypotheses)},
            modules,
            {str(i): " ".join(r) for i, r in enumerate(references)},
        )
    )
    assert summary["wer_errors"] == expected["errors"]
    assert summary["wer_sub"] == expected["substitutions"]
    assert summary["wer_del"] == expected["deletions"]
    assert summary["wer_ins"] == expected["insertions"]
    assert summary["wer_hit"] == expected["hits"]


@needs_sclite
def test_optional_deletion_matches_sclite():
    """A reference token in parentheses costs 2 to delete, not 3.

    And it does not match the bare word: sclite compares the token as
    written, so "(um)" against a spoken "um" is a substitution.
    """
    modules = load_score_modules([{"name": "wer", "tokenizer": "word"}])
    for ref, hyp in (
        ("i (um) think so", "i think so"),
        ("i (um) think so", "i um think so"),
    ):
        summary = load_summary(list_scoring({"u": hyp}, modules, {"u": ref}))
        expected = run_sclite([ref.split()], [hyp.split()])
        assert summary["wer_sub"] == expected["substitutions"], (ref, hyp)
        assert summary["wer_del"] == expected["deletions"], (ref, hyp)
        assert summary["wer_ins"] == expected["insertions"], (ref, hyp)
