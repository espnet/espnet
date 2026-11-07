"""Parity with sclite, the tool whose numbers ESPnet recipes publish.

Three layers, because each answers a question the others cannot:

* The **goldens** are sclite's own output for the fixtures in
  ``test_utils/splet``, recorded by ``test_utils/splet/make_golden.py``.
  They run everywhere, including the CI job that has no SCTK, which is the
  only reason parity is checkable at all on a normal machine.
* The **live comparison** runs the binary when there is one and checks the
  goldens still describe it, so an SCTK upgrade shows up here rather than
  in a recipe.
* The **fuzz** runs random pairs through both. The fixtures are ten
  utterances chosen by hand; they cannot cover the alignments that only
  turn up at scale.

Every count is compared, never only the rate: two implementations can agree
on the total and disagree on every one of S, D and I, and a results table
reports all four.
"""

import json
import pathlib
import random
import re
import shutil
import subprocess
import tempfile

import pytest

from splet.utterance_metrics.error_rate import (
    TOKENIZER_CHOICES,
    error_rate_metric,
    error_rate_setup,
)

FIXTURES = pathlib.Path(__file__).resolve().parents[2] / "test_utils" / "splet"
GOLDEN = json.loads((FIXTURES / "sclite_golden.json").read_text(encoding="utf-8"))
SPACE_SYMBOL = "<space>"


def _tokenizer_for(case):
    """Return the SPLET tokenizer a golden case needs.

    ``-c NOASCII`` is an sclite flag, not a tokenization: the case still
    records "word" because that is how its trn file was written, while
    sclite itself splits the non-ASCII words into characters. Reading
    only the tokenization would silently run it as plain WER.
    """
    if "NOASCII" in case["sclite_flags"]:
        return "noascii"
    return case["tokenization"]


def _sclite():
    """Return a working sclite, or None.

    Being on PATH is not the same as being runnable: ``tools/sctk`` can be a
    symlink into a directory shared between two clusters with different CPU
    architectures, where the binary exists and dies with "Exec format
    error". Run it and look for its banner.
    """
    path = shutil.which("sclite")
    if path is None:
        return None
    try:
        banner = subprocess.run([path], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if "SCTK Version" not in banner.stderr + banner.stdout:
        return None
    return path


SCLITE = _sclite()
needs_sclite = pytest.mark.skipif(SCLITE is None, reason="no runnable sclite")


def _read_scp(name):
    pairs = []
    for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            fields = line.split(maxsplit=1)
            pairs.append((fields[0], fields[1] if len(fields) > 1 else ""))
    return pairs


def _tokenize(text, mode):
    if mode == "word":
        return text.split()
    return [SPACE_SYMBOL if ch == " " else ch for ch in text]


def _splet_counts(pairs_ref, pairs_hyp, tokenizer, case, **kwargs):
    state = error_rate_setup(metric_id="m", tokenizer=tokenizer, case=case, **kwargs)
    total = dict(errors=0, hits=0, substitutions=0, deletions=0, insertions=0)
    suffix = dict(
        errors="m_errors",
        hits="m_hit",
        substitutions="m_sub",
        deletions="m_del",
        insertions="m_ins",
    )
    for (_, ref), (_, hyp) in zip(pairs_ref, pairs_hyp):
        result = error_rate_metric(state, hyp, ref)
        for key, reported in suffix.items():
            total[key] += result[reported]
    return total


def _cases():
    """Yield the golden cases, marking the ones SPLET cannot run yet."""
    for name, case in GOLDEN.items():
        if name.startswith("_"):
            continue
        tokenizer = _tokenizer_for(case)
        marks = []
        if tokenizer not in TOKENIZER_CHOICES:
            marks = [
                pytest.mark.skip(reason=f"no '{tokenizer}' tokenizer in SPLET yet")
            ]
        yield pytest.param(name, case, marks=marks, id=name)


@pytest.mark.parametrize("name, case", list(_cases()))
def test_matches_the_sclite_goldens(name, case):
    """Every count, against sclite's own output for the same fixtures."""
    counts = _splet_counts(
        _read_scp(case["ref"]),
        _read_scp(case["hyp"]),
        _tokenizer_for(case),
        "sensitive" if "-s" in case["sclite_flags"] else "fold",
    )
    assert counts == {
        key: case["counts"][key] for key in counts
    }, f"{name}: SPLET {counts} != sclite {case['counts']}"


def test_the_goldens_cover_both_tokenizations_and_both_case_policies():
    """A golden file that lost a case would weaken every test above silently."""
    cases = [case for name, case in GOLDEN.items() if not name.startswith("_")]
    assert {case["tokenization"] for case in cases} >= {"word", "char"}
    assert any("-s" in case["sclite_flags"] for case in cases)
    assert any(not case["sclite_flags"] for case in cases)


def test_the_goldens_record_the_sctk_that_made_them():
    assert "SCTK Version" in GOLDEN["_sctk_version"]


def _run_sclite(utterances, flags=()):
    """Score (ref_tokens, hyp_tokens) utterances and return pooled counts.

    One trn line per utterance, which is what make_golden.py writes and
    what sclite scores: it aligns each utterance separately, so feeding
    it one concatenated line is a different question with a different
    answer.
    """
    wanted = {
        "Percent Total Error": "errors",
        "Percent Correct": "hits",
        "Percent Substitution": "substitutions",
        "Percent Deletions": "deletions",
        "Percent Insertions": "insertions",
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for path, index in (("ref.trn", 0), ("hyp.trn", 1)):
            # sclite's -i rm reads the speaker as the id up to the first
            # "-" or "_", so a bare utterance id needs a speaker prefix.
            (tmp / path).write_text(
                "".join(
                    f"{' '.join(utterance[index])} (spk-u{number})\n"
                    for number, utterance in enumerate(utterances)
                ),
                encoding="utf-8",
            )
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
            timeout=120,
        )
        report = (tmp / "hyp.trn.dtl").read_text(encoding="utf-8")
    counts = {}
    for line in report.splitlines():
        for label, key in wanted.items():
            if line.startswith(label):
                match = re.search(r"\(\s*(\d+)\)", line)
                if match:
                    counts[key] = int(match.group(1))
    return counts


@needs_sclite
def test_the_goldens_still_describe_this_sclite():
    """Regenerating would change the committed file; say so here instead."""
    for name, case in GOLDEN.items():
        if name.startswith("_"):
            continue
        ref = _read_scp(case["ref"])
        hyp = _read_scp(case["hyp"])
        mode = case["tokenization"]
        utterances = [
            (_tokenize(ref_text, mode), _tokenize(hyp_text, mode))
            for (_, ref_text), (_, hyp_text) in zip(ref, hyp)
        ]
        got = _run_sclite(utterances, case["sclite_flags"])
        want = {key: case["counts"][key] for key in got}
        assert got == want, (
            f"{name}: this sclite reports {got}, the goldens say {want}. "
            "Rerun test_utils/splet/make_golden.py and commit the diff."
        )


@needs_sclite
@pytest.mark.parametrize("case", ["fold", "sensitive"])
def test_fuzz_against_the_binary(case):
    """Random pairs, every count, both case policies."""
    rng = random.Random(0 if case == "fold" else 1)
    vocabulary = "the cat sat on a mat dog ran fast Cat Dog".split()
    flags = ["-s"] if case == "sensitive" else []
    for _ in range(100):
        ref = [rng.choice(vocabulary) for _ in range(rng.randint(1, 12))]
        hyp = [rng.choice(vocabulary) for _ in range(rng.randint(1, 12))]
        state = error_rate_setup(metric_id="m", case=case)
        result = error_rate_metric(state, " ".join(hyp), " ".join(ref))
        got = {
            "errors": result["m_errors"],
            "hits": result["m_hit"],
            "substitutions": result["m_sub"],
            "deletions": result["m_del"],
            "insertions": result["m_ins"],
        }
        want = _run_sclite([(ref, hyp)], flags)
        assert got == want, f"ref={ref} hyp={hyp} case={case}"
