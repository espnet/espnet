"""Test egs2/fleurs_cs_lid/asr1/local/score_lidseq.py (ASR language sets)."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "egs2/fleurs_cs_lid/asr1/local/score_lidseq.py"
)
_spec = importlib.util.spec_from_file_location("fleurs_cs_lid_score_lidseq", SCRIPT)
score_lidseq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score_lidseq)


def write(path, text):
    """Write ``text`` to ``path`` and return the path as a string."""
    path.write_text(text, encoding="utf-8")
    return str(path)


def run(tmp_path, refs, hyps, seen="s1 eng ara\ns2 eng\ns3 ara cmn eng\n"):
    """Score lists of utt2langs / decoded-text contents; return the results."""
    argv = ["--seen_utt2langs", write(tmp_path / "seen", seen)]
    argv += ["--out", str(tmp_path / "out/lang_set_results.tsv"), "--ref"]
    argv += [write(tmp_path / f"ref{i}", r) for i, r in enumerate(refs)]
    argv += ["--hyp"] + [write(tmp_path / f"hyp{i}", h) for i, h in enumerate(hyps)]
    score_lidseq.main(argv)
    return (tmp_path / "out/lang_set_results.tsv").read_text()


@pytest.mark.parametrize(
    "hyp, exact",
    [
        ("<ara> <eng>", True),
        ("<eng> <ara>", True),  # order is ignored
        ("<eng> <eng>", False),  # a repeated token is not the set {ara, eng}
        ("<ara> <eng> <eng>", False),
        ("<ara> <unk>", False),
        ("", False),  # empty hypothesis
    ],
)
def test_is_exact(hyp, exact):
    """Exact means the same languages in any order, each once."""
    hyp = tuple(t.strip("<>") for t in hyp.split())
    assert score_lidseq.is_exact(hyp, ("ara", "eng")) == exact


def test_repeated_token_for_one_language(tmp_path):
    """Hypothesis <eng> <eng> for reference eng is wrong."""
    assert "all\t1\t0.00" in run(tmp_path, ["u1 eng\n"], ["u1 <eng> <eng>\n"])


def test_groups(tmp_path):
    """Seen / unseen only for two-language refs; "eng ara" in seen = "ara eng"."""
    ref = "u1 eng\nu2 ara eng\nu3 cmn eng\nu4 ara cmn eng\n"
    hyp = "u1 <unk>\nu2 <eng> <ara>\nu3\nu4 <eng> <cmn> <ara>\n"
    assert run(tmp_path, [ref], [hyp]) == (
        "group\tn\taccuracy\n"
        "all\t4\t50.00\n1-language\t1\t0.00\n2-language\t2\t50.00\n"
        "seen\t1\t100.00\nunseen\t1\t0.00\n3-language\t1\t100.00\n"
    )


def test_several_files_are_one_set(tmp_path):
    """Several --ref/--hyp pairs are scored together."""
    results = run(tmp_path, ["u1 eng\n", "u2 ara eng\n"], ["u1 <eng>\n", "u2 <ara>\n"])
    assert "all\t2\t50.00" in results


@pytest.mark.parametrize(
    "refs, hyps, message",
    [
        (["u1 eng\n\n"], ["u1 <eng>\n"], "ref0:2: expected"),
        (["u1 eng\n"], ["\nu1 <eng>\n"], "hyp0:1: expected"),
        (["u1 eng\nu2\n"], ["u1 <eng>\nu2 <eng>\n"], "ref0:2: expected"),
        (["u1 eng\nu2 eng eng\n"], ["u1 <eng>\nu2 <eng>\n"], "ref0:2: repeated"),
        (["u1 eng\nu1 ara\n"], ["u1 <eng>\n"], "ref0:2: duplicate"),
        (["u1 eng\n", "u1 ara\n"], ["u1 <eng>\n", "u2 <ara>\n"], "duplicate.*ref1"),
        (["u1 eng\n"], ["u1 <eng>\nu1 <ara>\n"], "hyp0:2: duplicate"),
        (
            ["u1 eng\n", "u2 ara\n"],
            ["u1 <eng>\n", "u1 <ara>\nu2 <ara>\n"],
            "duplicate.*hyp1",
        ),
        (["u1 eng\n"], ["u1 <eng>\nu2 <eng>\n"], "1 hypotheses without reference"),
        (["u1 eng\nu2 eng\n"], ["u1 <eng>\n"], "1 references without hypothesis"),
    ],
)
def test_input_errors(tmp_path, refs, hyps, message):
    """Malformed, duplicate or unmatched lines are errors."""
    with pytest.raises(ValueError, match=message):
        run(tmp_path, refs, hyps)


def test_blank_line_in_seen_file(tmp_path):
    """A blank --seen_utt2langs line is an error with file:line."""
    with pytest.raises(ValueError, match="seen:2: expected"):
        run(tmp_path, ["u1 eng\n"], ["u1 <eng>\n"], seen="s1 eng\n\n")
