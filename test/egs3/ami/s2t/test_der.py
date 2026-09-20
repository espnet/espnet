"""Tests for egs3/ami/s2t/src/metrics/der.py."""

import importlib.util
import logging
import stat
import sys
from pathlib import Path

import ami_sot_paths
import pytest

_MD_EVAL = ami_sot_paths.REPO / "tools" / "sctk" / "bin" / "md-eval.pl"

# der.py imports cpwer.py (for split_speakers), which needs scipy and
# editdistance. Guarded here to skip collection rather than fail it on an
# environment without them, matching test_cpwer.py's own guards for the same
# transitive dependency.
pytest.importorskip("scipy")
pytest.importorskip("editdistance")


def _load():
    sys.path.insert(0, str(ami_sot_paths.REPO))
    spec = importlib.util.spec_from_file_location(
        "ami_s2t_der", ami_sot_paths.RECIPE / "src" / "metrics" / "der.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


der_mod = _load()


def test_segments_from_sot_pairs_timestamps_within_a_speaker_block():
    sep = der_mod.SPEAKER_CHANGE_SYMBOL
    text = f"<|0.00|> hello<|1.20|> {sep} <|2.00|> world<|3.50|>"
    assert der_mod.segments_from_sot(text) == [
        (0, 0.0, 1.2),
        (1, 2.0, 3.5),
    ]


def test_segments_from_sot_drops_an_unclosed_final_segment():
    """prepare_sot omits the closing timestamp of a cut-truncated segment."""
    sep = der_mod.SPEAKER_CHANGE_SYMBOL
    text = f"<|0.00|> hello<|1.20|> {sep} <|2.00|> world"
    assert der_mod.segments_from_sot(text) == [(0, 0.0, 1.2)]


def test_segments_from_sot_drops_a_zero_length_segment():
    assert der_mod.segments_from_sot("<|1.00|> x<|1.00|>") == []


def test_write_rttm_uses_the_sctk_field_order(tmp_path):
    path = tmp_path / "out.rttm"
    der_mod.write_rttm([("u1", 0, 0.0, 1.5)], path)
    fields = path.read_text(encoding="utf-8").strip().split()
    assert fields[0] == "SPEAKER"
    assert fields[1] == "u1"
    assert fields[2] == "1"
    assert fields[3] == "0.000"
    assert fields[4] == "1.500"
    assert fields[5] == "<NA>" and fields[6] == "<NA>"
    assert fields[7] == "spk0"


def _fake_md_eval(tmp_path: Path, body: str) -> str:
    script = tmp_path / "md-eval.pl"
    script.write_text("#!/bin/sh\n" + body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


_DER_LINE = (
    "echo ' OVERALL SPEAKER DIARIZATION ERROR = 12.34 percent "
    "of scored speaker time  \\`(ALL)'\n"
)


def test_run_md_eval_parses_the_overall_line(tmp_path):
    md = _fake_md_eval(tmp_path, _DER_LINE + "exit 0\n")
    assert der_mod.run_md_eval(md, "ref", "hyp", 0.25) == 12.34


def test_run_md_eval_rejects_a_non_zero_exit(tmp_path):
    """md-eval can fail after printing, so a DER from that run is not valid."""
    md = _fake_md_eval(tmp_path, _DER_LINE + "exit 3\n")
    with pytest.raises(SystemExit):
        der_mod.run_md_eval(md, "ref", "hyp", 0.25)


def test_run_md_eval_rejects_output_without_a_der_line(tmp_path):
    md = _fake_md_eval(tmp_path, "echo 'nothing useful'\nexit 0\n")
    with pytest.raises(SystemExit):
        der_mod.run_md_eval(md, "ref", "hyp", 0.25)


def test_call_warns_once_when_a_reference_group_has_no_segments(tmp_path, caplog):
    """Call warns once when a reference group has no segments.

    A group whose reference carries no usable timestamp pair is dropped from DER with no
    counter and no log line today.

    Count and warn instead, so the drop is visible rather than silent.
    """
    ref = tmp_path / "ref_sot.scp"
    hyp = tmp_path / "hyp_sot.scp"
    ref.write_text(
        "u1 no timestamps here\nu2 <|0.00|> hello<|1.20|>\n", encoding="utf-8"
    )
    hyp.write_text(
        "u1 <|0.00|> hello<|1.20|>\nu2 <|0.00|> hello<|1.20|>\n", encoding="utf-8"
    )
    md = _fake_md_eval(tmp_path, _DER_LINE + "exit 0\n")
    with caplog.at_level(logging.WARNING):
        result = der_mod.DER(md_eval=md)({"ref": ref, "hyp": hyp}, "test", tmp_path)
    assert result["DER"] == 12.34
    assert "1 utterance group(s) had no reference segments" in caplog.text


def test_call_does_not_warn_when_every_reference_group_has_segments(tmp_path, caplog):
    ref = tmp_path / "ref_sot.scp"
    hyp = tmp_path / "hyp_sot.scp"
    ref.write_text("u1 <|0.00|> hello<|1.20|>\n", encoding="utf-8")
    hyp.write_text("u1 <|0.00|> hello<|1.20|>\n", encoding="utf-8")
    md = _fake_md_eval(tmp_path, _DER_LINE + "exit 0\n")
    with caplog.at_level(logging.WARNING):
        der_mod.DER(md_eval=md)({"ref": ref, "hyp": hyp}, "test", tmp_path)
    assert "had no reference segments" not in caplog.text


@pytest.mark.skipif(not _MD_EVAL.is_file(), reason="SCTK is not present")
def test_find_md_eval_returns_this_checkouts_md_eval():
    assert der_mod.find_md_eval() == str(_MD_EVAL)


def test_find_md_eval_does_not_climb_past_the_repo_root(tmp_path):
    """A checkout without SCTK must not silently pick up one from outside it.

    ``tools/sctk/bin/md-eval.pl`` is planted above a ``.git`` boundary, and the search
    starts from a file below that boundary. Unbounded upward search would find the outer
    md-eval.pl and return it; the DER this recipe reports would then depend on whichever
    checkout happens to sit above it on the filesystem, which is exactly the
    unreproducible outcome ``find_md_eval``'s docstring promises cannot happen.
    """
    outer_md_eval = tmp_path / "tools" / "sctk" / "bin" / "md-eval.pl"
    outer_md_eval.parent.mkdir(parents=True)
    outer_md_eval.write_text("#!/bin/sh\necho should not be used\n")

    inner = tmp_path / "inner"
    (inner / ".git").mkdir(parents=True)
    fake_module = inner / "egs3" / "ami" / "s2t" / "src" / "metrics" / "der.py"
    fake_module.parent.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="inner"):
        der_mod._find_md_eval_from(fake_module)


# needs_reference_decode only checks that the decode directory is there; the
# DER regression reads the timestamped "text_sot" inside it, and scores it with
# md-eval.pl, which a fresh checkout has not built yet.
_RECORDED_TEXT_SOT = (
    (ami_sot_paths.REFERENCE_DECODE / "text_sot")
    if ami_sot_paths.REFERENCE_DECODE
    else None
)


@pytest.mark.skipif(
    _RECORDED_TEXT_SOT is None
    or not _RECORDED_TEXT_SOT.is_file()
    or not _MD_EVAL.is_file(),
    reason="the recorded 'text_sot' or SCTK is not present",
)
@ami_sot_paths.needs_corpus
@ami_sot_paths.needs_reference_decode
@pytest.mark.execution_timeout(60.0)
def test_der_reproduces_the_recorded_full_test_set_score(tmp_path):
    ref = tmp_path / "ref_sot.scp"
    hyp = tmp_path / "hyp_sot.scp"

    # Both the recorded decode and a corpus prepared before this recipe spell
    # the separator "<sc>"; the metric splits on the checkpoint's own symbol.
    def _as_configured(text):
        return text.replace("<sc>", der_mod.SPEAKER_CHANGE_SYMBOL)

    ref.write_text(
        _as_configured(ami_sot_paths.TEST_TEXT.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    hyp.write_text(
        _as_configured(_RECORDED_TEXT_SOT.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    result = der_mod.DER()({"ref": ref, "hyp": hyp}, "test", tmp_path)
    assert result["DER"] == 8.57


def test_segments_from_sot_follows_the_configured_symbol(monkeypatch):
    """segments_from_sot must honour the configured symbol, not a literal.

    Only checking the separator constant would pass without proving
    segments_from_sot actually uses it. Without the fix, "@@" is not recognized as a
    separator, so the whole text stays one block and this assertion fails.
    """
    monkeypatch.setenv("AMI_SOT_SPEAKER_CHANGE_SYMBOL", "@@")
    reloaded = _load()
    segments = reloaded.segments_from_sot("<|0.00|> a<|1.20|> @@ <|2.00|> b<|3.50|>")
    assert segments == [(0, 0.0, 1.2), (1, 2.0, 3.5)]
