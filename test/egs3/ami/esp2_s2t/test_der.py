"""Recipe-level tests for the utterance-group DER metric.

The scorer is in espnet3/systems/esp2_s2t/metrics/der.py. These tests check the
scores that it gives on this recipe's corpus.
"""

import logging

import ami_sot_paths
import pytest

# der.py scores with pyannote.metrics and imports cpwer.py, which scores with
# meeteval. In an environment without them, these lines skip the module, so
# collection does not fail.
pytest.importorskip("pyannote.metrics")
pytest.importorskip("meeteval")

# The recorded decode that the regression test scores again. It is only present
# when AMI_SOT_REFERENCE_DECODE points to one.
_RECORDED_TEXT_SOT = (
    (ami_sot_paths.REFERENCE_DECODE / "text_sot")
    if ami_sot_paths.REFERENCE_DECODE
    else None
)


from espnet3.systems.esp2_s2t.metrics import der as der_mod  # noqa: E402

SEP = "????"


def test_segments_from_sot_pairs_timestamps_within_a_speaker_block():
    sep = SEP
    text = f"<|0.00|> hello<|1.20|> {sep} <|2.00|> world<|3.50|>"
    assert der_mod._parse_sot_segments(text, SEP) == [
        (0, 0.0, 1.2),
        (1, 2.0, 3.5),
    ]


def test_segments_from_sot_drops_an_unclosed_final_segment():
    """_parse_sot_segments drops a final segment that has no closing timestamp.

    The corpus has no closing timestamp for a segment that the cut truncated.
    """
    sep = SEP
    text = f"<|0.00|> hello<|1.20|> {sep} <|2.00|> world"
    assert der_mod._parse_sot_segments(text, SEP) == [(0, 0.0, 1.2)]


def test_segments_from_sot_drops_a_zero_length_segment():
    assert der_mod._parse_sot_segments("<|1.00|> x<|1.00|>", SEP) == []


def test_call_warns_once_when_a_reference_group_has_no_segments(tmp_path, caplog):
    """The DER metric warns once when a reference group has no segments.

    A group whose reference has no usable timestamp pair is left out of DER. The
    metric counts these groups and logs a warning, so the user can see the drop.
    """
    ref = tmp_path / "ref.scp"
    hyp = tmp_path / "hyp.scp"
    ref.write_text(
        "u1 no timestamps here\nu2 <|0.00|> hello<|1.20|>\n", encoding="utf-8"
    )
    hyp.write_text(
        "u1 <|0.00|> hello<|1.20|>\nu2 <|0.00|> hello<|1.20|>\n", encoding="utf-8"
    )
    with caplog.at_level(logging.WARNING):
        result = der_mod.UtteranceGroupDER(SEP)(
            {"ref": ref, "hyp": hyp}, "test", tmp_path
        )
    # u2's reference and hypothesis are identical, so the scored groups have a
    # perfect score. u1 is the group that was dropped.
    assert result["ug_DER"] == 0.0
    assert "1 utterance group(s) had no reference segments" in caplog.text


def test_call_does_not_warn_when_every_reference_group_has_segments(tmp_path, caplog):
    ref = tmp_path / "ref.scp"
    hyp = tmp_path / "hyp.scp"
    ref.write_text("u1 <|0.00|> hello<|1.20|>\n", encoding="utf-8")
    hyp.write_text("u1 <|0.00|> hello<|1.20|>\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        der_mod.UtteranceGroupDER(SEP)({"ref": ref, "hyp": hyp}, "test", tmp_path)
    assert "had no reference segments" not in caplog.text


@pytest.mark.skipif(
    _RECORDED_TEXT_SOT is None or not _RECORDED_TEXT_SOT.is_file(),
    reason="the recorded 'text_sot' or SCTK is not present",
)
@ami_sot_paths.needs_corpus
@ami_sot_paths.needs_reference_decode
@pytest.mark.execution_timeout(60.0)
def test_der_reproduces_the_recorded_full_test_set_score(tmp_path):
    ref = tmp_path / "ref.scp"
    hyp = tmp_path / "hyp.scp"

    # The recorded decode spells the separator "<sc>". The metric splits on the
    # checkpoint's own symbol, so the test converts the spelling.
    ami_sot_paths.write_test_reference(ref)
    hyp.write_text(
        _RECORDED_TEXT_SOT.read_text(encoding="utf-8").replace("<sc>", SEP),
        encoding="utf-8",
    )
    result = der_mod.UtteranceGroupDER(SEP, clean_types=["whisper_en"])(
        {"ref": ref, "hyp": hyp}, "test", tmp_path
    )
    assert result["ug_DER"] == 8.57


def test_segments_from_sot_follows_the_symbol_it_is_given():
    """_parse_sot_segments splits the text on the symbol passed to it."""
    segments = der_mod._parse_sot_segments(
        "<|0.00|> a<|1.20|> @@ <|2.00|> b<|3.50|>", "@@"
    )
    assert segments == [(0, 0.0, 1.2), (1, 2.0, 3.5)]
