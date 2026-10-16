"""Tests for the esp2_slu intent accuracy metric."""

from pathlib import Path

from espnet3.systems.esp2_slu.metrics.intent_accuracy import IntentAccuracy


def test_intent_accuracy_scores_and_reports_errors(tmp_path: Path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "ref_intent.scp").write_text(
        "0 news_query\n1 audio_volume_mute\n", encoding="utf-8"
    )
    (test_dir / "hyp_intent.scp").write_text(
        "0 news_query\n1 calendar_query\n", encoding="utf-8"
    )

    metric = IntentAccuracy()
    result = metric(
        {
            "ref_intent": test_dir / "ref_intent.scp",
            "hyp_intent": test_dir / "hyp_intent.scp",
        },
        "test",
        tmp_path,
    )

    assert result == {"IntentAccuracy": 50.0}
    errors = (test_dir / "intent_errors").read_text(encoding="utf-8").splitlines()
    assert errors[1] == "1\taudio_volume_mute\tcalendar_query"


def _write_hypotheses(source_dir: Path, split: str, transcripts) -> None:
    """Write the SCP the infer stage dumps for one split."""
    scp_path = source_dir / split / "hyp_transcript.scp"
    scp_path.parent.mkdir(parents=True, exist_ok=True)
    scp_path.write_text(
        "".join(f"{i} {text}\n" for i, text in enumerate(transcripts)),
        encoding="utf-8",
    )
