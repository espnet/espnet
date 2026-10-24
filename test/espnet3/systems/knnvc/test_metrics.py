"""Tests for the kNN-VC intelligibility metric."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from espnet3.contract.kinds.audio import Audio
from espnet3.contract.metrics import MetricContractError
from espnet3.systems.knnvc.metrics.intelligibility import ASRIntelligibility

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_scores_transcriptions_against_refs     | WER/CER reflect the actual   |
# |                                             | edits, and alignment is kept.|
# | test_perfect_transcription_scores_zero      | Identical text scores 0.     |
# | test_rejects_wrong_sample_rate              | A non-16k WAV raises.        |
# | test_prebuilt_asr_is_used_as_is             | A built ASR is not replaced  |
# |                                             | by a download.               |
# | test_chunks_score_like_one_batch            | Summed counts give the same  |
# |                          | WER/CER whether update sees one or many batches. |
# | test_update_accepts_decoded_audio           | Audio and arrays score like  |
# |                          | paths; a wrong-rate Audio is refused.            |
# | test_state_roundtrip                        | reset/state/merge/load give  |
# |                                             | the same compute().          |


class FakeASR:
    """Return a canned hypothesis per call, in order."""

    def __init__(self, hypotheses):
        self.hypotheses = list(hypotheses)
        self.calls = 0

    def __call__(self, speech):
        text = self.hypotheses[self.calls]
        self.calls += 1
        return [(text, None, None, None)]


def _write_case(tmp_path: Path, refs, rate=16000):
    """Write a wav.scp / ref.scp pair with one silent wav per utterance."""
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    wav_lines, ref_lines = [], []
    for i, ref in enumerate(refs):
        path = wav_dir / f"utt{i}.wav"
        sf.write(path, np.zeros(rate // 2, dtype=np.float32), rate)
        wav_lines.append(f"utt{i} {path}")
        ref_lines.append(f"utt{i} {ref}")
    (tmp_path / "wav.scp").write_text("\n".join(wav_lines) + "\n")
    (tmp_path / "ref.scp").write_text("\n".join(ref_lines) + "\n")
    return {"wav": tmp_path / "wav.scp", "ref": tmp_path / "ref.scp"}


def test_scores_transcriptions_against_refs(tmp_path):
    data = _write_case(tmp_path, ["the cat sat", "a dog ran"])
    metric = ASRIntelligibility(asr=FakeASR(["the cat sat", "a dog walked"]))

    scores = metric(data, "test", tmp_path / "out")

    # One substituted word out of six: "ran" -> "walked".
    assert scores["ASR_WER"] == pytest.approx(100 / 6, abs=0.01)
    # Five character edits over 20 reference characters, so CER exceeds WER
    # here; the two are scored independently.
    assert scores["ASR_CER"] == pytest.approx(25.0, abs=0.01)
    assert (tmp_path / "out" / "test" / "asr_wer_alignment").is_file()


def test_perfect_transcription_scores_zero(tmp_path):
    data = _write_case(tmp_path, ["the cat sat", "a dog ran"])
    metric = ASRIntelligibility(asr=FakeASR(["the cat sat", "a dog ran"]))

    scores = metric(data, "test", tmp_path / "out")

    assert scores == {"ASR_WER": 0.0, "ASR_CER": 0.0}


def test_rejects_wrong_sample_rate(tmp_path):
    data = _write_case(tmp_path, ["hello"], rate=8000)
    metric = ASRIntelligibility(asr=FakeASR(["hello"]))

    with pytest.raises(ValueError, match="expects 16000 Hz"):
        metric(data, "test", tmp_path / "out")


def test_prebuilt_asr_is_used_as_is(tmp_path):
    """A built ASR must be used directly, never replaced by a download."""
    asr = FakeASR(["hello"])
    metric = ASRIntelligibility(asr=asr)

    assert metric._build_asr() is asr

    data = _write_case(tmp_path, ["hello"])
    metric(data, "test", tmp_path / "out")
    assert asr.calls == 1


def test_chunks_score_like_one_batch(tmp_path):
    refs = ["the cat sat", "a dog ran", "one two three", "four"]
    hyps = ["the cat sat", "a dog walked", "one two", "four five"]
    data = _write_case(tmp_path, refs)

    whole = ASRIntelligibility(asr=FakeASR(hyps))
    scores = whole(data, "test")

    chunked = ASRIntelligibility(asr=FakeASR(hyps))
    chunked.chunk_size = 1
    assert chunked(data, "test") == scores
    assert chunked.state() == whole.state()

    # The declared inputs are checked before update runs.
    with pytest.raises(MetricContractError, match="hyp"):
        whole.update(ref=["a"], hyp=["a"])
    with pytest.raises(ValueError, match="nothing to score"):
        ASRIntelligibility(asr=FakeASR([])).compute()


def test_update_accepts_decoded_audio():
    metric = ASRIntelligibility(asr=FakeASR(["the cat sat", "a dog ran"]))
    samples = np.zeros(8000, dtype=np.float32)
    metric.update(ref=["the cat sat"], wav=[Audio(samples, 16000)])
    metric.update(ref=["a dog ran"], wav=[samples])
    assert metric.compute() == {"ASR_WER": 0.0, "ASR_CER": 0.0}

    with pytest.raises(ValueError, match="expects 16000 Hz"):
        metric.update(ref=["hello"], wav=[Audio(samples, 8000)])


def test_state_roundtrip(tmp_path):
    data = _write_case(tmp_path, ["the cat sat", "a dog ran"])
    metric = ASRIntelligibility(asr=FakeASR(["the cat sat", "a dog walked"]))
    scores = metric(data, "test")
    state = metric.state()
    assert state["word_substitutions"] == 1 and state["word_hits"] == 5

    metric.reset()
    assert all(value == 0 for value in metric.state().values())
    metric.load_state(metric.merge_states([state, state]))
    assert metric.compute() == scores
