"""Tests for the ASR error-rate metrics, which are adapters over SPLET.

The scoring itself is tested in test/splet/, against sclite. What is tested
here is the part that is ESPnet's: the SCP plumbing, the keys returned into
metrics.json, the alignment file, and the two behaviours that changed when
these stopped calling jiwer.
"""

from pathlib import Path

import pytest
import sentencepiece as spm

from espnet3.systems.asr.metrics.base_error_rate import normalize_config
from espnet3.systems.asr.metrics.cer import CER
from espnet3.systems.asr.metrics.ter import TER
from espnet3.systems.asr.metrics.wer import WER


def _build_inputs(tmp_path: Path, ref_lines, hyp_lines) -> dict[str, Path]:
    ref_path = tmp_path / "ref.scp"
    hyp_path = tmp_path / "hyp.scp"
    ref_path.write_text("\n".join(ref_lines), encoding="utf-8")
    hyp_path.write_text("\n".join(hyp_lines), encoding="utf-8")
    return {"ref": ref_path, "hyp": hyp_path}


def test_wer_writes_alignment_and_score(tmp_path: Path):
    metric = WER()
    data = _build_inputs(tmp_path, ["utt1 hello world"], ["utt1 hello word"])

    result = metric(data, "test-clean", tmp_path)

    assert result["WER"] == 50.0
    assert result["WER_errors"] == 1
    assert result["WER_ref_len"] == 2
    assert result["WER_sub"] == 1
    assert result["WER_hit"] == 1
    alignment_path = tmp_path / "test-clean" / "wer_alignment"
    assert alignment_path.exists()
    assert "utt1" in alignment_path.read_text()


def test_cer_writes_alignment_and_score(tmp_path: Path):
    metric = CER()
    data = _build_inputs(tmp_path, ["utt1 abc"], ["utt1 axc"])

    result = metric(data, "test-other", tmp_path)

    assert result["CER"] == 33.33
    assert result["CER_ref_len"] == 3
    assert (tmp_path / "test-other" / "cer_alignment").exists()


def test_cer_counts_spaces(tmp_path: Path):
    """As jiwer.cer did, and as CharTokenizer's <space> token does."""
    metric = CER()
    data = _build_inputs(tmp_path, ["utt1 ab cd"], ["utt1 ab cd"])
    assert metric(data, "t", tmp_path)["CER_ref_len"] == 5


def test_the_corpus_rate_is_pooled_not_averaged(tmp_path: Path):
    """sum(errors) / sum(ref_len), which is what SCTK reports.

    One error in a one-word utterance beside a clean eight-word one is 1/9,
    not the mean of 100% and 0%.
    """
    metric = WER()
    data = _build_inputs(
        tmp_path,
        ["utt1 alpha", "utt2 one two three four five six seven eight"],
        ["utt1 beta", "utt2 one two three four five six seven eight"],
    )
    result = metric(data, "t", tmp_path)
    assert result["WER_errors"] == 1
    assert result["WER_ref_len"] == 9
    assert result["WER"] == round(100 / 9, 2)


def test_wer_rejects_unaligned_utt_ids(tmp_path: Path):
    metric = WER()
    data = _build_inputs(tmp_path, ["utt1 hello"], ["utt2 hello"])
    with pytest.raises(AssertionError, match="UID mismatch"):
        metric(data, "test-clean", tmp_path)


def test_case_is_folded_by_default(tmp_path: Path):
    """sclite compares case-insensitively unless given -s, so this does too.

    jiwer was case-sensitive, so this is a deliberate change: it is what
    every egs2 recipe scores with.
    """
    metric = WER()
    data = _build_inputs(tmp_path, ["utt1 Hello World"], ["utt1 hello world"])
    assert metric(data, "t", tmp_path)["WER"] == 0.0


def test_case_sensitivity_is_available(tmp_path: Path):
    """sclite's -s, and what the previous jiwer implementation did."""
    metric = WER(case="sensitive")
    data = _build_inputs(tmp_path, ["utt1 Hello World"], ["utt1 hello world"])
    assert metric(data, "t", tmp_path)["WER"] == 100.0


def test_an_empty_hypothesis_is_all_deletions(tmp_path: Path):
    """No placeholder substitution.

    The previous code put "." on both sides of an empty string, so an
    undecodable utterance scored one substitution instead of a deletion per
    reference word. #6735 removed the same placeholder from the BLEU metric.
    """
    metric = WER()
    data = _build_inputs(tmp_path, ["utt1 one two three"], ["utt1"])
    result = metric(data, "t", tmp_path)
    assert result["WER_del"] == 3
    assert result["WER_sub"] == 0
    assert result["WER"] == 100.0


def test_unit_costs_are_available(tmp_path: Path):
    """What jiwer computed: a minimum edit distance.

    A transposition is two substitutions under unit costs and a deletion
    plus an insertion under sclite's, which is the difference the S/D/I
    breakdown exposes.
    """
    data = _build_inputs(tmp_path, ["utt1 the cat sat"], ["utt1 the sat cat"])
    sclite = WER()(data, "a", tmp_path)
    unit = WER(costs="unit")(data, "b", tmp_path)
    assert sclite["WER_errors"] == unit["WER_errors"] == 2
    assert (sclite["WER_sub"], sclite["WER_del"], sclite["WER_ins"]) == (0, 1, 1)
    assert (unit["WER_sub"], unit["WER_del"], unit["WER_ins"]) == (2, 0, 0)


def test_whisper_cleaners_are_wired_through(tmp_path: Path):
    """The only cleaners any egs2 ASR-family recipe sets.

    Scored end to end rather than by inspecting the config: whisper_en
    rewrites "Mr." to "mister", so cleaning both sides makes these agree.
    """
    data = _build_inputs(tmp_path, ["utt1 Mr. Smith"], ["utt1 mister smith"])
    assert WER()(data, "raw", tmp_path)["WER"] == 50.0
    assert WER(clean_types=["whisper_en"])(data, "clean", tmp_path)["WER"] == 0.0


def test_no_cleaner_is_the_identity():
    """clean_types is null in every shipped config."""
    assert normalize_config(None) is None


def test_an_unimplemented_cleaner_is_refused():
    """Skipping it would report a score under a normalization that never ran."""
    with pytest.raises(NotImplementedError, match="tacotron"):
        normalize_config(["tacotron"])


@pytest.fixture
def tiny_bpemodel(tmp_path: Path) -> str:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "\n".join(["hello world", "the cat sat", "a dog ran", "hello there"] * 8),
        encoding="utf-8",
    )
    prefix = tmp_path / "bpe"
    spm.SentencePieceTrainer.Train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=40,
        model_type="bpe",
        character_coverage=1.0,
    )
    return f"{prefix}.model"


@pytest.mark.execution_timeout(30)
def test_ter_zero_when_identical(tmp_path: Path, tiny_bpemodel: str):
    metric = TER(bpemodel=tiny_bpemodel)
    data = _build_inputs(tmp_path, ["utt1 hello world"], ["utt1 hello world"])
    result = metric(data, "test-clean", tmp_path)
    assert result["TER"] == 0.0
    assert result["TER_errors"] == 0
    assert (tmp_path / "test-clean" / "ter_alignment").exists()


@pytest.mark.execution_timeout(30)
def test_ter_positive_when_different(tmp_path: Path, tiny_bpemodel: str):
    metric = TER(bpemodel=tiny_bpemodel)
    data = _build_inputs(tmp_path, ["utt1 hello world"], ["utt1 the cat sat"])
    result = metric(data, "test-clean", tmp_path)
    assert result["TER"] > 0.0


@pytest.mark.execution_timeout(30)
def test_ter_counts_subword_tokens(tmp_path: Path, tiny_bpemodel: str):
    """Not words: the denominator is the SentencePiece tokenization.

    Checked against espnet2's own tokenizer, which is what asr.sh scores the
    bpe path with, rather than against SPLET's -- the point is that the two
    agree.
    """
    from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer

    expected = SentencepiecesTokenizer(tiny_bpemodel).text2tokens("hello world")

    metric = TER(bpemodel=tiny_bpemodel)
    data = _build_inputs(tmp_path, ["utt1 hello world"], ["utt1 hello world"])
    result = metric(data, "t", tmp_path)
    assert result["TER_ref_len"] == len(expected) > 2
