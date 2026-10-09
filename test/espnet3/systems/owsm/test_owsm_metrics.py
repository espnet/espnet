"""OWSM metrics, which must split one test set across two tasks."""

from __future__ import annotations

from pathlib import Path

import pytest

from espnet3.systems.owsm.metrics import CER, TER, WER
from espnet3.systems.owsm.metrics.tags import strip_tags, tag_pattern

ASR = "<eng><asr><0.00> the quick brown fox<2.50>"
DE = "<eng><st_deu><0.00> der schnelle braune Fuchs<2.50>"

# A stand-in for the recipe's nlsyms(): espnet3 must not import from egs3, and
# the metric takes the inventory from config for the same reason.
NLSYMS = [
    "<na>",
    "<nospeech>",
    "<notimestamps>",
    "<asr>",
    "<eng>",
    "<nan>",
    "<st_deu>",
] + [f"<{t / 100:.2f}>" for t in range(0, 3001, 2)]
PATTERN = tag_pattern(NLSYMS)


def _wer(**kwargs):
    return WER(nlsyms=NLSYMS, **kwargs)


def _cer(**kwargs):
    return CER(nlsyms=NLSYMS, **kwargs)


def _scp(tmp_path: Path, name: str, rows: list[tuple[str, str]]) -> Path:
    path = tmp_path / f"{name}.scp"
    path.write_text("".join(f"{utt} {text}\n" for utt, text in rows), encoding="utf-8")
    return path


def _data(tmp_path, refs, hyps):
    utts = [f"utt{i}" for i in range(len(refs))]
    return {
        "ref": _scp(tmp_path, "ref", list(zip(utts, refs))),
        "hyp": _scp(tmp_path, "hyp", list(zip(utts, hyps))),
    }


def test_markup_is_stripped_before_scoring():
    assert strip_tags(ASR, PATTERN) == "the quick brown fox"
    assert strip_tags("<eng><asr><notimestamps> no times", PATTERN) == "no times"


def test_a_longer_symbol_wins_over_a_prefix_of_it():
    """The inventory nests: <na> is a prefix of <nan>, a language in its own
    right, so a shortest-first alternation would leave "n>" behind."""
    assert strip_tags("<nan> hi", PATTERN) == "hi"


def test_nlsyms_is_required_to_remove_tags():
    """There is no default inventory here -- which symbols exist is the
    recipe's business, exactly as for tokenizer.nlsyms."""
    with pytest.raises(RuntimeError, match="nlsyms is required"):
        WER()

    assert WER(remove_tags=False).pattern is None


def test_tags_are_removed_so_a_perfect_transcription_scores_zero(tmp_path):
    """The hypothesis never carries tags, so a tagged reference cannot match."""
    pytest.importorskip("jiwer")
    data = _data(tmp_path, [ASR], ["the quick brown fox"])

    assert _wer()(data, "dev", tmp_path) == {"WER": 0.0}


def test_remove_tags_false_reproduces_s2t_sh(tmp_path):
    """s2t.sh asks for tag removal and cannot achieve it on OWSM text.

    Kept only so the two numbers can be compared; a perfect transcription
    scores a large WER this way.
    """
    pytest.importorskip("jiwer")
    data = _data(tmp_path, [ASR], ["the quick brown fox"])

    assert WER(remove_tags=False)(data, "dev", tmp_path)["WER"] > 0


def test_every_row_is_scored_including_translations(tmp_path):
    """s2t.sh does not split by task; a translation is scored as a transcript."""
    pytest.importorskip("jiwer")
    asr_dir = tmp_path / "asr_only"
    asr_dir.mkdir()
    # The ASR row is perfect; the translation row is entirely wrong.
    both = _data(tmp_path, [ASR, DE], ["the quick brown fox", "voellig falsch hier"])
    asr_only = _data(asr_dir, [ASR], ["the quick brown fox"])

    metric = _wer()
    assert metric(asr_only, "dev", tmp_path) == {"WER": 0.0}
    # Skipping the ST row would keep this at 0.
    assert metric(both, "dev", tmp_path)["WER"] > 0


def test_cer_counts_characters_not_words(tmp_path):
    pytest.importorskip("jiwer")
    # One wrong letter in a four-word line: small CER, large WER.
    data = _data(tmp_path, [ASR], ["the quick brown fix"])

    cer = _cer()(data, "dev", tmp_path)["CER"]
    wer = _wer()(data, "dev", tmp_path)["WER"]
    assert 0 < cer < wer


def _tiny_bpe(tmp_path):
    """Train a throwaway SentencePiece model, so the test needs no artifact.

    The metric strips tags before tokenizing, so the pieces this has to cover
    are plain words; the vocabulary only needs to be large enough to hold the
    alphabet.
    """
    spm = pytest.importorskip("sentencepiece")
    corpus = tmp_path / "spm_train.txt"
    corpus.write_text(
        "\n".join(
            f"the quick brown fox number {i} jumps over the lazy dog"
            for i in range(200)
        ),
        encoding="utf-8",
    )
    prefix = tmp_path / "bpe"
    spm.SentencePieceTrainer.Train(
        input=str(corpus),
        model_prefix=str(prefix),
        model_type="bpe",
        vocab_size=120,
        character_coverage=1.0,
    )
    return prefix.with_suffix(".model")


def test_ter_counts_bpe_pieces(tmp_path):
    """s2t.sh's ter is a token error rate over subwords, not sacreBLEU's TER."""
    pytest.importorskip("jiwer")
    model = _tiny_bpe(tmp_path)
    data = _data(tmp_path, [ASR], ["the quick brown fix"])

    ter = TER(bpemodel=model, nlsyms=NLSYMS)(data, "dev", tmp_path)["TER"]
    assert ter > 0
    assert (tmp_path / "dev" / "ter_alignment").is_file()


def test_cleaner_is_separate_for_reference_and_hypothesis(tmp_path):
    """s2t.sh keeps --cleaner and --hyp_cleaner apart, so this does too."""
    pytest.importorskip("jiwer")
    metric = _wer(ref_cleaner=["whisper_basic"], hyp_cleaner=None)

    assert metric.ref_cleaner is not metric.hyp_cleaner


def test_tags_are_stripped_from_the_hypothesis_too(tmp_path):
    """Stripping one side turns an exact match into a total miss."""
    pytest.importorskip("jiwer")
    # hyp_key may name the tagged text rather than text_nospecial.
    data = _data(tmp_path, [ASR], [ASR])

    assert _wer()(data, "dev", tmp_path) == {"WER": 0.0}


def test_only_the_recipes_own_symbols_are_stripped(tmp_path):
    """Deriving the pattern from nlsyms is what keeps real content intact.

    A grammar-shaped pattern such as ``<[a-z]{3}>`` also eats ``<pre>`` and
    ``<xyz>``, which are not symbols OWSM ever emits.
    """
    assert strip_tags("5 < 10 > 3", PATTERN) == "5 < 10 > 3"
    assert strip_tags("x <-- y", PATTERN) == "x <-- y"
    assert strip_tags("wrap it in <xyz> tags", PATTERN) == "wrap it in <xyz> tags"
    assert strip_tags("<eng><st_deu><0.00> hallo<1.00>", PATTERN) == "hallo"
    assert strip_tags("<na> <nospeech> <notimestamps> hi", PATTERN) == "hi"


def test_cer_counts_a_space_as_a_token(tmp_path):
    """espnet2's CharTokenizer emits <space>, so s2t.sh's char pass counts it.

    Dropping spaces instead would score a missing word boundary as perfect.
    """
    pytest.importorskip("jiwer")
    assert _cer().units(["the fox"]) == ["t h e <space> f o x"]

    merged = _data(tmp_path, [ASR], ["thequick brown fox"])
    assert _cer()(merged, "dev", tmp_path)["CER"] > 0
