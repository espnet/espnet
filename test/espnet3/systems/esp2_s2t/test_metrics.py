"""OWSM metrics: the three rates s2t.sh reports, over tagged text."""

from __future__ import annotations

from pathlib import Path

import pytest

from espnet3.api.inference import Field
from espnet3.contract.metrics import MetricContractError
from espnet3.systems.esp2_s2t.metrics.cer import CER
from espnet3.systems.esp2_s2t.metrics.tags import strip_tags, tag_pattern
from espnet3.systems.esp2_s2t.metrics.ter import TER
from espnet3.systems.esp2_s2t.metrics.wer import WER

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


@pytest.fixture(scope="module")
def tiny_bpemodel(tmp_path_factory):
    """Train a throwaway SentencePiece model, so the test needs no artifact.

    The metric strips tags before tokenizing, so the pieces this has to cover
    are plain words; the vocabulary only needs to be large enough to hold the
    alphabet.
    """
    spm = pytest.importorskip("sentencepiece")
    tmp_path = tmp_path_factory.mktemp("bpe")
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


# --------------------------------------------------------------------------
# Tags
# --------------------------------------------------------------------------


def test_markup_is_stripped_before_scoring():
    assert strip_tags(ASR, PATTERN) == "the quick brown fox"
    assert strip_tags("<eng><asr><notimestamps> no times", PATTERN) == "no times"


def test_a_longer_symbol_wins_over_a_prefix_of_it():
    """A longer symbol wins over a prefix of it.

    The inventory nests: <na> is a prefix of <nan>, a language in its own
    right, so a shortest-first alternation would leave "n>" behind.
    """
    assert strip_tags("<nan> hi", PATTERN) == "hi"


def test_nlsyms_is_required_to_remove_tags():
    """nlsyms is required whenever tags are removed.

    There is no default inventory here -- which symbols exist is the
    recipe's business, exactly as for tokenizer.nlsyms.
    """
    with pytest.raises(RuntimeError, match="nlsyms is required"):
        WER()

    assert WER(remove_tags=False).pattern is None


def test_only_the_recipes_own_symbols_are_stripped():
    """Deriving the pattern from nlsyms is what keeps real content intact.

    A grammar-shaped pattern such as ``<[a-z]{3}>`` also eats ``<pre>`` and
    ``<xyz>``, which are not symbols OWSM ever emits.
    """
    assert strip_tags("5 < 10 > 3", PATTERN) == "5 < 10 > 3"
    assert strip_tags("x <-- y", PATTERN) == "x <-- y"
    assert strip_tags("wrap it in <xyz> tags", PATTERN) == "wrap it in <xyz> tags"
    assert strip_tags("<eng><st_deu><0.00> hallo<1.00>", PATTERN) == "hallo"
    assert strip_tags("<na> <nospeech> <notimestamps> hi", PATTERN) == "hi"


def test_tags_are_removed_so_a_perfect_transcription_scores_zero(tmp_path):
    """The hypothesis never carries tags, so a tagged reference cannot match."""
    pytest.importorskip("jiwer")
    data = _data(tmp_path, [ASR], ["the quick brown fox"])

    assert _wer()(data, "dev", tmp_path) == {"WER": 0.0}


def test_tags_are_stripped_from_the_hypothesis_too(tmp_path):
    """Stripping one side turns an exact match into a total miss."""
    pytest.importorskip("jiwer")
    # `hyp` may be bound to the tagged text rather than text_nospecial.
    data = _data(tmp_path, [ASR], [ASR])

    assert _wer()(data, "dev", tmp_path) == {"WER": 0.0}


def test_remove_tags_false_reproduces_s2t_sh(tmp_path):
    """s2t.sh asks for tag removal and cannot achieve it on OWSM text.

    Kept only so the two numbers can be compared; a perfect transcription
    scores a large WER this way.
    """
    pytest.importorskip("jiwer")
    data = _data(tmp_path, [ASR], ["the quick brown fox"])

    assert WER(remove_tags=False)(data, "dev", tmp_path)["WER"] > 0


# --------------------------------------------------------------------------
# What each rate counts
# --------------------------------------------------------------------------


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


def test_cer_counts_a_space_as_a_character(tmp_path):
    """espnet2's CharTokenizer emits <space>, so s2t.sh's char pass counts it.

    Dropping spaces instead would score a missing word boundary as perfect.
    """
    pytest.importorskip("jiwer")
    # "the fox" against "thefox": one deletion out of seven characters.
    data = _data(tmp_path, ["the fox"], ["thefox"])

    assert _cer()(data, "dev", tmp_path) == {"CER": 14.29}


def test_ter_counts_bpe_pieces(tmp_path, tiny_bpemodel):
    """s2t.sh's ter is a token error rate over subwords, not sacreBLEU's TER."""
    pytest.importorskip("jiwer")
    data = _data(tmp_path, [ASR], ["the quick brown fix"])

    ter = TER(bpemodel=tiny_bpemodel, nlsyms=NLSYMS)(data, "dev", tmp_path)["TER"]
    assert ter > 0
    assert (tmp_path / "dev" / "ter_alignment").is_file()


def test_cleaner_is_separate_for_reference_and_hypothesis():
    """s2t.sh keeps --cleaner and --hyp_cleaner apart, so this does too."""
    metric = _wer(ref_cleaner=["whisper_basic"], hyp_cleaner=None)

    assert metric.ref_cleaner is not metric.hyp_cleaner


# --------------------------------------------------------------------------
# The metric contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls, key", [(CER, "CER"), (WER, "WER"), (TER, "TER")])
def test_declares_text_inputs_and_number_output(cls, key):
    assert cls.inputs == (Field("ref", "text"), Field("hyp", "text"))
    assert cls.outputs[0].name == key
    assert cls.outputs[0].kind == "number"


@pytest.mark.parametrize("make, key", [(_cer, "CER"), (_wer, "WER")])
def test_two_chunks_equal_one_shot(make, key):
    pytest.importorskip("jiwer")
    refs = [ASR, DE]
    hyps = ["the quick brown fox", "der schnelle braune Fuchs"]
    one = make()
    one.update(ref=refs, hyp=hyps)
    two = make()
    two.update(ref=refs[:1], hyp=hyps[:1])
    two.update(ref=refs[1:], hyp=hyps[1:])

    assert one.compute() == two.compute() == {key: 0.0}


def test_ter_two_chunks_equal_one_shot(tiny_bpemodel):
    pytest.importorskip("jiwer")
    refs = [ASR, DE]
    hyps = ["the quick brown fix", "der schnelle braune Fuchs"]
    one = TER(bpemodel=tiny_bpemodel, nlsyms=NLSYMS)
    one.update(ref=refs, hyp=hyps)
    two = TER(bpemodel=tiny_bpemodel, nlsyms=NLSYMS)
    two.update(ref=refs[:1], hyp=hyps[:1])
    two.update(ref=refs[1:], hyp=hyps[1:])

    assert one.compute() == two.compute()


def test_nothing_to_score_raises():
    with pytest.raises(ValueError, match="nothing to score"):
        _wer().compute()


def test_update_rejects_unequal_lengths():
    with pytest.raises(MetricContractError, match="has 1 items"):
        _wer().update(ref=["a", "b"], hyp=["a"])


def test_update_without_artifacts_keeps_no_text():
    pytest.importorskip("jiwer")
    metric = _wer()
    metric.update(ref=[ASR], hyp=["the quick brown fix"])
    metric.compute()
    state = metric.state()

    assert set(state) == {"hits", "substitutions", "deletions", "insertions"}
    assert all(isinstance(value, int) for value in state.values())
    assert metric._alignments == []
    assert metric.collect_artifacts is False


def test_call_without_output_dir_writes_no_alignment(tmp_path):
    pytest.importorskip("jiwer")
    data = _data(tmp_path, [ASR], ["the quick brown fix"])

    assert _wer()(data, "dev", None)["WER"] > 0
    assert not (tmp_path / "dev").exists()


def test_call_with_output_dir_writes_the_alignment_once(tmp_path):
    pytest.importorskip("jiwer")
    data = _data(tmp_path, [ASR], ["the quick brown fix"])
    metric = _wer()
    metric(data, "dev", tmp_path / "out")
    text = (tmp_path / "out" / "dev" / "wer_alignment").read_text(encoding="utf-8")

    assert "WER: 25.00" in text
    assert metric._alignments == []
    assert metric.collect_artifacts is False


PAIRS = [
    (ASR, "the quick brown fox"),
    (ASR, "the quick brown fix"),
    (DE, "der schnelle braune Fuchs"),
    (DE, "voellig falsch hier"),
    (ASR, "the fox"),
] * 2
SPLITS = [(5, 5), (9, 1), (10, 0), (0, 10)]


def _pool(make, splits):
    ranks, start = [], 0
    for size in splits:
        metric = make()
        part = PAIRS[start : start + size]
        if part:
            metric.update(ref=[r for r, _ in part], hyp=[h for _, h in part])
        if size == 0:
            zeros = {"hits": 0, "substitutions": 0, "deletions": 0, "insertions": 0}
            assert metric.state() == zeros
        start += size
        ranks.append(metric)
    merged = make()
    merged.load_state(merged.merge_states([m.state() for m in ranks]))
    whole = make()
    whole.update(ref=[r for r, _ in PAIRS], hyp=[h for _, h in PAIRS])

    assert merged.compute() == whole.compute()
    assert all(m._alignments == [] for m in ranks)


@pytest.mark.parametrize("splits", SPLITS)
@pytest.mark.parametrize("make", [_cer, _wer])
def test_two_ranks_pool_counts(make, splits):
    pytest.importorskip("jiwer")
    _pool(make, splits)


@pytest.mark.parametrize("splits", SPLITS)
def test_two_ranks_pool_counts_ter(tiny_bpemodel, splits):
    pytest.importorskip("jiwer")
    _pool(lambda: TER(bpemodel=tiny_bpemodel, nlsyms=NLSYMS), splits)
