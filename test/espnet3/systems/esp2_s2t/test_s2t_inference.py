"""Tests for espnet3/systems/esp2_s2t/src/s2t_inference.py.

The module subclasses espnet2's S2T inference; these cover what the subclass
adds: a speaker-change symbol the beam search is told about, and the Whisper
timestamp rules, together with the two ways of asking for them that have no
decoder to ride on.
"""

from pathlib import Path

import numpy as np
import pytest

from espnet2.bin.s2t_inference import ScoreFilter
from espnet2.legacy.nets.beam_search import Hypothesis
from espnet2.tasks.s2t import S2TTask
from espnet3.systems.esp2_s2t.src.s2t_inference import Speech2Text

pytest.importorskip("whisper")


@pytest.fixture()
def token_list(tmp_path: Path):
    with (tmp_path / "tokens.txt").open("w") as f:
        tokens = [
            "<blank>",
            "<unk>",
            "<na>",
            "<nospeech>",
            "<eng>",
            "<zho>",
            "<asr>",
            "<st_eng>",
            "<st_zho>",
            "<notimestamps>",
            "<0.00>",
            "<1.00>",
            "a",
            "i",
            "<sos>",
            "<eos>",
            "<sop>",
        ]
        for tok in tokens:
            f.write(f"{tok}\n")
    return tmp_path / "tokens.txt"


@pytest.fixture()
def s2t_config_file(tmp_path: Path, token_list):
    # Write default configuration file
    S2TTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "s2t"),
            "--token_list",
            str(token_list),
            "--token_type",
            "char",
            "--decoder",
            "rnn",
            "--preprocessor_conf",
            "notime_symbol='<notimestamps>'",
            "--preprocessor_conf",
            "first_time_symbol='<0.00>'",
            "--preprocessor_conf",
            "last_time_symbol='<1.00>'",
            "--preprocessor_conf",
            "fs=2000",
            "--preprocessor_conf",
            "speech_length=1",
        ]
    )
    return tmp_path / "s2t" / "config.yaml"


def test_adaptive_timestamp_is_refused_when_ctc_takes_all_the_weight(
    s2t_config_file,
):
    """The adaptive rule rides on the decoder score, so it needs some weight.

    At ctc_weight 1.0 the decoder weight is zero, BeamSearch keeps only the
    scorers that have a weight, and the timestamp rules would be dropped
    without a word.
    """
    with pytest.raises(ValueError, match="ctc_weight"):
        Speech2Text(
            s2t_train_config=s2t_config_file,
            ctc_weight=1.0,
            adaptive_timestamp=True,
        )


def test_a_speaker_change_symbol_is_not_caught_by_the_ctc_weight_guard(
    s2t_config_file,
):
    """speaker_change_symbol registers as scorefilter, at a fixed weight.

    It does not ride on the decoder score, so ctc_weight 1.0 must not reject
    it. This fixture's token list is not Whisper-shaped, so the filter's own
    layout check stops it further in; reaching that check is the assertion,
    because the guard above would have stopped it earlier.
    """
    with pytest.raises(ValueError, match="Whisper layout"):
        Speech2Text(
            s2t_train_config=s2t_config_file,
            ctc_weight=1.0,
            beam_size=1,
            speaker_change_symbol="a",
        )


@pytest.fixture()
def token_list_sot(tmp_path: Path):
    """Token list that also carries a speaker-change symbol.

    Ordered like the Whisper vocabulary the timestamp scorer is written for:
    text below ``<eos>``, then the specials, then a contiguous timestamp range
    at the top. The speaker-change symbol sits with the text, as ``????``
    (id 25629) does below Whisper's ``<|endoftext|>`` (id 50257).
    ``WhisperTimestampFilter`` rejects any other arrangement, because its
    "a segment is closed, so no text" rule is the slice ``mask[:eos]``.
    """
    with (tmp_path / "tokens_sot.txt").open("w") as f:
        tokens = [
            "<blank>",
            "<unk>",
            "<na>",
            "a",
            "i",
            "<sc>",
            "<eos>",
            "<sos>",
            "<sop>",
            "<nospeech>",
            "<eng>",
            "<zho>",
            "<asr>",
            "<st_eng>",
            "<st_zho>",
            "<notimestamps>",
            "<0.00>",
            "<1.00>",
        ]
        for tok in tokens:
            f.write(f"{tok}\n")
    return tmp_path / "tokens_sot.txt"


@pytest.fixture()
def s2t_config_file_sot(tmp_path: Path, token_list_sot):
    S2TTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "s2t_sot"),
            "--token_list",
            str(token_list_sot),
            "--token_type",
            "char",
            "--decoder",
            "rnn",
            "--preprocessor_conf",
            "notime_symbol='<notimestamps>'",
            "--preprocessor_conf",
            "first_time_symbol='<0.00>'",
            "--preprocessor_conf",
            "last_time_symbol='<1.00>'",
            "--preprocessor_conf",
            "fs=2000",
            "--preprocessor_conf",
            "speech_length=1",
        ]
    )
    return tmp_path / "s2t_sot" / "config.yaml"


@pytest.mark.execution_timeout(10)
def test_Speech2Text_speaker_change_symbol_installs_filter(s2t_config_file_sot):
    speech2text = Speech2Text(
        s2t_train_config=s2t_config_file_sot,
        beam_size=1,
        maxlenratio=-5,
        predict_time=True,
        speaker_change_symbol="<sc>",
    )
    filt = speech2text.timestamp_filter
    assert filt is not None, "the SOT filter must replace the default one"
    assert speech2text.beam_search.scorers["scorefilter"] is filt
    token_list = speech2text.s2t_model.token_list
    assert filt.speaker_change == token_list.index("<sc>")
    assert filt.first_time == token_list.index("<0.00>")
    assert filt.last_time == token_list.index("<1.00>")

    results = speech2text(np.random.randn(1000))
    # The prompt is <sos><eng><asr> when timestamps are predicted, so the
    # filter must have been told where the generated tokens start.
    assert filt.sample_begin == 3
    for _text, token, token_int, _text_nospecial, hyp in results:
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)


def test_Speech2Text_adaptive_timestamp_installs_the_decoder_wrapper(
    s2t_config_file_sot,
):
    from espnet3.systems.esp2_s2t.src.whisper_timestamp_scorer import (
        WhisperTimestampDecoder,
    )

    speech2text = Speech2Text(
        s2t_train_config=s2t_config_file_sot,
        beam_size=1,
        maxlenratio=-5,
        predict_time=True,
        speaker_change_symbol="<sc>",
        adaptive_timestamp=True,
    )
    # The adaptive rule lives in the wrapper that replaces the decoder scorer;
    # without this assertion, swapping the wrapper for the bare decoder leaves
    # every other test green.
    wrapped = speech2text.beam_search.scorers["decoder"]
    assert isinstance(wrapped, WhisperTimestampDecoder)
    assert wrapped.filter is speech2text.timestamp_filter
    assert "scorefilter" not in speech2text.beam_search.scorers

    results = speech2text(np.random.randn(1000))
    for _text, token, token_int, _text_nospecial, hyp in results:
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)


@pytest.mark.execution_timeout(10)
def test_Speech2Text_without_speaker_change_symbol_is_unchanged(s2t_config_file):
    speech2text = Speech2Text(
        s2t_train_config=s2t_config_file,
        beam_size=1,
        maxlenratio=-5,
    )
    assert speech2text.timestamp_filter is None
    assert isinstance(speech2text.beam_search.scorers["scorefilter"], ScoreFilter)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"speaker_change_symbol": "<sc>"},
        {"adaptive_timestamp": True},
        {"speaker_change_symbol": "<sc>", "adaptive_timestamp": True},
    ],
)
def test_Speech2Text_timestamp_options_reject_decoding_without_time(
    s2t_config_file_sot, kwargs
):
    # A prompt built without timestamps forbids them, while both options
    # constrain them, so decoding that way has to be rejected instead of
    # running into a dead end. predict_time is overridable per call, so the
    # effective value is what matters, not the one given here.
    speech2text = Speech2Text(
        s2t_train_config=s2t_config_file_sot,
        beam_size=1,
        maxlenratio=-5,
        predict_time=True,
        **kwargs,
    )
    with pytest.raises(ValueError):
        speech2text(np.random.randn(1000), predict_time=False)


def test_Speech2Text_timestamp_options_allow_a_per_call_override(s2t_config_file_sot):
    # predict_time defaults to False, so installing the filter must not block
    # a caller that turns timestamps on for one utterance.
    speech2text = Speech2Text(
        s2t_train_config=s2t_config_file_sot,
        beam_size=1,
        maxlenratio=-5,
        speaker_change_symbol="<sc>",
    )
    assert speech2text.timestamp_filter is not None
    results = speech2text(np.random.randn(1000), predict_time=True)
    assert speech2text.timestamp_filter.sample_begin == 3
    for _text, token, token_int, _text_nospecial, hyp in results:
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)


def _whisper_decoder_stub(separator, position):
    """A Speech2Text holding only what _keep_special_tokens reads.

    The parent's Whisper converter, and a token list in the multilingual
    layout: <|notimestamps|> at 50363, <|0.00|> to <|30.00|> after it, and the
    separator at ``position``.
    """
    from types import SimpleNamespace

    from espnet2.text.whisper_token_id_converter import (
        OpenAIWhisperTokenIDConverter as DroppingConverter,
    )

    tokens = ["x"] * 51866
    tokens[50363], tokens[50364], tokens[51864] = (
        "<|notimestamps|>",
        "<|0.00|>",
        "<|30.00|>",
    )
    tokens[position] = separator
    decoder = object.__new__(Speech2Text)
    decoder.converter = DroppingConverter(model_type="whisper_multilingual")
    decoder.s2t_model = SimpleNamespace(token_list=tokens)
    decoder.s2t_train_args = SimpleNamespace(
        preprocessor_conf={
            "first_time_symbol": "<|0.00|>",
            "last_time_symbol": "<|30.00|>",
            "notime_symbol": "<|notimestamps|>",
        }
    )
    return decoder


@pytest.mark.execution_timeout(60)
@pytest.mark.parametrize(
    "separator, position",
    [("????", 25629), ("<sc>", 51865)],
    ids=["in the BPE vocabulary", "appended to it"],
)
def test_a_whisper_hypothesis_keeps_its_timestamps_and_separator(separator, position):
    """The parent's Whisper converter drops special tokens; this one must not.

    Otherwise the text DER is scored on has no timestamps, and a separator the
    checkpoint appended to its vocabulary is lost.
    """
    decoder = _whisper_decoder_stub(separator, position)
    Speech2Text._keep_special_tokens(decoder, separator)
    assert decoder.converter.ids2tokens([50364, position, 51864]) == [
        "<|0.00|>",
        separator,
        "<|30.00|>",
    ]


@pytest.mark.execution_timeout(60)
def test_a_token_list_that_disagrees_with_the_converter_is_refused():
    """The text must say what the model emitted, so the two ids must match."""
    decoder = _whisper_decoder_stub("????", 100)
    with pytest.raises(ValueError, match="token list"):
        Speech2Text._keep_special_tokens(decoder, "????")
