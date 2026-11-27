"""Tests for espnet3/systems/esp2_s2t/src/sot/s2t_inference.py.

The module subclasses espnet2's S2T inference. These tests cover what the subclass
adds. It passes a speaker-change symbol to the beam search, and it adds the
Whisper timestamp rules. The tests also check the two options at ctc_weight 1.0,
where the decoder has no weight.
"""

from pathlib import Path

import numpy as np
import pytest

from espnet2.bin.s2t_inference import ScoreFilter
from espnet2.legacy.nets.beam_search import Hypothesis
from espnet2.tasks.s2t import S2TTask
from espnet3.systems.esp2_s2t.src.sot.s2t_inference import Speech2Text

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
    """adaptive_timestamp raises ValueError at ctc_weight 1.0.

    The adaptive rule uses the decoder score, so the decoder needs some weight. At
    ctc_weight 1.0 the decoder weight is zero. BeamSearch keeps only the scorers
    that have a weight, so the timestamp rules would be dropped without a warning.
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
    """The ctc_weight check does not reject speaker_change_symbol.

    speaker_change_symbol is registered as scorefilter with a fixed weight, and it
    does not use the decoder score. This fixture's token list does not have the
    Whisper layout, so the filter's own layout check raises an error. The
    ctc_weight check runs before it, so this error shows that the ctc_weight check
    accepted the option.
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
    """Write a token list that also has a speaker-change symbol.

    The order follows the Whisper vocabulary that the timestamp scorer is written
    for. Text tokens come before ``<eos>``, then the special tokens, then a
    contiguous timestamp range at the end. The speaker-change symbol is among the
    text tokens, as ``????`` (id 25629) is below Whisper's ``<|endoftext|>``
    (id 50257). ``WhisperTimestampFilter`` rejects any other order, because its
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
    # The prompt is <sos><eng><asr> when timestamps are predicted. So the filter
    # must know where the generated tokens start.
    assert filt.sample_begin == 3
    for _text, token, token_int, _text_nospecial, hyp in results:
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)


def test_Speech2Text_adaptive_timestamp_installs_the_decoder_wrapper(
    s2t_config_file_sot,
):
    from espnet3.systems.esp2_s2t.src.sot.whisper_timestamp_scorer import (
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
    # The adaptive rule is in the wrapper that replaces the decoder scorer.
    # Without this assertion, every other test would still pass if the wrapper
    # were replaced by the plain decoder.
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
    # A prompt built without timestamps forbids them, and both options constrain
    # them. So decoding in this way must raise an error. Otherwise it would reach
    # a dead end. predict_time can be set again for each call, so the check must
    # use the value of the call (False). The constructor value here is True.
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
    # predict_time defaults to False. Installing the filter must not stop a caller
    # from turning timestamps on for one utterance.
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
    """Build a Speech2Text with only the parts that _keep_special_tokens reads.

    It has the parent's Whisper converter and a token list in the multilingual
    layout. <|notimestamps|> is at 50363, <|0.00|> to <|30.00|> come after it, and
    the separator is at ``position``.
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
    """_keep_special_tokens keeps the timestamps and the separator in the text.

    The parent's Whisper converter drops special tokens. With that converter, the
    text that DER scores would have no timestamps, and a separator that the
    checkpoint appended to its vocabulary would be lost.
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
    """_keep_special_tokens raises ValueError when the two separator ids differ.

    The text must show what the model produced, so the token list and the
    converter must give the separator the same id.
    """
    decoder = _whisper_decoder_stub("????", 100)
    with pytest.raises(ValueError, match="token list"):
        Speech2Text._keep_special_tokens(decoder, "????")
