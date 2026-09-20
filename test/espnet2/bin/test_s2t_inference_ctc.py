from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest
import torch

from espnet2.bin.s2t_inference import Speech2Text as Speech2TextBase
from espnet2.bin.s2t_inference_ctc import (
    Speech2Text,
    Speech2TextGreedySearch,
    get_parser,
    main,
)
from espnet2.legacy.nets.beam_search import Hypothesis
from espnet2.tasks.s2t_ctc import S2TTask


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


@pytest.fixture()
def token_list(tmp_path: Path):
    with (tmp_path / "tokens.txt").open("w") as f:
        tokens = [
            "<blank>",
            "<unk>",
            "<na>",
            "<nolang>",
            "<eng>",
            "<zho>",
            "<asr>",
            "<st_eng>",
            "a",
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
            "--promptencoder_conf",
            "output_size=4",
            "--preprocessor_conf",
            "fs=2000",
            "--preprocessor_conf",
            "speech_length=3",
            "--frontend_conf",
            "fs=16k",
            "--frontend_conf",
            "hop_length=160",
            "--encoder_conf",
            "input_layer=conv2d8",
        ]
    )
    return tmp_path / "s2t" / "config.yaml"


@pytest.mark.execution_timeout(5)
def test_Speech2Text(s2t_config_file):
    speech2text = Speech2Text(
        s2t_train_config=s2t_config_file,
        beam_size=1,
        maxlenratio=-5,
    )
    speech = np.random.randn(3000)
    results = speech2text(speech)
    for text, token, token_int, text_nospecial, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(text_nospecial, str)
        assert isinstance(hyp, Hypothesis)


@pytest.mark.execution_timeout(5)
def test_Speech2Text_overwrite_args(s2t_config_file):
    speech2text = Speech2Text(
        s2t_train_config=s2t_config_file,
        beam_size=1,
        maxlenratio=-5,
    )
    speech = np.random.randn(3000)
    results = speech2text(
        speech,
        text_prev="<na>",
        lang_sym="<zho>",
        task_sym="<st_eng>",
    )
    for text, token, token_int, text_nospecial, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(text_nospecial, str)
        assert isinstance(hyp, Hypothesis)


@pytest.mark.execution_timeout(5)
def test_Speech2Text_quantized(s2t_config_file):
    speech2text = Speech2Text(
        s2t_train_config=s2t_config_file,
        beam_size=1,
        maxlenratio=-5,
        quantize_s2t_model=True,
    )
    speech = np.random.randn(3000)
    results = speech2text(speech)
    for text, token, token_int, text_nospecial, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(text_nospecial, str)
        assert isinstance(hyp, Hypothesis)


@pytest.mark.execution_timeout(10)
def test_the_base_class_loads_a_ctc_only_checkpoint(s2t_config_file):
    # one interface for both kinds of model: no decoder here, so there is no
    # beam search and __call__ decodes the best path
    speech2text = Speech2TextBase(s2t_train_config=s2t_config_file, beam_size=1)
    assert speech2text.ctc_only is True
    # it searches over the CTC scorer alone: there is no decoder to score with
    assert speech2text.beam_search is not None
    assert "ctc" in speech2text.beam_search.scorers
    assert "decoder" not in speech2text.beam_search.scorers

    speech = np.random.randn(3000)
    searched = speech2text(speech)
    assert isinstance(searched[0][4], Hypothesis)

    # and best_path is the unsearched route, on the same object
    assert speech2text.best_path(speech)[0][4] is None


@pytest.mark.execution_timeout(10)
def test_the_base_class_batch_decodes_a_ctc_only_checkpoint(s2t_config_file):
    speech2text = Speech2TextBase(s2t_train_config=s2t_config_file, beam_size=1)
    batched = speech2text.batch_decode(torch.randn(3, 3000))
    assert len(batched) == 3
    for results in batched:
        assert len(results) == 1 and isinstance(results[0][4], Hypothesis)


@pytest.mark.execution_timeout(30)
def test_decode_long_returns_one_segment_for_a_ctc_only_checkpoint(s2t_config_file):
    speech2text = Speech2TextBase(s2t_train_config=s2t_config_file)
    # longer than the buffer the model was trained on, so it is chunked
    speech = np.random.randn(int(speech2text.sample_rate * 7))
    segments = speech2text.decode_long(speech, batch_size=2, context_len_in_secs=0.5)

    assert len(segments) == 1  # no timestamps in a CTC-only model
    start, end, text = segments[0]
    assert start == 0.0
    assert end == pytest.approx(len(speech) / speech2text.sample_rate)
    assert isinstance(text, str)


@pytest.mark.execution_timeout(40)
def test_the_deprecated_class_decodes_the_same_way(s2t_config_file):
    speech2text = Speech2TextBase(s2t_train_config=s2t_config_file)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        old = Speech2TextGreedySearch(s2t_train_config=s2t_config_file)

    speech = np.random.randn(int(speech2text.sample_rate * 7))
    segments = speech2text.decode_long(speech, batch_size=2, context_len_in_secs=0.5)
    assert (
        old.decode_long_batched_buffered(speech, batch_size=2, context_len_in_secs=0.5)
        == segments[0][2]
    )
    assert (
        old.batch_decode(speech, batch_size=2, context_len_in_secs=0.5)
        == segments[0][2]
    )
    # a list in, a list out
    assert old.batch_decode([speech], batch_size=2, context_len_in_secs=0.5) == [
        segments[0][2]
    ]


@pytest.mark.execution_timeout(10)
def test_from_pretrained_builds_the_class_it_was_called_on(s2t_config_file):
    # it named the base class, so every subclass got the base class back
    with pytest.warns(DeprecationWarning):
        built = Speech2TextGreedySearch.from_pretrained(
            s2t_train_config=s2t_config_file
        )
    assert isinstance(built, Speech2TextGreedySearch)


@pytest.mark.execution_timeout(10)
def test_greedy_search_refuses_a_checkpoint_with_a_decoder(tmp_path, token_list):
    # its long-form paths read the CTC head directly; an encoder-decoder
    # model would silently decode with something else
    from espnet2.tasks.s2t import S2TTask as S2TAttentionTask

    S2TAttentionTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "attention"),
            "--token_list",
            str(token_list),
            "--token_type",
            "char",
            "--decoder",
            "rnn",
            "--preprocessor_conf",
            "notime_symbol='<na>'",
            "--preprocessor_conf",
            "first_time_symbol='<na>'",
            "--preprocessor_conf",
            "last_time_symbol='<na>'",
        ]
    )
    with pytest.raises(ValueError, match="CTC-only"):
        Speech2TextGreedySearch(s2t_train_config=tmp_path / "attention" / "config.yaml")


@pytest.mark.execution_timeout(5)
def test_Speech2TextGreedy(s2t_config_file):
    speech2text = Speech2TextGreedySearch(
        s2t_train_config=s2t_config_file,
        maxlenratio=-5,
    )
    speech = np.random.randn(3000)
    results = speech2text(speech)
    for text, token, token_int, text_nospecial, _ in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(text_nospecial, str)


@pytest.mark.execution_timeout(5)
def test_Speech2TextGreedy_overwrite_args(s2t_config_file):
    speech2text = Speech2TextGreedySearch(
        s2t_train_config=s2t_config_file,
        maxlenratio=-5,
    )
    speech = np.random.randn(3000)
    results = speech2text(
        speech,
        text_prev="<na>",
        lang_sym="<zho>",
        task_sym="<st_eng>",
    )
    for text, token, token_int, text_nospecial, _ in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(text_nospecial, str)


@pytest.mark.execution_timeout(5)
def test_Speech2TextGreedy_quantized(s2t_config_file):
    speech2text = Speech2TextGreedySearch(
        s2t_train_config=s2t_config_file,
        maxlenratio=-5,
        quantize_s2t_model=True,
    )
    speech = np.random.randn(3000)
    results = speech2text(speech)
    for text, token, token_int, text_nospecial, _ in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(text_nospecial, str)


@pytest.mark.execution_timeout(5)
def test_Speech2TextGreedy_longform(s2t_config_file):
    speech2text = Speech2TextGreedySearch(
        s2t_train_config=s2t_config_file,
        maxlenratio=-5,
    )
    speech = np.random.randn(3000)
    result = speech2text.decode_long_batched_buffered(
        speech,
        context_len_in_secs=1,
    )
    assert isinstance(result, str)


@pytest.mark.execution_timeout(20)
def test_Speech2TextGreedy_batchdecode(s2t_config_file):
    speech2text = Speech2TextGreedySearch(
        s2t_train_config=s2t_config_file,
    )
    result = speech2text.batch_decode(
        [
            np.random.randn(1000),
            np.random.randn(7000),
        ],
        batch_size=2,
        context_len_in_secs=1,
    )
    assert isinstance(result[0], str) and isinstance(result[1], str)
