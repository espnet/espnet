from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest
import torch

from espnet2.bin.s2t_inference import Speech2Text, get_parser, main
from espnet2.legacy.nets.beam_search import Hypothesis
from espnet2.tasks.s2t import S2TTask


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


@pytest.mark.execution_timeout(5)
def test_Speech2Text(s2t_config_file):
    speech2text = Speech2Text(
        s2t_train_config=s2t_config_file,
        beam_size=1,
        maxlenratio=-5,
    )
    speech = np.random.randn(1000)
    results = speech2text(speech)
    for text, token, token_int, text_nospecial, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(text_nospecial, str)
        assert isinstance(hyp, Hypothesis)


@pytest.fixture()
def s2t_config_file_transformer(tmp_path: Path, token_list):
    # A decoder whose scorers are batch scorers, as required by batch decoding
    S2TTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "s2t_transformer"),
            "--token_list",
            str(token_list),
            "--token_type",
            "char",
            "--decoder",
            "transformer",
            "--preprocessor_conf",
            "notime_symbol='<notimestamps>'",
            "--preprocessor_conf",
            "first_time_symbol='<0.00>'",
            "--preprocessor_conf",
            "last_time_symbol='<1.00>'",
            "--preprocessor_conf",
            "fs=2000",
            # long enough that the encoder produces more frames than the
            # prompt has tokens, which `CTCPrefixScoreTH` requires
            "--preprocessor_conf",
            "speech_length=4",
        ]
    )
    return tmp_path / "s2t_transformer" / "config.yaml"


@pytest.mark.execution_timeout(60)
@pytest.mark.parametrize("ctc_weight", [0.0, 0.3])
def test_Speech2Text_batch_decode(s2t_config_file_transformer, ctc_weight):
    """`batch_decode` must reproduce the per-utterance results."""
    kwargs = dict(
        s2t_train_config=s2t_config_file_transformer,
        beam_size=2,
        # NOTE: `CTCPrefixScoreTH` cannot score a prefix that is longer than
        # the encoder output, and an S2T hypothesis already starts with a
        # 4-token prompt, so cap the output length well below that.
        maxlenratio=-8,
        ctc_weight=ctc_weight,
    )
    single = Speech2Text(batch_size=1, **kwargs)
    batched = Speech2Text(batch_size=3, **kwargs)
    # both are randomly initialized, so make them the same model
    batched.s2t_model.load_state_dict(single.s2t_model.state_dict())
    batched.beam_search.nn_dict.load_state_dict(single.beam_search.nn_dict.state_dict())

    # every utterance is padded or trimmed to the same fixed length anyway
    lengths = [4000, 3000, 9000]
    speeches = [np.random.randn(n) for n in lengths]

    expected = [single(sp) for sp in speeches]

    padded = np.zeros((len(speeches), max(lengths)))
    for i, sp in enumerate(speeches):
        padded[i, : len(sp)] = sp
    actual = batched.batch_decode(torch.tensor(padded).float(), torch.tensor(lengths))

    assert len(actual) == len(speeches)
    for exp, act in zip(expected, actual):
        assert [e[1] for e in exp] == [a[1] for a in act]
        np.testing.assert_allclose(
            float(exp[0][4].score), float(act[0][4].score), rtol=1e-4
        )


@pytest.mark.execution_timeout(30)
def test_Speech2Text_batch_decode_rejects_non_batch_scorer(s2t_config_file):
    """The RNN decoder is not a batch scorer, so batching must be refused."""
    with pytest.raises(NotImplementedError):
        Speech2Text(
            s2t_train_config=s2t_config_file,
            beam_size=1,
            maxlenratio=-5,
            batch_size=2,
        )


@pytest.mark.execution_timeout(60)
def test_Speech2Text_batch_decode_text_prev(s2t_config_file_transformer):
    """Per-utterance prompts must all have the same length."""
    batched = Speech2Text(
        s2t_train_config=s2t_config_file_transformer,
        beam_size=2,
        maxlenratio=-5,
        batch_size=2,
    )
    speech = torch.randn(2, 8000)
    lengths = torch.tensor([8000, 8000])

    text_prev = torch.tensor([[12, 13], [13, 12]])  # "a i" / "i a"
    results = batched.batch_decode(speech, lengths, text_prev=text_prev)
    assert len(results) == 2

    with pytest.raises(ValueError):
        batched.batch_decode(
            speech,
            lengths,
            text_prev=torch.tensor([[12, 13], [13, 0]]),
            text_prev_lengths=torch.tensor([2, 1]),
        )


@pytest.mark.execution_timeout(5)
def test_Speech2Text_overwrite_args(s2t_config_file):
    speech2text = Speech2Text(
        s2t_train_config=s2t_config_file,
        beam_size=1,
        maxlenratio=-5,
    )
    speech = np.random.randn(1000)
    results = speech2text(
        speech,
        text_prev="<na>",
        lang_sym="<zho>",
        task_sym="<st_eng>",
        predict_time=True,
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
    speech = np.random.randn(1000)
    results = speech2text(speech)
    for text, token, token_int, text_nospecial, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(text_nospecial, str)
        assert isinstance(hyp, Hypothesis)
