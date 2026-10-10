import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest
import torch

from espnet2.bin.s2t_inference import Speech2Text, get_parser, main
from espnet2.legacy.nets.beam_search import Hypothesis
from espnet2.tasks.s2t import S2TTask
from espnet2.tasks.s2t_ctc import S2TTask as S2TCTCTask


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


@pytest.fixture()
def s2t_long_form_config_file(tmp_path: Path, token_list):
    """The fixture above with the rate, hop and input layer the long-form paths need.

    2 kHz audio, a 200-sample hop and conv2d's four-fold subsampling: the
    window is one second, spanned by <0.00> and <1.00>.
    """
    S2TTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "s2t_long"),
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
            "--frontend_conf",
            "fs=2000",
            "--frontend_conf",
            "hop_length=200",
            "--encoder",
            "transformer",
            "--encoder_conf",
            "input_layer=conv2d",
            "--encoder_conf",
            "output_size=16",
            "--encoder_conf",
            "linear_units=16",
            "--encoder_conf",
            "num_blocks=1",
            "--encoder_conf",
            "attention_heads=2",
        ]
    )
    return tmp_path / "s2t_long" / "config.yaml"


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


@pytest.mark.execution_timeout(5)
def test_the_long_form_window_is_read_from_the_model(s2t_config_file):
    """The window is `speech_length`; the timestamp symbols only spell it.

    The threshold for "this utterance was cut off by the window" used to be
    written out as OWSM's `<29.00>` - 30 s minus a second - so a 20 s model
    decoded a long recording into `KeyError: '<29.00>'`. It is derived now,
    and from the window rather than from a symbol's name, so a checkpoint
    that spells its timestamps differently needs no format agreed here.
    """
    speech2text = Speech2Text(s2t_train_config=s2t_config_file, beam_size=1)
    first, last, step = speech2text._time_ids()

    # this fixture's window is one second, spanned by <0.00> and <1.00>
    assert step == pytest.approx(1.0)
    assert speech2text._near_window_end() == last - 1 == first

    # and what OWSM and POWSM each describe, without either config present
    for window, symbols, expected in ((30, 1500, 0.02), (20, 1000, 0.02)):
        speech2text.preprocessor_conf["speech_length"] = window
        speech2text.converter.token2id["<last>"] = first + symbols
        speech2text.preprocessor_conf["last_time_symbol"] = "<last>"
        _, last, step = speech2text._time_ids()
        assert step == pytest.approx(expected)
        # a second before the end of the window, in that model's own steps
        assert speech2text._near_window_end() == last - round(1 / expected)


@pytest.mark.execution_timeout(5)
def test_a_window_or_a_step_may_change_without_changing_this(s2t_config_file):
    """Both come from the checkpoint, so a new one needs nothing agreed here.

    30 s at 0.02 is OWSM, 20 s at 0.02 is POWSM, and the other two are models
    that do not exist yet: a finer step, and a step coarser than the second
    the threshold is measured in.
    """
    speech2text = Speech2Text(s2t_train_config=s2t_config_file, beam_size=1)
    first = speech2text.converter.token2id["<0.00>"]

    for window, steps, expected_step in (
        (30, 1500, 0.02),  # OWSM
        (20, 1000, 0.02),  # POWSM
        (10, 1000, 0.01),  # a finer grid
        (60, 20, 3.0),  # a step coarser than a second
    ):
        speech2text.converter.token2id["<last>"] = first + steps
        speech2text.preprocessor_conf.update(
            speech_length=window,
            last_time_symbol="<last>",
            first_time_symbol="<0.00>",
        )
        _, last, step = speech2text._time_ids()
        assert step == pytest.approx(expected_step)

        # a second before the end, or one step when a step is longer than a
        # second - never the end itself, which would be no threshold at all
        back = max(1, round(1.0 / expected_step))
        assert speech2text._near_window_end() == last - back


@pytest.mark.execution_timeout(5)
def test_a_config_at_odds_with_itself_is_named(s2t_config_file):
    """Timestamps named after times have to agree with the window.

    The check is on the config, not the contract: it runs only because both
    symbols here are named after seconds. A checkpoint that writes its
    timestamps some other way is read from its ids alone.
    """
    speech2text = Speech2Text(s2t_train_config=s2t_config_file, beam_size=1)
    first = speech2text.converter.token2id["<0.00>"]
    speech2text.converter.token2id["<20.00>"] = first + 1000
    speech2text.preprocessor_conf.update(
        speech_length=30, last_time_symbol="<20.00>"  # the symbols say 20
    )

    with pytest.raises(RuntimeError, match="span 20 s.*speech_length says 30"):
        speech2text._time_ids()

    # a symbol that is not a time is not checked, only used
    speech2text.converter.token2id["<end>"] = first + 1000
    speech2text.preprocessor_conf["last_time_symbol"] = "<end>"
    assert speech2text._time_ids()[2] == pytest.approx(0.03)


@pytest.mark.execution_timeout(5)
def test_a_configs_stated_resolution_does_not_override_its_vocabulary(
    s2t_config_file,
):
    """POWSM states 0.04 and steps 0.02; the vocabulary is the one to believe.

    Read at twice its value, a timestamp points past the end of the window it
    came from, and long-form decoding cuts the recording in the wrong places.
    """
    speech2text = Speech2Text(s2t_train_config=s2t_config_file, beam_size=1)
    first = speech2text.converter.token2id["<0.00>"]
    speech2text.converter.token2id["<last>"] = first + 1000
    speech2text.preprocessor_conf.update(
        speech_length=20, speech_resolution=0.04, last_time_symbol="<last>"
    )

    assert speech2text._time_ids()[2] == pytest.approx(0.02)


@pytest.mark.execution_timeout(5)
def test_best_path_on_an_encoder_decoder_model(s2t_config_file):
    # the CTC branch of a model that also has a decoder
    speech2text = Speech2Text(s2t_train_config=s2t_config_file, beam_size=1)
    assert speech2text.ctc_only is False
    results = speech2text.best_path(np.random.randn(1000))
    assert len(results) == 1
    text, token, token_int, text_nospecial, hyp = results[0]
    assert isinstance(text, str) and isinstance(text_nospecial, str)
    assert all(isinstance(t, str) for t in token)
    # nothing was searched, so there is no hypothesis to report
    assert hyp is None
    blank = speech2text.s2t_model.blank_id
    assert blank not in token_int
    assert all(a != b for a, b in zip(token_int, token_int[1:]))


def test_ctc_only_decoding_says_it_is_not_best_path(s2t_config_file, caplog):
    # "CTC decoding" is the name of two different things, and a user who
    # asked for one and got the other can only tell by the clock
    with caplog.at_level(logging.INFO):
        s2t = Speech2Text(s2t_train_config=s2t_config_file, ctc_weight=1.0, beam_size=1)
        said = caplog.text
        assert "prefix beam search" in said and "not best-path" in said
        assert "best_path()" in said

        # said when the model is built, not when it is used: a loop over a
        # test set would otherwise print it once per utterance
        assert said.count("prefix beam search") == 1
        for _ in range(3):
            s2t(np.random.randn(1000))
        assert caplog.text.count("prefix beam search") == 1

    caplog.clear()
    with caplog.at_level(logging.INFO):
        Speech2Text(s2t_train_config=s2t_config_file, ctc_weight=0.3)
    # a search that uses the decoder is not the one being confused with
    # best path, so it says nothing
    assert "prefix beam search" not in caplog.text


def test_decode_window_asks_the_checkpoint_which_way(s2t_config_file):
    """The kind of checkpoint decides how a window is read, not the caller.

    A CTC-only model is read off its head; one with a decoder is called,
    because its CTC branch answers what that branch was trained on rather
    than what the task symbol asks for. `espnet phonemize` and the browser
    demo both had to know this, and it is the checkpoint that knows.
    """
    speech2text = Speech2Text(s2t_train_config=s2t_config_file, beam_size=1)
    speech = np.random.randn(2000)
    assert speech2text.ctc_only is False

    called = []
    speech2text.best_path = lambda *a, **k: called.append("head") or [("ctc",)]

    # a decoder to read the task with: the object, not the head
    text = speech2text.decode_window(speech, lang_sym="<eng>", task_sym="<asr>")
    assert isinstance(text, str) and called == []

    # and without one, the head
    speech2text.ctc_only = True
    assert speech2text.decode_window(speech) == "ctc"
    assert called == ["head"]


def test_iter_long_yields_each_utterance_as_it_is_decoded(s2t_long_form_config_file):
    """decode_long is iter_long's list; iter_long hands out an utterance per window.

    The fixture's window is one second, spanned by <0.00> and <1.00>; the
    stand-in decoder answers every window with "a" from its start to its end,
    so a 2.5 s recording is three windows and three utterances, and the
    first is out before the second window is decoded.
    """
    speech2text = Speech2Text(s2t_train_config=s2t_long_form_config_file, beam_size=1)
    ids = speech2text.converter.token2id
    tokens = [ids["<eng>"], ids["<asr>"], ids["<0.00>"], ids["a"], ids["<1.00>"]]
    windows = []

    def decoder(**kwargs):
        windows.append(len(kwargs["speech"]))
        return [("a", ["a"], list(tokens), "a", None)]

    speech2text.__call__ = decoder
    speech = np.zeros(5000)

    pieces = speech2text.iter_long(speech, lang_sym="<eng>", task_sym="<asr>")
    assert next(pieces) == (0.0, 1.0, "a")
    assert windows == [2000]  # the second window is not decoded yet
    assert list(pieces) == [(1.0, 2.0, "a"), (2.0, 3.0, "a")]
    assert windows == [2000, 2000, 1000]

    assert speech2text.decode_long(speech, lang_sym="<eng>", task_sym="<asr>") == [
        (0.0, 1.0, "a"),
        (1.0, 2.0, "a"),
        (2.0, 3.0, "a"),
    ]


def test_best_path_takes_one_utterance(s2t_config_file):
    speech2text = Speech2Text(s2t_train_config=s2t_config_file, beam_size=1)
    with pytest.raises(ValueError, match="one utterance"):
        speech2text.best_path(np.random.randn(2, 1000))


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


@pytest.fixture()
def ctc_config_file(tmp_path: Path, token_list, request):
    input_layer, window = getattr(request, "param", ("conv2d8", 4))
    ctc_tokens = tmp_path / "ctc_tokens.txt"
    ctc_tokens.write_text(token_list.read_text() + "<nolang>\n")
    S2TCTCTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "ctc"),
            "--token_list",
            str(ctc_tokens),
            "--token_type",
            "char",
            "--promptencoder_conf",
            "output_size=4",
            "--preprocessor_conf",
            "fs=16000",
            "--preprocessor_conf",
            f"speech_length={window}",
            "--frontend_conf",
            "fs=16k",
            "--frontend_conf",
            "hop_length=160",
            "--encoder_conf",
            f"input_layer={input_layer}",
        ]
    )
    return tmp_path / "ctc" / "config.yaml"


@pytest.mark.parametrize("context", [0.56, 0.8, 1.6])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_ctc_log_probs_row_i_is_the_audio_at_frame_i(
    ctc_config_file, monkeypatch, context, batch_size
):
    """The language and task positions are not audio, so they are not frames.

    The encoder puts their embeddings in front of the frames. Counted as
    frames, they shift every row by the prefix length, which `ForcedAligner`
    reports as time, and drop that many frames from the recording.
    """
    speech2text = Speech2Text(s2t_train_config=ctc_config_file)
    # a 4 s window with 0.8 s of context is a whole number of frames throughout
    monkeypatch.setitem(speech2text.preprocessor_conf, "speech_length", 4)
    hop = round(speech2text.sample_rate / speech2text.frames_per_sec)

    def encode(speech, prefix, **kwargs):
        # one position per prefix symbol, then one per frame holding the first
        # sample of that frame; one frame short, as the convolutions are
        frames = speech[:, ::hop][:, :-1].unsqueeze(-1)
        marks = speech.new_full((speech.size(0), prefix.size(1), 1), -1.0)
        return torch.cat([marks, frames], dim=1), None

    monkeypatch.setattr(speech2text.s2t_model, "encode", encode)
    monkeypatch.setattr(speech2text.s2t_model.ctc, "log_softmax", lambda enc: enc)

    # every sample says where it is; 0 is what the padding holds
    seconds = 10
    speech = np.arange(1, seconds * speech2text.sample_rate + 1, dtype=np.float32)
    probs = speech2text.ctc_log_probs(
        speech, batch_size=batch_size, context_len_in_secs=context
    )

    assert len(probs) == round(seconds * speech2text.frames_per_sec)
    assert probs[:, 0].tolist() == (1 + hop * np.arange(len(probs))).tolist()


@pytest.mark.parametrize(
    "context,window,argument",
    [
        (0.5, 4, "context_len_in_secs"),
        (0.8, 3, "speech_length"),
    ],
)
def test_ctc_log_probs_rejects_fractional_frames(
    ctc_config_file, context, window, argument
):
    """A sample step between encoder frames would accumulate timing drift."""
    speech2text = Speech2Text(s2t_train_config=ctc_config_file)
    speech2text.preprocessor_conf["speech_length"] = window
    with pytest.raises(ValueError, match=argument + ".*whole number"):
        speech2text.ctc_log_probs(
            np.zeros(16000, dtype=np.float32), context_len_in_secs=context
        )


@pytest.mark.parametrize("context", [-0.8, 0, 2, 2.4])
def test_ctc_log_probs_rejects_missing_chunk_frames(ctc_config_file, context):
    """Refuse empty steps and contexts too short to preserve complete chunks."""
    speech2text = Speech2Text(s2t_train_config=ctc_config_file)
    with pytest.raises(ValueError, match="context"):
        speech2text.ctc_log_probs(
            np.zeros(16000, dtype=np.float32), context_len_in_secs=context
        )


@pytest.mark.parametrize("ctc_config_file", [("conv2d6", 30)], indirect=True)
def test_conv2d6_default_context_keeps_contiguous_frames(ctc_config_file, monkeypatch):
    """Omitted two-second context becomes 33 frames, without changing the stride."""
    speech2text = Speech2Text(s2t_train_config=ctc_config_file)
    hop = 960  # 160-sample frontend hop, subsampled six times

    def encode(speech, prefix, **kwargs):
        frames = speech[:, ::hop][:, :-1].unsqueeze(-1)
        marks = speech.new_full((speech.size(0), prefix.size(1), 1), -1.0)
        return torch.cat([marks, frames], dim=1), None

    monkeypatch.setattr(speech2text.s2t_model, "encode", encode)
    monkeypatch.setattr(speech2text.s2t_model.ctc, "log_softmax", lambda enc: enc)
    speech = np.arange(1, 70 * speech2text.sample_rate + 1, dtype=np.float32)
    probs = speech2text.ctc_log_probs(speech, batch_size=2)
    expected = 1 + hop * np.arange(round(len(speech) / hop))
    np.testing.assert_array_equal(probs[:, 0], expected)
    # The decoding wrapper also preserves an omitted context; one output
    # column makes every best-path label blank.
    assert speech2text.decode_long(speech, batch_size=2) == [(0.0, 70.0, "")]
    for decode in (speech2text.ctc_log_probs, speech2text.decode_long):
        with pytest.raises(ValueError, match="context_len_in_secs.*whole number"):
            decode(speech, context_len_in_secs=2)


def test_Speech2Text_batch_decode_accepts_numpy_and_lists(s2t_config_file_transformer):
    """A padded numpy array or a list of utterances decodes like the tensor."""
    batched = Speech2Text(
        s2t_train_config=s2t_config_file_transformer,
        beam_size=2,
        batch_size=3,
        maxlenratio=-8,
    )
    lengths = [4000, 3000, 9000]
    speeches = [np.random.randn(n).astype(np.float32) for n in lengths]
    padded = np.zeros((len(speeches), max(lengths)), dtype=np.float32)
    for i, sp in enumerate(speeches):
        padded[i, : len(sp)] = sp
    expected = batched.batch_decode(torch.tensor(padded), torch.tensor(lengths))

    from_numpy = batched.batch_decode(padded, np.array(lengths))
    from_list = batched.batch_decode(speeches)
    for actual in (from_numpy, from_list):
        assert len(actual) == len(expected)
        for exp, act in zip(expected, actual):
            assert [e[1] for e in exp] == [a[1] for a in act]
            np.testing.assert_allclose(
                float(exp[0][4].score), float(act[0][4].score), rtol=1e-4
            )
