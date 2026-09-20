"""Tests for the AMI SOT preprocessor in egs3/ami/s2t/src/preprocessor.py."""

import argparse
import importlib.util

import ami_sot_paths
import numpy as np
import pytest

pytest.importorskip("whisper")

_TOKENS = None


def _load():
    spec = importlib.util.spec_from_file_location(
        "ami_s2t_preprocessor", ami_sot_paths.RECIPE / "src" / "preprocessor.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pp = _load()


def _token_list(extra=()):
    """The Whisper vocabulary, plus any symbol the test appends to it.

    A symbol the vocabulary lacks is registered by the preprocessor at the next free id,
    which is one past this list unless the test extends it.
    """
    global _TOKENS
    if _TOKENS is None:
        from espnet2.text.whisper_token_id_converter import (
            OpenAIWhisperTokenIDConverter,
        )

        vocab = OpenAIWhisperTokenIDConverter(
            "whisper_multilingual", "en", task="transcribe"
        ).tokenizer.tokenizer.get_vocab()
        tokens = [None] * len(vocab)
        for token, idx in vocab.items():
            tokens[idx] = token
        _TOKENS = tokens
    return _TOKENS + list(extra)


def _build(token_list=None, **kwargs):
    options = dict(
        train=False,
        token_type="whisper_multilingual",
        token_list=token_list if token_list is not None else _token_list(),
        bpemodel="whisper_multilingual",
        notime_symbol="<|notimestamps|>",
        first_time_symbol="<|0.00|>",
        last_time_symbol="<|30.00|>",
        speech_length=30,
        speech_resolution=0.02,
        fs=16000,
        time_apply_prob=1.0,
    )
    options.update(kwargs)
    return pp.AmiSotS2TPreprocessor(**options)


_LINE = "<|en|><|transcribe|><|0.00|> one<|1.00|> ???? <|1.20|> two<|2.00|>"


def _run(preprocessor, text=_LINE):
    return preprocessor(
        "utt-0",
        {
            "speech": np.zeros(16000 * 5, dtype=np.float32),
            "text": text,
            "text_prev": "<|nospeech|>",
            "text_ctc": "<|nospeech|>",
        },
    )


@pytest.mark.execution_timeout(120)
def test_the_target_begins_with_the_language_and_task_symbols():
    """Training must produce the prefix decoding primes with."""
    tokens = _token_list()
    out = _run(_build())
    assert [tokens[i] for i in out["text"][:2]] == ["<|en|>", "<|transcribe|>"]


@pytest.mark.execution_timeout(120)
def test_the_separator_survives_as_one_token():
    """Without sot=True the BPE splits "????" into two ordinary tokens."""
    assert 25629 in _run(_build())["text"].tolist()


@pytest.mark.execution_timeout(120)
def test_no_notimestamps_is_injected_when_the_line_has_timestamps():
    """The base class prepends one; a target cannot say both."""
    assert 50363 not in _run(_build())["text"].tolist()


@pytest.mark.execution_timeout(120)
def test_speech_leaves_one_dimensional():
    """Speech leaves one dimensional.

    A frontend-less Whisper encoder hands the raw tensor to torch.stft, which rejects a
    trailing channel axis.
    """
    assert _run(_build())["speech"].ndim == 1


@pytest.mark.execution_timeout(120)
def test_a_configured_separator_other_than_the_default_is_honoured():
    tokens = _token_list(extra=["@@"])
    out = _run(
        _build(token_list=tokens, speaker_change_symbol="@@"),
        "<|en|><|transcribe|><|0.00|> one<|1.00|> @@ <|1.20|> two<|2.00|>",
    )
    assert "@@" in [tokens[i] for i in out["text"].tolist()]


@pytest.mark.execution_timeout(120)
def test_a_separator_the_token_list_disagrees_about_is_rejected():
    """A separator the token list disagrees about is rejected.

    A corpus and a vocabulary built with different separators is the silent failure this
    whole recipe is shaped around.

    sot=True makes any string tokenize to one id, so "is it one token" cannot catch it;
    only checking the token list can.
    """
    with pytest.raises(ValueError, match="token list holds"):
        # The list has no "@@" row, so the sot tokenizer puts it one past the end.
        _build(speaker_change_symbol="@@")


@pytest.mark.execution_timeout(600)
def test_a_target_from_this_preprocessor_gives_the_model_a_finite_loss():
    """The cheapest proof that the pieces fit: preprocessor -> model -> loss.

    Everything else in this suite checks one component. This one catches a mismatch
    between them, which is where the expensive surprises live.
    """
    import torch

    from espnet2.tasks.s2t import S2TTask

    tokens = _token_list()
    preprocessor = _build(train=True)
    batch = [
        _run(preprocessor, _LINE),
        _run(preprocessor, _LINE),
    ]

    args = S2TTask.get_default_config()
    args.update(
        token_list=list(tokens),
        frontend=None,
        input_size=1,
        specaug=None,
        normalize=None,
        encoder="whisper",
        encoder_conf={"whisper_model": "tiny", "dropout_rate": 0.0},
        decoder="whisper",
        decoder_conf={"whisper_model": "tiny"},
        model_conf={
            "ctc_weight": 0.0,
            "sym_blank": "<|translate|>",
            "sym_sos": "<|startoftranscript|>",
            "sym_eos": "<|endoftext|>",
            "sym_sop": "<|startofprev|>",
            "sym_na": "<|nospeech|>",
        },
    )
    model = S2TTask.build_model(argparse.Namespace(**args))

    def stack(key):
        values = [torch.tensor(sample[key]) for sample in batch]
        lengths = torch.tensor([len(v) for v in values])
        width = int(lengths.max())
        padded = torch.zeros(len(values), width, dtype=values[0].dtype)
        for i, value in enumerate(values):
            padded[i, : len(value)] = value
        return padded, lengths

    speech, speech_lengths = stack("speech")
    text, text_lengths = stack("text")
    text_prev, text_prev_lengths = stack("text_prev")
    text_ctc, text_ctc_lengths = stack("text_ctc")

    loss, _, _ = model(
        speech=speech,
        speech_lengths=speech_lengths,
        text=text,
        text_lengths=text_lengths,
        text_prev=text_prev,
        text_prev_lengths=text_prev_lengths,
        text_ctc=text_ctc,
        text_ctc_lengths=text_ctc_lengths,
    )
    assert torch.isfinite(loss), loss


@pytest.mark.execution_timeout(900)
def test_a_separator_the_vocabulary_lacks_grows_the_model_by_one_trainable_row():
    """The other half of the separator story, and the one with no loud failure.

    A symbol Whisper already has costs nothing. One it lacks is appended to the token
    list, so the model is sized one row wider than the pretrained embedding. With
    load_origin_token_embedding left at its default False that mismatch silently
    discards all 51865 pretrained rows; with it true the rows must survive and the
    appended one must still learn.
    """
    import torch
    import whisper

    from espnet2.tasks.s2t import S2TTask

    separator = "<sc>"
    tokens = _token_list(extra=[separator])
    assert len(tokens) == 51866

    preprocessor = _build(
        token_list=tokens, speaker_change_symbol=separator, train=True
    )
    assert preprocessor.speaker_change_id == 51865

    sample = _run(
        preprocessor,
        f"<|en|><|transcribe|><|0.00|> one<|1.00|> {separator} <|1.20|> two<|2.00|>",
    )
    assert sample["text"].tolist().count(51865) == 1

    args = S2TTask.get_default_config()
    args.update(
        token_list=list(tokens),
        frontend=None,
        input_size=1,
        specaug=None,
        normalize=None,
        encoder="whisper",
        encoder_conf={"whisper_model": "tiny", "dropout_rate": 0.0},
        decoder="whisper",
        decoder_conf={
            "whisper_model": "tiny",
            "load_origin_token_embedding": True,
        },
        model_conf={
            "ctc_weight": 0.0,
            "sym_blank": "<|translate|>",
            "sym_sos": "<|startoftranscript|>",
            "sym_eos": "<|endoftext|>",
            "sym_sop": "<|startofprev|>",
            "sym_na": "<|nospeech|>",
        },
    )
    model = S2TTask.build_model(argparse.Namespace(**args))

    embedding = model.decoder.decoders.token_embedding
    assert embedding.num_embeddings == 51866
    pretrained = whisper.load_model("tiny", device="cpu").decoder.token_embedding
    assert torch.equal(
        embedding.weight[:51865].float().cpu(), pretrained.weight.float().cpu()
    )
    # The appended row must not shadow a symbol the model resolves by index.
    assert 51865 not in (model.blank_id, model.sos, model.eos, model.sop, model.na)

    batch = [sample, sample]

    def stack(key):
        values = [torch.tensor(s[key]) for s in batch]
        lengths = torch.tensor([len(v) for v in values])
        padded = torch.zeros(len(values), int(lengths.max()), dtype=values[0].dtype)
        for i, value in enumerate(values):
            padded[i, : len(value)] = value
        return padded, lengths

    speech, speech_lengths = stack("speech")
    text, text_lengths = stack("text")
    text_prev, text_prev_lengths = stack("text_prev")
    text_ctc, text_ctc_lengths = stack("text_ctc")

    loss, _, _ = model(
        speech=speech,
        speech_lengths=speech_lengths,
        text=text,
        text_lengths=text_lengths,
        text_prev=text_prev,
        text_prev_lengths=text_prev_lengths,
        text_ctc=text_ctc,
        text_ctc_lengths=text_ctc_lengths,
    )
    assert torch.isfinite(loss), loss

    loss.backward()
    assert embedding.add_emb.weight.grad is not None
    assert embedding.add_emb.weight.grad.abs().sum() > 0
