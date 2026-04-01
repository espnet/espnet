import sys

import pytest
import torch
from packaging.version import parse as V

pytest.importorskip("whisper")

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.whisper_decoder import OpenAIWhisperDecoder
from espnet2.asr.encoder.whisper_encoder import OpenAIWhisperEncoder
from espnet2.asr.sot_espnet_model import SOTWhisperModel

is_torch_1_7_plus = V(torch.__version__) >= V("1.7.0")
is_python_3_8_plus = sys.version_info >= (3, 8)

VOCAB_SIZE = 51866  # 51865 base + 1 added token (<sc>)
BASE_VOCAB = 51865


def _make_token_list():
    """Generate a minimal token list for testing."""
    # Real token list has 51881 entries; for testing, use placeholders
    # but include real lowercase/uppercase pairs for uppercase loss testing
    tokens = [f"<token_{i}>" for i in range(BASE_VOCAB)]
    # Overwrite a few with real case pairs for min-CE loss testing
    tokens[64] = "a"  # lowercase
    tokens[65] = "A"  # uppercase
    tokens[66] = "hello"
    tokens[67] = "Hello"
    tokens[50257] = "<|endoftext|>"
    tokens[50258] = "<|startoftranscript|>"
    # Added token: <sc>
    tokens.append("<sc>")
    return tokens


@pytest.fixture(scope="module")
def sot_model():
    token_list = _make_token_list()
    encoder = OpenAIWhisperEncoder(whisper_model="tiny", dropout_rate=0.0)
    decoder = OpenAIWhisperDecoder(
        vocab_size=VOCAB_SIZE,
        encoder_output_size=encoder.output_size(),
        whisper_model="tiny",
        load_origin_token_embedding=True,
    )
    ctc = CTC(odim=VOCAB_SIZE, encoder_output_size=encoder.output_size())
    model = SOTWhisperModel(
        vocab_size=VOCAB_SIZE,
        token_list=token_list,
        frontend=None,
        specaug=None,
        normalize=None,
        preencoder=None,
        encoder=encoder,
        postencoder=None,
        decoder=decoder,
        ctc=ctc,
        joint_network=None,
        ctc_weight=0.0,
        use_uppercase_loss=True,
        sym_sos="<|startoftranscript|>",
        sym_eos="<|endoftext|>",
    )
    return model


@pytest.mark.skipif(
    not is_python_3_8_plus or not is_torch_1_7_plus,
    reason="whisper not supported on python<3.8, torch<1.7",
)
@pytest.mark.timeout(20)
def test_sot_model_init(sot_model):
    assert sot_model.vocab_size == VOCAB_SIZE
    assert sot_model.sos == 50258
    assert sot_model.eos == 50257
    assert sot_model.use_uppercase_loss is True
    assert len(sot_model.upper_cased_tokens) > 0


@pytest.mark.skipif(
    not is_python_3_8_plus or not is_torch_1_7_plus,
    reason="whisper not supported on python<3.8, torch<1.7",
)
@pytest.mark.timeout(20)
def test_sot_model_forward(sot_model):
    sot_model.train()
    speech = torch.randn(2, 16000)
    speech_lengths = torch.tensor([16000, 12000])
    text = torch.randint(2, 100, (2, 8))
    text_lengths = torch.tensor([8, 6])

    loss, stats, weight = sot_model(speech, speech_lengths, text, text_lengths)
    assert torch.isfinite(loss)
    assert "loss_att" in stats
    assert "acc" in stats


@pytest.mark.skipif(
    not is_python_3_8_plus or not is_torch_1_7_plus,
    reason="whisper not supported on python<3.8, torch<1.7",
)
@pytest.mark.timeout(20)
def test_sot_model_backward(sot_model):
    sot_model.train()
    speech = torch.randn(1, 16000)
    speech_lengths = torch.tensor([16000])
    text = torch.randint(2, 100, (1, 6))
    text_lengths = torch.tensor([6])

    loss, _, _ = sot_model(speech, speech_lengths, text, text_lengths)
    loss.backward()


@pytest.mark.skipif(
    not is_python_3_8_plus or not is_torch_1_7_plus,
    reason="whisper not supported on python<3.8, torch<1.7",
)
@pytest.mark.timeout(20)
def test_sot_model_no_uppercase_loss():
    token_list = _make_token_list()
    encoder = OpenAIWhisperEncoder(whisper_model="tiny", dropout_rate=0.0)
    decoder = OpenAIWhisperDecoder(
        vocab_size=VOCAB_SIZE,
        encoder_output_size=encoder.output_size(),
        whisper_model="tiny",
    )
    ctc = CTC(odim=VOCAB_SIZE, encoder_output_size=encoder.output_size())
    model = SOTWhisperModel(
        vocab_size=VOCAB_SIZE,
        token_list=token_list,
        frontend=None,
        specaug=None,
        normalize=None,
        preencoder=None,
        encoder=encoder,
        postencoder=None,
        decoder=decoder,
        ctc=ctc,
        joint_network=None,
        ctc_weight=0.0,
        use_uppercase_loss=False,
    )
    assert model.use_uppercase_loss is False
    assert len(model.upper_cased_tokens) == 0

    model.train()
    speech = torch.randn(1, 16000)
    speech_lengths = torch.tensor([16000])
    text = torch.randint(2, 100, (1, 6))
    text_lengths = torch.tensor([6])

    loss, _, _ = model(speech, speech_lengths, text, text_lengths)
    assert torch.isfinite(loss)
