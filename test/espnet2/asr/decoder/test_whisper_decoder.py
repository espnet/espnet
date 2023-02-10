import sys

import pytest
import torch
from packaging.version import parse as V

from espnet2.asr.decoder.whisper_decoder import OpenAIWhisperDecoder

VOCAB_SIZE_WHISPER_MULTILINGUAL = 51865

pytest.importorskip("whisper")

# NOTE(Shih-Lun): needed for `persistent` param in
#                 torch.nn.Module.register_buffer()
is_torch_1_7_plus = V(torch.__version__) >= V("1.7.0")
is_python_3_8_plus = sys.version_info >= (3, 8)


@pytest.mark.skipif(
    not is_python_3_8_plus or not is_torch_1_7_plus,
    reason="whisper not supported on python<3.8, torch<1.7",
)
@pytest.fixture()
def whisper_decoder(request):
    return OpenAIWhisperDecoder(
        vocab_size=VOCAB_SIZE_WHISPER_MULTILINGUAL,
        encoder_output_size=384,
        whisper_model="tiny",
    )


@pytest.mark.skipif(
    not is_python_3_8_plus or not is_torch_1_7_plus,
    reason="whisper not supported on python<3.8, torch<1.7",
)
@pytest.mark.timeout(50)
def test_decoder_init(whisper_decoder):
    assert (
        whisper_decoder.decoders.token_embedding.num_embeddings
        == VOCAB_SIZE_WHISPER_MULTILINGUAL
    )


@pytest.mark.skipif(
    not is_python_3_8_plus or not is_torch_1_7_plus,
    reason="whisper not supported on python<3.8, torch<1.7",
)
@pytest.mark.timeout(50)
def test_decoder_reinit_emb():
    vocab_size = 1000
    decoder = OpenAIWhisperDecoder(
        vocab_size=vocab_size,
        encoder_output_size=384,
        whisper_model="tiny",
    )
    assert decoder.decoders.token_embedding.num_embeddings == vocab_size


@pytest.mark.skipif(
    not is_python_3_8_plus or not is_torch_1_7_plus,
    reason="whisper not supported on python<3.8, torch<1.7",
)
def test_decoder_invalid_init():
    with pytest.raises(AssertionError):
        decoder = OpenAIWhisperDecoder(
            vocab_size=VOCAB_SIZE_WHISPER_MULTILINGUAL,
            encoder_output_size=384,
            whisper_model="aaa",
        )
        del decoder


@pytest.mark.skipif(
    not is_python_3_8_plus or not is_torch_1_7_plus,
    reason="whisper not supported on python<3.8, torch<1.7",
)
@pytest.mark.timeout(50)
def test_decoder_forward_backward(whisper_decoder):
    hs_pad = torch.randn(4, 100, 384, device=next(whisper_decoder.parameters()).device)
    ys_in_pad = torch.randint(
        0, 3000, (4, 10), device=next(whisper_decoder.parameters()).device
    )
    out, _ = whisper_decoder(hs_pad, None, ys_in_pad, None)

    assert out.size() == torch.Size([4, 10, VOCAB_SIZE_WHISPER_MULTILINGUAL])
    out.sum().backward()


@pytest.mark.skipif(
    not is_python_3_8_plus or not is_torch_1_7_plus,
    reason="whisper not supported on python<3.8, torch<1.7",
)
@pytest.mark.timeout(50)
def test_decoder_scoring(whisper_decoder):
    hs_pad = torch.randn(4, 100, 384, device=next(whisper_decoder.parameters()).device)
    ys_in_pad = torch.randint(
        0, 3000, (4, 10), device=next(whisper_decoder.parameters()).device
    )
    out, _ = whisper_decoder.batch_score(ys_in_pad, None, hs_pad)

    assert out.size() == torch.Size([4, VOCAB_SIZE_WHISPER_MULTILINGUAL])

    hs_pad = torch.randn(100, 384, device=next(whisper_decoder.parameters()).device)
    ys_in_pad = torch.randint(
        0, 3000, (10,), device=next(whisper_decoder.parameters()).device
    )
    out, _ = whisper_decoder.score(ys_in_pad, None, hs_pad)

    assert out.size() == torch.Size([VOCAB_SIZE_WHISPER_MULTILINGUAL])
