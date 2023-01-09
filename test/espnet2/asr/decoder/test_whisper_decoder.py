import pytest
import torch

from espnet2.asr.decoder.whisper_decoder import OpenAIWhisperDecoder

VOCAB_SIZE_WHISPER_MULTILINGUAL = 51865

@pytest.fixture()
def whisper_decoder(request):
    return OpenAIWhisperDecoder(
        vocab_size=VOCAB_SIZE_WHISPER_MULTILINGUAL,
        encoder_output_size=768,
    )

def test_decoder_init(whisper_decoder):
    assert (
        whisper_decoder.decoders.token_embedding.num_embeddings == 
        VOCAB_SIZE_WHISPER_MULTILINGUAL
    )

def test_decoder_reinit_emb():
    vocab_size = 1000
    decoder = OpenAIWhisperDecoder(
                        vocab_size=vocab_size,
                        encoder_output_size=768,
                    )
    assert (
        decoder.decoders.token_embedding.num_embeddings == 
        vocab_size
    )

def test_decoder_invalid_init():
    with pytest.raises(AssertionError):
        decoder = OpenAIWhisperDecoder(
                        vocab_size=VOCAB_SIZE_WHISPER_MULTILINGUAL,
                        encoder_output_size=768,
                        whisper_model="aaa"
                    )

def test_decoder_forward_backward(whisper_decoder):
    hs_pad = torch.randn(
        4, 100, 768,
        device=next(whisper_decoder.parameters()).device
    )
    ys_in_pad = torch.randint(
        0, 3000, (4, 10),
        device=next(whisper_decoder.parameters()).device
    )
    out, _ = whisper_decoder(hs_pad, None, ys_in_pad, None)

    assert (
        out.size() == 
        torch.Size([4, 10, VOCAB_SIZE_WHISPER_MULTILINGUAL])
    )
    out.sum().backward()

def test_decoder_scoring(whisper_decoder):
    hs_pad = torch.randn(
        4, 100, 768,
        device=next(whisper_decoder.parameters()).device
    )
    ys_in_pad = torch.randint(
        0, 3000, (4, 10),
        device=next(whisper_decoder.parameters()).device
    )
    out, _ = whisper_decoder.batch_score(ys_in_pad, None, hs_pad)

    assert (
        out.size() == 
        torch.Size([4, VOCAB_SIZE_WHISPER_MULTILINGUAL])
    )

    hs_pad = torch.randn(
        100, 768,
        device=next(whisper_decoder.parameters()).device
    )
    ys_in_pad = torch.randint(
        0, 3000, (10,),
        device=next(whisper_decoder.parameters()).device
    )
    out, _ = whisper_decoder.score(ys_in_pad, None, hs_pad)

    assert (
        out.size() == 
        torch.Size([VOCAB_SIZE_WHISPER_MULTILINGUAL])
    )
