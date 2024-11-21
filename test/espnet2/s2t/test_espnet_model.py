import pytest
import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.s2t.espnet_model import ESPnetS2TModel


@pytest.mark.parametrize("encoder_arch", [TransformerEncoder])
@pytest.mark.parametrize("decoder_arch", [TransformerDecoder])
def test_espnet_model(encoder_arch, decoder_arch):
    token_list = [
        "<blank>",
        "<unk>",
        "<na>",
        "<nospeech>",
        "<en>",
        "<asr>",
        "<st_en>",
        "<notimestamps>",
        "<0.00>",
        "<30.00>",
        "a",
        "i",
        "<sos>",
        "<eos>",
        "<sop>",
    ]
    vocab_size = len(token_list)
    enc_out = 4
    encoder = encoder_arch(
        20, output_size=enc_out, linear_units=4, num_blocks=2, use_flash_attn=False
    )
    decoder = decoder_arch(
        vocab_size, enc_out, linear_units=4, num_blocks=2, use_flash_attn=False
    )
    ctc = CTC(odim=vocab_size, encoder_output_size=enc_out)

    model = ESPnetS2TModel(
        vocab_size,
        token_list=token_list,
        frontend=None,
        specaug=None,
        normalize=None,
        preencoder=None,
        encoder=encoder,
        postencoder=None,
        decoder=decoder,
        ctc=ctc,
    )

    inputs = dict(
        speech=torch.randn(2, 10, 20, requires_grad=True),
        speech_lengths=torch.tensor([10, 8], dtype=torch.long),
        text=torch.randint(2, 4, [2, 4], dtype=torch.long),
        text_lengths=torch.tensor([4, 3], dtype=torch.long),
        text_prev=torch.randint(2, 4, [2, 4], dtype=torch.long),
        text_prev_lengths=torch.tensor([4, 3], dtype=torch.long),
        text_ctc=torch.randint(2, 4, [2, 4], dtype=torch.long),
        text_ctc_lengths=torch.tensor([4, 3], dtype=torch.long),
    )
    loss, *_ = model(**inputs)
    loss.backward()
