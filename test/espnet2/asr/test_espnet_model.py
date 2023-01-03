import pytest
import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.espnet_model import ESPnetASRModel


@pytest.mark.parametrize("encoder_arch", [TransformerEncoder, ConformerEncoder])
@pytest.mark.parametrize("aux_ctc", [None, {"0": "lid"}])
def test_espnet_model(encoder_arch, aux_ctc):
    vocab_size = 5
    enc_out = 4
    encoder = encoder_arch(20, output_size=enc_out, linear_units=4, num_blocks=2)
    decoder = TransformerDecoder(vocab_size, enc_out, linear_units=4, num_blocks=2)
    ctc = CTC(odim=vocab_size, encoder_output_size=enc_out)

    model = ESPnetASRModel(
        vocab_size,
        token_list=["<blank>", "<unk>", "a", "i", "<eos>"],
        frontend=None,
        specaug=None,
        normalize=None,
        preencoder=None,
        encoder=encoder,
        postencoder=None,
        decoder=decoder,
        ctc=ctc,
        joint_network=None,
        aux_ctc=aux_ctc,
    )

    inputs = dict(
        speech=torch.randn(2, 10, 20, requires_grad=True),
        speech_lengths=torch.tensor([10, 8], dtype=torch.long),
        text=torch.randint(2, 4, [2, 4], dtype=torch.long),
        text_lengths=torch.tensor([4, 3], dtype=torch.long),
    )
    loss, *_ = model(**inputs)
    loss.backward()
