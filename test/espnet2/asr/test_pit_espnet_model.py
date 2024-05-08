import pytest
import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.transformer_encoder_multispkr import TransformerEncoder
from espnet2.asr.pit_espnet_model import ESPnetASRModel


@pytest.mark.parametrize("encoder_arch", [TransformerEncoder])
@pytest.mark.parametrize("num_inf", [1, 2, 3])
def test_pit_espnet_model(encoder_arch, num_inf):
    vocab_size = 5
    enc_out = 4
    encoder = encoder_arch(
        20,
        output_size=enc_out,
        linear_units=4,
        num_blocks=1,
        num_blocks_sd=1,
        num_inf=num_inf,
    )
    decoder = TransformerDecoder(vocab_size, enc_out, linear_units=4, num_blocks=2)
    ctc = CTC(odim=vocab_size, encoder_output_size=enc_out, reduce=False)

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
        num_inf=num_inf,
        num_ref=num_inf,
    )

    inputs = dict(
        speech=torch.randn(2, 10, 20, requires_grad=True),
        speech_lengths=torch.tensor([10, 8], dtype=torch.long),
        text=torch.randint(2, 4, [2, 4], dtype=torch.long),
        text_lengths=torch.tensor([4, 3], dtype=torch.long),
    )
    if num_inf > 1:
        for i in range(2, num_inf + 1):
            inputs[f"text_spk{i}"] = torch.randint(2, 4, [2, 4], dtype=torch.long)
            inputs[f"text_spk{i}_lengths"] = torch.tensor([4, 3], dtype=torch.long)

    loss, *_ = model(**inputs)
    loss.backward()
