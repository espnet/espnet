import pytest
import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.discrete_asr_espnet_model import ESPnetDiscreteASRModel
from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder
from espnet2.mt.frontend.embedding import Embedding


@pytest.mark.parametrize("input_layer_type", ["conv1d2", "conv1d3"])
def test_discrete_asr_espnet_model(input_layer_type):
    vocab_size = 5
    src_vocab_size = 4
    enc_out = 4
    frontend = Embedding(
        input_size=src_vocab_size,
        embed_dim=6,
        positional_dropout_rate=0,
    )
    encoder = EBranchformerEncoder(
        6,
        output_size=enc_out,
        linear_units=4,
        num_blocks=2,
        input_layer=input_layer_type,
    )
    decoder = TransformerDecoder(vocab_size, enc_out, linear_units=4, num_blocks=2)
    ctc = CTC(odim=vocab_size, encoder_output_size=enc_out)

    model = ESPnetDiscreteASRModel(
        vocab_size,
        src_vocab_size=src_vocab_size,
        token_list=["<blank>", "<unk>", "a", "i", "<eos>"],
        frontend=frontend,
        specaug=None,
        preencoder=None,
        encoder=encoder,
        postencoder=None,
        decoder=decoder,
        ctc=ctc,
    )

    inputs = dict(
        src_text=torch.randint(0, 3, [2, 10], dtype=torch.long),
        src_text_lengths=torch.tensor([10, 8], dtype=torch.long),
        text=torch.randint(2, 4, [2, 4], dtype=torch.long),
        text_lengths=torch.tensor([4, 3], dtype=torch.long),
    )
    loss, *_ = model(**inputs)
    loss.backward()
