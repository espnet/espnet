import pytest
import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.slu.espnet_model import ESPnetSLUModel
from espnet2.slu.postdecoder.hugging_face_transformers_postdecoder import (
    HuggingFaceTransformersPostDecoder,
)
from espnet2.slu.postencoder.conformer_postencoder import ConformerPostEncoder
from espnet2.slu.postencoder.transformer_postencoder import TransformerPostEncoder


@pytest.mark.parametrize("encoder_arch", [TransformerEncoder, ConformerEncoder])
@pytest.mark.parametrize(
    "encoder_del", [None, TransformerPostEncoder, ConformerPostEncoder]
)
@pytest.mark.parametrize("decoder_post", [None, HuggingFaceTransformersPostDecoder])
@pytest.mark.execution_timeout(50)
def test_asr_training(encoder_arch, encoder_del, decoder_post):
    vocab_size = 5
    enc_out = 20
    encoder = encoder_arch(
        20,
        output_size=enc_out,
        linear_units=4,
        num_blocks=2,
    )
    decoder = TransformerDecoder(
        vocab_size,
        enc_out,
        linear_units=4,
        num_blocks=2,
    )
    if encoder_del is not None:
        del_encoder = encoder_del(
            20,
            output_size=enc_out,
            linear_units=4,
        )
    else:
        del_encoder = None
    if decoder_post is not None:
        post_decoder = decoder_post(
            "bert-base-uncased",
            output_size=enc_out,
        )
    else:
        post_decoder = None
    ctc = CTC(odim=vocab_size, encoder_output_size=enc_out)

    model = ESPnetSLUModel(
        vocab_size,
        token_list=["<blank>", "<unk>", "a", "i", "<eos>"],
        transcript_token_list=["<blank>", "<unk>", "a", "i", "<eos>"],
        frontend=None,
        specaug=None,
        normalize=None,
        preencoder=None,
        encoder=encoder,
        postencoder=None,
        decoder=decoder,
        ctc=ctc,
        deliberationencoder=del_encoder,
        postdecoder=post_decoder,
        joint_network=None,
    )

    inputs = dict(
        speech=torch.randn(2, 10, 20, requires_grad=True),
        speech_lengths=torch.tensor([10, 8], dtype=torch.long),
        text=torch.randint(2, 4, [2, 4], dtype=torch.long),
        text_lengths=torch.tensor([4, 3], dtype=torch.long),
        transcript=torch.randint(2, 4, [2, 4], dtype=torch.long),
        transcript_lengths=torch.tensor([4, 3], dtype=torch.long),
    )
    loss, *_ = model(**inputs)
    loss.backward()
