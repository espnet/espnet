import pytest
import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.e_branchformer_ctc_encoder import EBranchformerCTCEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.s2t.espnet_ctc_model import ESPnetS2TCTCModel


@pytest.mark.execution_timeout(5)
@pytest.mark.parametrize("encoder_arch", [EBranchformerCTCEncoder])
@pytest.mark.parametrize("prompt_encoder_arch", [TransformerEncoder])
def test_espnet_model(encoder_arch, prompt_encoder_arch):
    token_list = [
        "<blank>",
        "<unk>",
        "<na>",
        "<nolang>",
        "<eng>",
        "<asr>",
        "<st_eng>",
        "a",
        "<sos>",
        "<eos>",
        "<sop>",
    ]
    vocab_size = len(token_list)
    enc_out = 1
    encoder = encoder_arch(
        15,
        output_size=enc_out,
        attention_heads=1,
        attention_layer_type="selfattn",
        pos_enc_layer_type="abs_pos",
        linear_units=2,
        cgmlp_linear_units=2,
        num_blocks=2,
        cgmlp_conv_kernel=3,
        interctc_layer_idx=[1],
        interctc_use_conditioning=True,
        use_cross_attention=[False, True],
        use_flash_attn=False,
        dropout_rate=0,
        positional_dropout_rate=0,
        attention_dropout_rate=0,
    )
    prompt_encoder = prompt_encoder_arch(
        enc_out,
        attention_heads=1,
        output_size=enc_out,
        linear_units=2,
        num_blocks=1,
        input_layer=None,
        use_flash_attn=False,
        dropout_rate=0,
        positional_dropout_rate=0,
        attention_dropout_rate=0,
    )
    ctc = CTC(odim=vocab_size, encoder_output_size=enc_out)

    model = ESPnetS2TCTCModel(
        vocab_size,
        token_list=token_list,
        frontend=None,
        specaug=None,
        normalize=None,
        encoder=encoder,
        prompt_encoder=prompt_encoder,
        ctc=ctc,
        interctc_weight=0.5,
        ctc_asr_only=[True, False],
    )

    inputs = dict(
        speech=torch.randn(2, 16, 15, requires_grad=True),
        speech_lengths=torch.tensor([16, 16], dtype=torch.long),
        text=torch.randint(2, 4, [2, 4], dtype=torch.long),
        text_lengths=torch.tensor([4, 3], dtype=torch.long),
        text_prev=torch.tensor([[2], [7]], dtype=torch.long),
        text_prev_lengths=torch.tensor([1, 1], dtype=torch.long),
        text_ctc=torch.randint(2, 4, [2, 4], dtype=torch.long),
        text_ctc_lengths=torch.tensor([4, 3], dtype=torch.long),
        prefix=torch.tensor([[4, 5], [4, 6]], dtype=torch.long),
        prefix_lengths=torch.tensor([2, 2], dtype=torch.long),
    )
    loss, *_ = model(**inputs)
