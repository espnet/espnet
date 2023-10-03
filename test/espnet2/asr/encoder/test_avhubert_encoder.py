import pytest
import torch

from espnet2.asr.encoder.avhubert_encoder import FairseqAVHubertEncoder


@pytest.mark.parametrize(
    "extracted, freeze_finetune_updates",
    [
        (False, 0),
        (False, 1),
        (True, 0),
        (True, 1),
    ],
)
def test_Encoder_forward_backward(extracted, freeze_finetune_updates):
    encoder = FairseqAVHubertEncoder(
        20,
        freeze_finetune_updates=freeze_finetune_updates,
        encoder_embed_dim=16,
        encoder_layers=2,
        encoder_ffn_embed_dim=64,
        encoder_attention_heads=2,
        extracted=extracted,
        pretrain=False,
    )
    if extracted:
        x = torch.randn(2, 10, 32, requires_grad=True)
        x_lens = torch.LongTensor([10, 5])
        encoder.train()
        y, _, _ = encoder(x, x_lens)
        if freeze_finetune_updates == 0:
            y.sum().backward()
        else:
            y.sum()  # requires_grad=False if freezing
    else:
        x = {
            "video": torch.randn(2, 1, 10, 88, 88),
            "audio": torch.randn(2, 26 * 4, 10),
        }
        x_lens = torch.LongTensor([10, 5])
        encoder.train()
        y, _, _ = encoder(x, x_lens)
        if freeze_finetune_updates == 0:
            y.sum().backward()
        else:
            y.sum()  # requires_grad=False if freezing


def test_Encoder_output_size():
    encoder = FairseqAVHubertEncoder(
        20,
        encoder_embed_dim=16,
        encoder_layers=2,
        encoder_ffn_embed_dim=64,
        encoder_attention_heads=2,
        extracted=True,
        pretrain=False,
    )
    assert encoder.output_size() == 16


def test_Encoder_reload_params():
    encoder = FairseqAVHubertEncoder(
        20,
        encoder_embed_dim=16,
        encoder_layers=2,
        encoder_ffn_embed_dim=64,
        encoder_attention_heads=2,
        extracted=True,
        pretrain=False,
    )
    encoder.reload_pretrained_parameters()


def test_Encoder_invalid_type():
    with pytest.raises(ValueError):
        FairseqAVHubertEncoder(
            20,
            encoder_embed_dim=10,
            encoder_layers=2,
            encoder_ffn_embed_dim=40,
            encoder_attention_heads=2,
            extracted=True,
            pretrain=False,
        )
