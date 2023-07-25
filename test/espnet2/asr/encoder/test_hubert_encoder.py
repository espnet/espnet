import pytest
import torch
from packaging.version import parse as V

from espnet2.asr.encoder.hubert_encoder import TorchAudioHuBERTPretrainEncoder

is_torch_1_12_1_plus = V(torch.__version__) >= V("1.12.1")


@pytest.mark.parametrize(
    "finetuning, eval, freeze_encoder_updates",
    [
        (False, False, 0),
        (True, False, 0),
        (True, False, 1),
        (True, True, 0),
    ],
)
def test_Encoder_forward_backward(finetuning, eval, freeze_encoder_updates):
    if not is_torch_1_12_1_plus:
        return

    encoder = TorchAudioHuBERTPretrainEncoder(
        20,
        extractor_conv_layer_config=[[3, 3, 2]],
        encoder_pos_conv_kernel=16,
        encoder_pos_conv_groups=4,
        encoder_embed_dim=4,
        encoder_num_layers=1,
        encoder_num_heads=1,
        encoder_ff_interm_features=4,
        num_classes=10,
        final_dim=10,
        finetuning=finetuning,
        freeze_encoder_updates=freeze_encoder_updates,
    )
    x = torch.randn(2, 32, requires_grad=True)
    y = torch.randint(low=0, high=10, size=(2, 15), dtype=torch.long)
    x_lens = torch.LongTensor([32, 16])
    y, _, p = encoder(x, x_lens, y)
    if not eval:
        encoder.train()
        if not finetuning:
            p.sum().backward()
        else:
            if freeze_encoder_updates == 0:
                y.sum().backward()
            else:
                y.sum()  # requires_grad=False if freezing
    else:
        encoder.eval()
        y, _, p = encoder(x, x_lens, y)
        y.sum()


def test_Encoder_output_size():
    if not is_torch_1_12_1_plus:
        return

    encoder = TorchAudioHuBERTPretrainEncoder(
        20,
        encoder_embed_dim=16,
        encoder_num_layers=1,
        encoder_num_heads=1,
        encoder_ff_interm_features=16,
        num_classes=10,
        final_dim=10,
    )
    assert encoder.output_size() == 16


def test_Encoder_reload_params():
    if not is_torch_1_12_1_plus:
        return

    encoder = TorchAudioHuBERTPretrainEncoder(
        20,
        encoder_embed_dim=16,
        encoder_num_layers=1,
        encoder_num_heads=1,
        encoder_ff_interm_features=16,
        num_classes=10,
        final_dim=10,
    )
    encoder.reload_pretrained_parameters()


def test_Encoder_invalid_type():
    if not is_torch_1_12_1_plus:
        return

    with pytest.raises(ValueError):
        TorchAudioHuBERTPretrainEncoder(
            20,
            encoder_embed_dim=10,
            encoder_num_layers=1,
            encoder_num_heads=1,
            encoder_ff_interm_features=10,
            num_classes=10,
            final_dim=10,
            freeze_encoder_updates=False,
        )
