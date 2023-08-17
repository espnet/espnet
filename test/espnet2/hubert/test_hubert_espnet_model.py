import pytest
import torch
from packaging.version import parse as V

from espnet2.asr.encoder.hubert_encoder import TorchAudioHuBERTPretrainEncoder
from espnet2.hubert.espnet_model import TorchAudioHubertPretrainModel

is_torch_1_12_1_plus = V(torch.__version__) >= V("1.12.1")


@pytest.mark.parametrize("finetuning", [False])
def test_forward_backward_finetuning_false(finetuning):
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
        num_classes=5,
        final_dim=10,
        finetuning=finetuning,
    )

    model = TorchAudioHubertPretrainModel(
        5,
        token_list=["0", "1", "2", "3", "4"],
        frontend=None,
        specaug=None,
        normalize=None,
        preencoder=None,
        encoder=encoder,
    )

    inputs = dict(
        speech=torch.randn(2, 32, requires_grad=True),
        speech_lengths=torch.tensor([32, 16], dtype=torch.long),
        text=torch.randint(0, 5, [2, 15], dtype=torch.long),
        text_lengths=torch.tensor([15, 7], dtype=torch.long),
    )
    loss, *_ = model(**inputs)
    loss.backward()
