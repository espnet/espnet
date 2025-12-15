import pytest
import torch

from espnet2.asr.frontend.espnet_ssl import ESPnetSSLFrontend

mask_conf_1 = {}
mask_conf_2 = {"mask_length": 2}


@pytest.mark.parametrize("model", ["espnet/hubert_dummy"])
@pytest.mark.parametrize("masking_conf", [mask_conf_1, mask_conf_2])
@pytest.mark.parametrize("multilayer", [True, False])
@pytest.mark.parametrize("layer", [-1, 0])
@pytest.mark.parametrize("freeze_encoder_steps", [0, 10])
@pytest.mark.parametrize("mask_feats", [True, False])
@pytest.mark.parametrize("use_final_output", [True, False])
@pytest.mark.execution_timeout(30)
def test_frontend_backward(
    model,
    masking_conf,
    multilayer,
    layer,
    freeze_encoder_steps,
    mask_feats,
    use_final_output,
):

    frontend = ESPnetSSLFrontend(
        fs=16000,
        frontend_conf={"path_or_url": model},
        masking_conf=masking_conf,
        multilayer_feature=multilayer,
        layer=layer,
        freeze_encoder_steps=freeze_encoder_steps,
        mask_feats=mask_feats,
        use_final_output=use_final_output,
    )

    linear = torch.nn.Linear(frontend.output_size(), 2)

    test_length = 640
    wavs = torch.randn(2, test_length, requires_grad=True)
    lengths = torch.LongTensor([test_length, test_length])
    feats, f_lengths = frontend(wavs, lengths)
    feats = linear(feats)
    feats.sum().backward()
