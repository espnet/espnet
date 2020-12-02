from distutils.version import LooseVersion

import pytest
import torch

from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.enh.nets.beamformer_net import BeamformerNet
from espnet2.enh.nets.tasnet import TasNet
from espnet2.enh.nets.tf_mask_net import TFMaskingNet
from test.espnet2.enh.layers.test_enh_layers import random_speech

is_torch_1_2_plus = LooseVersion(torch.__version__) >= LooseVersion("1.2.0")


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("mask_type", ["IBM", "IRM", "IAM", "PSM", "PSM^2"])
@pytest.mark.parametrize(
    "loss_type", ["mask_mse", "magnitude", "spectrum", "spectrum_log"]
)
@pytest.mark.parametrize("num_spk", [1, 2, 3])
@pytest.mark.parametrize("use_noise_mask", [True, False])
@pytest.mark.parametrize("stft_consistency", [True, False])
def test_forward_with_beamformer_net(
    training, mask_type, loss_type, num_spk, use_noise_mask, stft_consistency
):
    # Skip some testing cases
    if not loss_type.startswith("mask") and mask_type != "IBM":
        # `mask_type` has no effect when `loss_type` is not "mask..."
        return

    ch = 2
    inputs = random_speech[..., :ch].float()
    ilens = torch.LongTensor([16, 12])
    speech_refs = [torch.randn(2, 16, ch).float() for spk in range(num_spk)]
    noise_ref1 = torch.randn(2, 16, ch, dtype=torch.float)
    dereverb_ref1 = torch.randn(2, 16, ch, dtype=torch.float)
    model = BeamformerNet(
        train_mask_only=True,
        mask_type=mask_type,
        loss_type=loss_type,
        n_fft=8,
        hop_length=2,
        num_spk=num_spk,
        use_wpe=True,
        wlayers=2,
        wunits=2,
        wprojs=2,
        use_dnn_mask_for_wpe=True,
        multi_source_wpe=True,
        use_beamformer=True,
        blayers=2,
        bunits=2,
        bprojs=2,
        badim=2,
        ref_channel=0,
        use_noise_mask=use_noise_mask,
        beamformer_type="mvdr_souden",
    )
    enh_model = ESPnetEnhancementModel(model, stft_consistency=stft_consistency)
    if training:
        enh_model.train()
        if stft_consistency and not is_torch_1_2_plus:
            # torchaudio.functional.istft is only available with pytorch 1.2+
            return
    else:
        enh_model.eval()
        if not is_torch_1_2_plus:
            # torchaudio.functional.istft is only available with pytorch 1.2+
            return

    kwargs = {
        "speech_mix": inputs,
        "speech_mix_lengths": ilens,
        **{"speech_ref{}".format(i + 1): speech_refs[i] for i in range(num_spk)},
        "noise_ref1": noise_ref1,
        "dereverb_ref1": dereverb_ref1,
    }
    loss, stats, weight = enh_model(**kwargs)


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("loss_type", ["si_snr"])
@pytest.mark.parametrize("num_spk", [1, 2, 3])
def test_forward_with_tasnet(training, loss_type, num_spk):
    inputs = torch.randn(2, 160)
    ilens = torch.LongTensor([160, 120])
    speech_refs = [torch.randn(2, 160).float() for spk in range(num_spk)]
    model = TasNet(
        N=5,
        L=20,
        B=5,
        H=10,
        P=3,
        X=8,
        R=4,
        num_spk=num_spk,
        loss_type=loss_type,
    )
    enh_model = ESPnetEnhancementModel(model)
    if training:
        enh_model.train()
    else:
        enh_model.eval()

    kwargs = {
        "speech_mix": inputs,
        "speech_mix_lengths": ilens,
        **{"speech_ref{}".format(i + 1): speech_refs[i] for i in range(num_spk)},
    }
    loss, stats, weight = enh_model(**kwargs)


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("mask_type", ["IBM", "IRM", "IAM", "PSM"])
@pytest.mark.parametrize(
    "loss_type", ["mask_mse", "magnitude", "spectrum", "spectrum_log"]
)
@pytest.mark.parametrize("num_spk", [1, 2, 3])
@pytest.mark.parametrize("stft_consistency", [True, False])
def test_forward_with_tf_mask_net(
    training, mask_type, loss_type, num_spk, stft_consistency
):
    inputs = torch.randn(2, 16).float()
    ilens = torch.LongTensor([16, 12])
    speech_refs = [torch.randn(2, 16).float() for spk in range(num_spk)]
    model = TFMaskingNet(
        n_fft=8,
        win_length=None,
        hop_length=2,
        rnn_type="blstm",
        layer=3,
        unit=8,
        dropout=0.0,
        num_spk=num_spk,
        mask_type=mask_type,
        loss_type=loss_type,
    )
    enh_model = ESPnetEnhancementModel(model, stft_consistency=stft_consistency)
    if training:
        enh_model.train()
        if stft_consistency and not is_torch_1_2_plus:
            # torchaudio.functional.istft is only available with pytorch 1.2+
            return
    else:
        enh_model.eval()
        if not is_torch_1_2_plus:
            # torchaudio.functional.istft is only available with pytorch 1.2+
            return

    kwargs = {
        "speech_mix": inputs,
        "speech_mix_lengths": ilens,
        **{"speech_ref{}".format(i + 1): speech_refs[i] for i in range(num_spk)},
    }
    loss, stats, weight = enh_model(**kwargs)
