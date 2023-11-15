import pytest
import torch

from espnet2.enh.layers.dnsmos import DNSMOS_local


def test_audio_melspec_consistency():
    dnsmos_th = DNSMOS_local(None, None, use_gpu=False, convert_to_torch=True)
    dnsmos_onnx = DNSMOS_local(None, None, use_gpu=False, convert_to_torch=False)
    x = torch.randn(8000)
    audio_melspec_th = dnsmos_th.audio_melspec(x)
    audio_melspec_onnx = dnsmos_onnx.audio_melspec(x.numpy())
    torch.testing.assert_close(audio_melspec_th, torch.from_numpy(audio_melspec_onnx))


@pytest.mark.parametrize("is_personalized_MOS", [True, False])
def test_get_polyfit_val_consistency(is_personalized_MOS):
    dnsmos_th = DNSMOS_local(None, None, use_gpu=False, convert_to_torch=True)
    dnsmos_onnx = DNSMOS_local(None, None, use_gpu=False, convert_to_torch=False)
    sig, bak, ovr = torch.rand(3) * 5
    mos_sig1, mos_bak1, mos_ovr1 = dnsmos_th.get_polyfit_val(
        sig, bak, ovr, is_personalized_MOS
    )
    mos_sig2, mos_bak2, mos_ovr2 = dnsmos_onnx.get_polyfit_val(
        sig.item(), bak.item(), ovr.item(), is_personalized_MOS
    )
    torch.testing.assert_close(mos_sig1, torch.as_tensor(mos_sig2).float())
    torch.testing.assert_close(mos_bak1, torch.as_tensor(mos_bak2).float())
    torch.testing.assert_close(mos_ovr1, torch.as_tensor(mos_ovr2).float())
