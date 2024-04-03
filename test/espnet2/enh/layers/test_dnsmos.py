import urllib.request
from pathlib import Path

import numpy as np
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


@pytest.fixture()
def test_model_consistency(tmp_path: Path):
    url = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"  # noqa: E501
    urllib.request.urlretrieve(url, str(tmp_path / "sig_bak_ovr.onnx"))
    url = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/model_v8.onnx"  # noqa: E501
    urllib.request.urlretrieve(url, str(tmp_path / "model_v8.onnx"))

    dnsmos_th = DNSMOS_local(
        tmp_path / "sig_bak_ovr.onnx",
        tmp_path / "model_v8.onnx",
        use_gpu=False,
        convert_to_torch=True,
    )
    dnsmos_onnx = DNSMOS_local(
        tmp_path / "sig_bak_ovr.onnx",
        tmp_path / "model_v8.onnx",
        use_gpu=False,
        convert_to_torch=False,
    )
    fs = 32000
    speech = torch.randn(fs // 2)
    ret_th = dnsmos_th(speech.numpy(), fs)
    ret_onnx = dnsmos_onnx(speech, fs)
    for k in ret_th.keys():
        np.testing.assert_allclose(ret_th[k], ret_onnx[k])
