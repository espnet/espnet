import pytest
import torch

from espnet2.asr.frontend.enh_frontend import EnhFrontend


@pytest.mark.parametrize(
    "enh_type, mask_type",
    [
        ("tf_masking", "IBM"),
        ("tf_masking", "IRM"),
        ("tf_masking", "IAM"),
        ("tf_masking", "PSM"),
        ("tf_masking", "NPSM"),
        ("tf_masking", "ICM"),
        ("tasnet", ""),
        ("wpe_beamformer", ""),
    ],
)
def test_frontend_repr(enh_type, mask_type):
    frontend = EnhFrontend(enh_type, mask_type, enh_conf=dict())
    print(frontend)


@pytest.mark.parametrize(
    "enh_type, mask_type", [("tasnet", "",),],
)
def test_time_domain_frontend_output_size(enh_type, mask_type):
    frontend = EnhFrontend(enh_type, mask_type, tf_factor=0, enh_conf=dict(),)
    x = torch.randn((3, 3000))
    input_lens = torch.IntTensor([3000, 3000, 3000])
    y, *__ = frontend(x, input_lens)
    assert y[0].shape == x.shape


@pytest.mark.parametrize(
    "enh_type, mask_type, tf_factor",
    [
        ("tf_masking", "IBM", 1),
        ("tf_masking", "IRM", 0.5),
        ("tf_masking", "IAM", 1),
        ("tf_masking", "PSM", 0.5),
        ("tf_masking", "NPSM", 1),
        ("tf_masking", "ICM", 1),
        ("tasnet", "", 0),
        ("wpe_beamformer", "", 0),
    ],
)
def test_frontend_backward(enh_type, mask_type, tf_factor):
    frontend = EnhFrontend(enh_type, mask_type, tf_factor=tf_factor, enh_conf=dict(),)
    x = torch.randn((3, 3000), requires_grad=True)
    input_lens = torch.IntTensor([3000, 3000, 3000])
    y, *__ = frontend(x, input_lens)
    y[0].sum().backward()


def test_frontend_backward_multi_channel():
    pass


def test_frontend_output_size_multi_channel():
    pass
