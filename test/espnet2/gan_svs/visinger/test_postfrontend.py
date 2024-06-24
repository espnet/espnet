import pytest
import torch
from packaging.version import parse as V

from espnet2.gan_svs.post_frontend.fused import S3prlPostFrontend

is_torch_1_8_plus = V(torch.__version__) >= V("1.8.0")


@pytest.mark.skipif(not is_torch_1_8_plus, reason="Not supported")
def test_frontend_init():
    frontend = S3prlPostFrontend(
        fs=16000,
        input_fs=24000,
        postfrontend_conf=dict(upstream="mel"),
    )
    assert frontend.frontend_type == "s3prl"
    assert frontend.output_size() > 0


@pytest.mark.skipif(not is_torch_1_8_plus, reason="Not supported")
def test_frontend_output_size():
    frontend = S3prlPostFrontend(
        fs=16000,
        input_fs=24000,
        postfrontend_conf=dict(upstream="mel"),
        download_dir="./hub",
    )

    wavs = torch.randn(2, 1600)
    lengths = torch.LongTensor([1600, 1600])
    feats, _ = frontend(wavs, lengths)
    assert feats.shape[-1] == frontend.output_size()


@pytest.mark.skipif(not is_torch_1_8_plus, reason="Not supported")
@pytest.mark.parametrize(
    "fs, input_fs, postfrontend_conf, multilayer_feature, layer",
    [
        (16000, 24000, dict(upstream="mel"), True, -1),
        (16000, 24000, dict(upstream="mel"), False, -1),
        (16000, 24000, dict(upstream="mel", tile_factor=1), False, -1),
        (16000, 24000, dict(upstream="mel"), False, 0),
    ],
)
def test_frontend_backward(fs, input_fs, postfrontend_conf, multilayer_feature, layer):
    frontend = S3prlPostFrontend(
        fs=fs,
        input_fs=input_fs,
        postfrontend_conf=postfrontend_conf,
        download_dir="./hub",
        multilayer_feature=multilayer_feature,
        layer=layer,
    )
    wavs = torch.randn(2, 1600, requires_grad=True)
    lengths = torch.LongTensor([1600, 1600])
    feats, f_lengths = frontend(wavs, lengths)
    feats.sum().backward()
