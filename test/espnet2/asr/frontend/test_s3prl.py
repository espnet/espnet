import pytest
import torch
from packaging.version import parse as V

from espnet2.asr.frontend.s3prl import S3prlFrontend

is_torch_1_8_plus = V(torch.__version__) >= V("1.8.0")


def test_frontend_init():
    if not is_torch_1_8_plus:
        return

    frontend = S3prlFrontend(
        fs=16000,
        frontend_conf=dict(upstream="mel"),
    )
    assert frontend.frontend_type == "s3prl"
    assert frontend.output_size() > 0


def test_frontend_output_size():
    # Skip some testing cases
    if not is_torch_1_8_plus:
        return

    frontend = S3prlFrontend(
        fs=16000,
        frontend_conf=dict(upstream="mel"),
        download_dir="./hub",
    )

    wavs = torch.randn(2, 1600)
    lengths = torch.LongTensor([1600, 1600])
    feats, _ = frontend(wavs, lengths)
    assert feats.shape[-1] == frontend.output_size()


@pytest.mark.parametrize(
    "fs, frontend_conf, multilayer_feature",
    "layer"[
        (16000, dict(upstream="mel"), True, -1),
        (16000, dict(upstream="mel"), False, -1),
        (16000, dict(upstream="mel", tile_factor=1), False, -1),
        (16000, dict(upstream="mel"), False, 0),
    ],
)
def test_frontend_backward(fs, frontend_conf, multilayer_feature, layer):
    if not is_torch_1_8_plus:
        return

    frontend = S3prlFrontend(
        fs=fs,
        frontend_conf=frontend_conf,
        download_dir="./hub",
        multilayer_feature=multilayer_feature,
    )
    wavs = torch.randn(2, 1600, requires_grad=True)
    lengths = torch.LongTensor([1600, 1600])
    feats, f_lengths = frontend(wavs, lengths)
    feats.sum().backward()
