from distutils.version import LooseVersion

import torch

from espnet2.asr.frontend.s3prl import S3prlFrontend

is_torch_1_7_plus = LooseVersion(torch.__version__) >= LooseVersion("1.7.0")


def test_frontend_init():
    if not is_torch_1_7_plus:
        return

    frontend = S3prlFrontend(
        fs=16000,
        frontend_conf=dict(upstream="mel"),
    )
    assert frontend.frontend_type == "s3prl"
    assert frontend.output_dim > 0


def test_frontend_output_size():
    # Skip some testing cases
    if not is_torch_1_7_plus:
        return

    frontend = S3prlFrontend(
        fs=16000,
        frontend_conf=dict(upstream="mel"),
        download_dir="./hub",
    )

    wavs = torch.randn(2, 1600)
    lengths = torch.LongTensor([1600, 1600])
    feats, _ = frontend(wavs, lengths)
    assert feats.shape[-1] == frontend.output_dim


def test_frontend_backward():
    if not is_torch_1_7_plus:
        return

    frontend = S3prlFrontend(
        fs=16000,
        frontend_conf=dict(upstream="mel"),
        download_dir="./hub",
    )
    wavs = torch.randn(2, 1600, requires_grad=True)
    lengths = torch.LongTensor([1600, 1600])
    feats, f_lengths = frontend(wavs, lengths)
    feats.sum().backward()
