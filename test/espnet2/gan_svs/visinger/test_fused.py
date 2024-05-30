import torch

from espnet2.gan_svs.post_frontend.fused import FusedPostFrontends

frontend1 = {
    "postfrontend_type": "s3prl",
    "postfrontend_conf": {"upstream": "mel"},
    "download_dir": "./hub",
    "multilayer_feature": True,
}

frontend2 = {
    "postfrontend_type": "s3prl",
    "postfrontend_conf": {"upstream": "mel"},
    "download_dir": "./hub",
    "multilayer_feature": True,
}

list_frontends = [frontend1, frontend2]


def test_frontend_init():
    frontend = FusedPostFrontends(
        fs="16k",
        input_fs="24k",
        align_method="linear_projection",
        proj_dim=100,
        postfrontends=list_frontends,
    )
    assert len(frontend.postfrontends) == 2
    assert len(frontend.factors) == len(frontend.postfrontends)


def test_frontend_output_size():
    frontend = FusedPostFrontends(
        fs="16k",
        input_fs="24k",
        align_method="linear_projection",
        proj_dim=100,
        postfrontends=list_frontends,
    )
    assert frontend.output_size() == 100 * len(list_frontends)


def test_frontend_backward():
    frontend = FusedPostFrontends(
        fs="16k",
        input_fs="24k",
        align_method="linear_projection",
        proj_dim=100,
        postfrontends=list_frontends,
    )
    x = torch.randn(2, 1600, requires_grad=True)
    x_lengths = torch.LongTensor([1600, 800])
    y, y_lengths = frontend(x, x_lengths)
    y.sum().backward()
