import torch

from espnet2.asr.frontend.fused import FusedFrontends

frontend1 = {"frontend_type": "default", "n_mels": 80, "n_fft": 512}
frontend2 = {"frontend_type": "default", "hop_length": 128}

list_frontends = [frontend1, frontend2]


def test_frontend_init():
    frontend = FusedFrontends(
        fs=16000,
        align_method="linear_projection",
        proj_dim=100,
        frontends=list_frontends,
    )
    assert len(frontend.frontends) == 2
    assert len(frontend.factors) == len(frontend.frontends)
    assert frontend.frontends[0].frontend_type == "default"


def test_frontend_output_size():
    frontend = FusedFrontends(
        fs=16000,
        align_method="linear_projection",
        proj_dim=100,
        frontends=list_frontends,
    )
    assert frontend.output_size() == 100 * len(list_frontends)


def test_frontend_backward():
    frontend = FusedFrontends(
        fs=16000,
        align_method="linear_projection",
        proj_dim=100,
        frontends=list_frontends,
    )
    x = torch.randn(2, 300, requires_grad=True)
    x_lengths = torch.LongTensor([300, 89])
    y, y_lengths = frontend(x, x_lengths)
    y.sum().backward()


def test_frontend_lengths_do_not_depend_on_batch():
    frontend = FusedFrontends(
        fs=16000,
        align_method="linear_projection",
        proj_dim=16,
        frontends=[
            {"frontend_type": "default", "n_mels": 20, "hop_length": 128},
            {"frontend_type": "default", "n_mels": 20, "hop_length": 256},
        ],
    )
    frontend.eval()
    x = torch.randn(2, 16000)
    x[1, 8000:] = 0
    x_lengths = torch.LongTensor([16000, 8000])
    y, y_lengths = frontend(x, x_lengths)
    assert y.shape[1] == y_lengths[0]
    for i in range(2):
        _, alone = frontend(x[i : i + 1, : x_lengths[i]], x_lengths[i : i + 1])
        assert alone.tolist() == y_lengths[i : i + 1].tolist()
