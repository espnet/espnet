import pytest
import torch

from espnet2.asr.frontend.default import DefaultFrontend

random_rir = torch.tensor(
    [
        [0.0291, -0.0318, 0.0325, -0.0298, 0.0216, -0.0057, -0.0263, 0.0989],
        [-0.0564, 0.1428, -0.2752, 0.4941, -0.5598, 1.2620, 0.6407, -1.7159],
    ]
)


def test_frontend_repr():
    frontend = DefaultFrontend(fs="16k")
    print(frontend)


def test_frontend_output_size():
    frontend = DefaultFrontend(fs="16k", n_mels=40)
    assert frontend.output_size() == 40


def test_frontend_backward():
    frontend = DefaultFrontend(
        fs=160, n_fft=128, win_length=32, hop_length=32, frontend_conf=None
    )
    x = torch.randn(2, 300, requires_grad=True)
    x_lengths = torch.LongTensor([300, 89])
    y, y_lengths = frontend(x, x_lengths)
    y.sum().backward()


@pytest.mark.parametrize("use_wpe", [True, False])
@pytest.mark.parametrize("use_beamformer", [True, False])
@pytest.mark.parametrize("train", [True, False])
def test_frontend_backward_multi_channel(train, use_wpe, use_beamformer):
    torch.random.manual_seed(14)
    frontend = DefaultFrontend(
        fs=300,
        n_fft=128,
        win_length=128,
        frontend_conf={"use_wpe": use_wpe, "use_beamformer": use_beamformer},
    )
    for name, p in frontend.named_parameters():
        if ".bias" in name and p.dim() == 1:
            p.data.fill_(1)
        else:
            p.data.zero_()
    if train:
        frontend.train()
    else:
        frontend.eval()
    pad = random_rir.size(1)
    envelope = torch.sin(torch.linspace(0, 20 * torch.pi, 1024)) + 1.5
    x = torch.sin(envelope * 2 * torch.pi * torch.arange(1024) / 300)
    x = torch.nn.functional.pad(x.repeat(2, 1, 1), (pad, pad))
    x = torch.nn.functional.conv1d(x, random_rir.unsqueeze(1)).transpose(1, 2)
    x = x[:, (pad - 1) // 2 : (pad - 1) // 2 + 1024]
    x.requires_grad = True
    x_lengths = torch.LongTensor([1024, 1000])
    y, y_lengths = frontend(x, x_lengths)
    y.sum().backward()
