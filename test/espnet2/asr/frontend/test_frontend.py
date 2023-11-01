import pytest
import torch

from espnet2.asr.frontend.default import DefaultFrontend

random_speech = torch.tensor(
    [
        [
            [0.026, 0.031],
            [0.027, 0.031],
            [0.026, 0.030],
            [0.024, 0.028],
            [0.025, 0.027],
            [0.027, 0.026],
            [0.028, 0.026],
            [0.029, 0.024],
            [0.028, 0.024],
            [0.029, 0.026],
            [0.029, 0.027],
            [0.029, 0.031],
            [0.030, 0.038],
            [0.029, 0.040],
            [0.028, 0.040],
            [0.028, 0.041],
        ],
        [
            [0.015, 0.021],
            [0.005, 0.034],
            [0.011, 0.029],
            [0.031, 0.036],
            [0.031, 0.031],
            [0.050, 0.038],
            [0.025, 0.054],
            [0.042, 0.056],
            [0.053, 0.048],
            [0.054, 0.044],
            [0.053, 0.036],
            [0.039, 0.025],
            [0.053, 0.032],
            [0.054, 0.034],
            [0.031, 0.025],
            [0.008, 0.023],
        ],
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
    frontend = DefaultFrontend(
        fs=300,
        n_fft=128,
        win_length=128,
        frontend_conf={"use_wpe": use_wpe, "use_beamformer": use_beamformer},
    )
    if train:
        frontend.train()
    else:
        frontend.eval()
    x = random_speech.repeat(1, 1024, 1)
    x.requires_grad = True
    x_lengths = torch.LongTensor([1024, 1000])
    y, y_lengths = frontend(x, x_lengths)
    y.sum().backward()
