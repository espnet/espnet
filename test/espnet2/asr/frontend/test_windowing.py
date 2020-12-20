import torch

from espnet2.asr.frontend.windowing import SlidingWindow


def test_frontend_output_size():
    win_length = 400
    frontend = SlidingWindow(win_length=win_length, hop_length=32, fs="16k")
    assert frontend.output_size() == win_length


def test_frontend_forward():
    frontend = SlidingWindow(fs=160, win_length=32, hop_length=32, padding=0)
    x = torch.randn(2, 300, requires_grad=True)
    x_lengths = torch.LongTensor([300, 89])
    y, y_lengths = frontend(x, x_lengths)
    y.sum().backward()
    # check for correct output lengths
    # needs change if padding applied!
    assert all(y_lengths == torch.tensor([9, 2]))
    assert y.shape == torch.Size([2, 9, 1, 32])
