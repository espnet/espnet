import torch

from espnet2.layers.sinc_conv import BarkScale, LogCompression, MelScale, SincConv


def test_log_compression():
    activation = LogCompression()
    x = torch.randn([5, 20, 1, 40], requires_grad=True)
    y = activation(x)
    assert x.shape == y.shape


def test_sinc_filters():
    filters = SincConv(
        in_channels=1, out_channels=128, kernel_size=101, stride=1, fs=16000
    )
    x = torch.randn([50, 1, 400], requires_grad=True)
    y = filters(x)
    assert y.shape == torch.Size([50, 128, 300])
    # now test multichannel
    filters = SincConv(
        in_channels=2, out_channels=128, kernel_size=101, stride=1, fs=16000
    )
    x = torch.randn([50, 2, 400], requires_grad=True)
    y = filters(x)
    assert y.shape == torch.Size([50, 128, 300])


def test_sinc_filter_static_functions():
    N = 400
    x = torch.linspace(1, N, N)
    print(f"no window function: {SincConv.none_window(x)}")
    print(f"hamming window function: {SincConv.hamming_window(x)}")
    SincConv.sinc(torch.tensor(50.0))


def test_sinc_filter_output_size():
    sinc_conv = SincConv(in_channels=1, out_channels=128, kernel_size=101)
    assert sinc_conv.get_odim(400) == 300


def test_bark_scale():
    f = 16000.0
    x = BarkScale.convert(f)
    f_back = BarkScale.invert(x)
    assert torch.abs(f_back - f) < 0.1
    BarkScale.bank(128, 16000.0)


def test_mel_scale():
    f = 16000.0
    x = MelScale.convert(f)
    f_back = MelScale.invert(x)
    assert torch.abs(f_back - f) < 0.1
    MelScale.bank(128, 16000.0)
