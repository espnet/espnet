import pytest
import torch
import torch_complex
from packaging.version import parse as V
from torch_complex import ComplexTensor

from espnet2.enh.decoder.stft_decoder import STFTDecoder
from espnet2.enh.encoder.stft_encoder import STFTEncoder

is_torch_1_12_1_plus = V(torch.__version__) >= V("1.12.1")
is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


@pytest.mark.parametrize("n_fft", [512])
@pytest.mark.parametrize("win_length", [512])
@pytest.mark.parametrize("hop_length", [128])
@pytest.mark.parametrize("window", ["hann"])
@pytest.mark.parametrize("center", [True])
@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("onesided", [True, False])
@pytest.mark.parametrize("spec_transform_type", ["none", "exponent", "log"])
def test_STFTDecoder_backward(
    n_fft,
    win_length,
    hop_length,
    window,
    center,
    normalized,
    onesided,
    spec_transform_type,
):
    decoder = STFTDecoder(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
        center=center,
        normalized=normalized,
        onesided=onesided,
        spec_transform_type=spec_transform_type,
    )

    real = torch.rand(2, 300, n_fft // 2 + 1 if onesided else n_fft, requires_grad=True)
    imag = torch.rand(2, 300, n_fft // 2 + 1 if onesided else n_fft, requires_grad=True)
    x = ComplexTensor(real, imag)
    x_lens = torch.tensor([300 * hop_length, 295 * hop_length], dtype=torch.long)
    y, ilens = decoder(x, x_lens)
    y.sum().backward()


@pytest.mark.parametrize("n_fft", [512])
@pytest.mark.parametrize("win_length", [512])
@pytest.mark.parametrize("hop_length", [128])
@pytest.mark.parametrize("window", ["hann"])
@pytest.mark.parametrize("center", [True])
@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("onesided", [True, False])
def test_STFTDecoder_invalid_type(
    n_fft, win_length, hop_length, window, center, normalized, onesided
):
    decoder = STFTDecoder(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
        center=center,
        normalized=normalized,
        onesided=onesided,
    )
    with pytest.raises(TypeError):
        real = torch.rand(
            2, 300, n_fft // 2 + 1 if onesided else n_fft, requires_grad=True
        )
        x_lens = torch.tensor([300 * hop_length, 295 * hop_length], dtype=torch.long)
        y, ilens = decoder(real, x_lens)


@pytest.mark.parametrize("n_fft", [512])
@pytest.mark.parametrize("win_length", [512, 400])
@pytest.mark.parametrize("hop_length", [128, 256])
@pytest.mark.parametrize("onesided", [True, False])
def test_stft_enc_dec_streaming(n_fft, win_length, hop_length, onesided):
    input_audio = torch.randn((1, 16000))
    ilens = torch.LongTensor([16000])

    encoder = STFTEncoder(
        n_fft=n_fft, win_length=win_length, hop_length=hop_length, onesided=onesided
    )
    decoder = STFTDecoder(
        n_fft=n_fft, win_length=win_length, hop_length=hop_length, onesided=onesided
    )
    frames, flens = encoder(input_audio, ilens)
    wav, ilens = decoder(frames, ilens)

    splited = encoder.streaming_frame(input_audio)

    sframes = [encoder.forward_streaming(s) for s in splited]
    swavs = [decoder.forward_streaming(s) for s in sframes]
    merged = decoder.streaming_merge(swavs, ilens)

    if not (is_torch_1_9_plus and encoder.use_builtin_complex):
        sframes = torch_complex.cat(sframes, dim=1)
    else:
        sframes = torch.cat(sframes, dim=1)

    torch.testing.assert_close(sframes.real, frames.real)
    torch.testing.assert_close(sframes.imag, frames.imag)
    torch.testing.assert_close(wav, input_audio)
    torch.testing.assert_close(wav, merged)


@pytest.mark.skipif(not is_torch_1_12_1_plus, reason="torch.complex32 is used")
@pytest.mark.parametrize("n_fft", [512])
@pytest.mark.parametrize("win_length", [512])
@pytest.mark.parametrize("hop_length", [128])
@pytest.mark.parametrize("window", ["hann"])
@pytest.mark.parametrize("center", [True])
@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("onesided", [True, False])
def test_STFTDecoder_complex32_dtype(
    n_fft, win_length, hop_length, window, center, normalized, onesided
):
    decoder = STFTDecoder(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
        center=center,
        normalized=normalized,
        onesided=onesided,
    )
    x = torch.rand(
        2,
        300,
        n_fft // 2 + 1 if onesided else n_fft,
        dtype=torch.complex32,
        requires_grad=True,
    )
    x_lens = torch.tensor([300 * hop_length, 295 * hop_length], dtype=torch.long)
    y, ilens = decoder(x, x_lens)
    (y.real.pow(2) + y.imag.pow(2)).sum().backward()


def test_STFTDecoder_reconfig_for_fs():
    decoder = STFTDecoder(
        n_fft=512,
        win_length=512,
        hop_length=256,
        window="hann",
        center=True,
        normalized=False,
        onesided=True,
        default_fs=16000,
    )

    x = torch.rand(1, 32, 129, dtype=torch.complex64)
    ilens = torch.tensor([8448], dtype=torch.long)
    y_8k, _ = decoder(x, ilens, fs=8000)

    x = torch.rand(1, 32, 513, dtype=torch.complex64)
    y_32k, _ = decoder(x, ilens * 4, fs=32000)

    x = torch.rand(1, 32, 257, dtype=torch.complex64)
    y_16k, _ = decoder(x, ilens * 2)
