import pytest
import torch
from torch_complex import ComplexTensor

from espnet2.bin.enh_inference_streaming import merge_audio, split_audio
from espnet2.enh.decoder.stft_decoder import STFTDecoder
from espnet2.enh.encoder.stft_encoder import STFTEncoder


@pytest.mark.parametrize("n_fft", [512])
@pytest.mark.parametrize("win_length", [512])
@pytest.mark.parametrize("hop_length", [128])
@pytest.mark.parametrize("window", ["hann"])
@pytest.mark.parametrize("center", [True])
@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("onesided", [True, False])
def test_STFTDecoder_backward(
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


def test_conv_enc_dec_streaming():
    input_audio = torch.randn((1, 16000))
    ilens = torch.LongTensor(
        [
            16000,
        ]
    )

    encoder = STFTEncoder(n_fft=256, hop_length=128, onesided=True)
    decoder = STFTDecoder(n_fft=256, hop_length=128, onesided=True)
    frames, flens = encoder(input_audio, ilens)
    seq_wav, ilens = decoder(frames, ilens)

    splited, rest = split_audio(input_audio, frame_size=256, hop_size=128)
    sframes = [encoder.forward_streaming(s) for s in splited]
    swavs = [decoder.forward_streaming(s) for s in sframes]
    merged_wav = merge_audio(swavs, 256, 128, rest)

    torch.testing.assert_allclose(seq_wav, merged_wav)
