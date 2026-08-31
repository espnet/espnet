import pytest
import torch

from espnet3.systems.tts.f5_tts.vocoder_mel import VocoderMelSpec

FS = 24000
HOP = 256
N_MELS = 100


@pytest.fixture
def mel():
    return VocoderMelSpec(
        fs=FS, n_fft=1024, hop_length=HOP, win_length=1024, n_mels=N_MELS
    )


def test_output_size_is_the_mel_dimension(mel):
    """This is what F5TTS reads to size its backbone."""
    assert mel.output_size == N_MELS


def test_forward_returns_time_first_mel(mel):
    """F5TTS expects [B, T, n_mels], not the [B, n_mels, T] MelSpec returns."""
    feats, lengths = mel(torch.randn(2, FS))
    assert feats.shape[0] == 2
    assert feats.shape[2] == N_MELS
    assert lengths.shape == (2,)


def test_frame_count_matches_the_center_stft_formula(mel):
    n_samples = FS
    feats, lengths = mel(torch.randn(1, n_samples))
    expected = n_samples // HOP + 1
    assert feats.shape[1] == expected
    assert int(lengths[0]) == expected


def test_lengths_track_per_utterance_input_lengths(mel):
    input_lengths = torch.tensor([FS, FS // 2])
    _, lengths = mel(torch.randn(2, FS), input_lengths)
    assert int(lengths[0]) == FS // HOP + 1
    assert int(lengths[1]) == (FS // 2) // HOP + 1


def test_output_is_finite(mel):
    """A log-mel with a zeroed frame must not produce -inf."""
    wav = torch.randn(1, FS)
    wav[:, : HOP * 4] = 0.0
    feats, _ = mel(wav)
    assert torch.isfinite(feats).all()


def test_get_parameters_uses_espnet2_key_names(mel):
    """Downstream vocoder tooling expects espnet2's `n_shift`, not `hop_length`."""
    params = mel.get_parameters()
    assert params["n_shift"] == HOP
    assert "hop_length" not in params
    assert params["n_mels"] == N_MELS
    assert params["fs"] == FS
    assert params["mel_spec_type"] == "vocos"


def test_bigvgan_is_accepted_as_a_mel_type():
    spec = VocoderMelSpec(mel_spec_type="bigvgan")
    assert spec.output_size == 100


@pytest.mark.parametrize("mel_spec_type, extra_frame", [("vocos", 1), ("bigvgan", 0)])
def test_feats_lengths_match_the_frame_count_of_each_mel(mel_spec_type, extra_frame):
    """bigvgan runs center=False, so it yields one frame fewer than vocos.

    A padded batch is the case that matters: clamping to the batch width hides
    the discrepancy for the longest utterance only.
    """
    extract = VocoderMelSpec(mel_spec_type=mel_spec_type, hop_length=256)
    wav = torch.zeros(2, 12000)
    wav[1, :6000] = torch.randn(6000)

    feats, lengths = extract(wav, torch.tensor([12000, 6000]))

    assert feats.shape[1] == 12000 // 256 + extra_frame
    assert lengths.tolist() == [
        12000 // 256 + extra_frame,
        6000 // 256 + extra_frame,
    ]
