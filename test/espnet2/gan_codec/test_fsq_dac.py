import numpy as np  # noqa
import pytest
import torch

from espnet2.gan_codec.dac.dac import DACDiscriminator
from espnet2.gan_codec.fsq_dac.fsq_dac import FSQDAC, CustomRoundFunc, DACGenerator
from espnet2.gan_codec.shared.loss.freq_loss import MultiScaleMelSpectrogramLoss
from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
)


def make_fsqdac_args(**kwargs):
    default = dict(
        sampling_rate=120,
        generator_params={
            "hidden_dim": 2,
            "encdec_channels": 1,
            "encdec_n_filters": 32,
            "encdec_n_residual_layers": 1,
            "encdec_ratios": [2, 2],
            "encdec_activation": "ELU",
            "encdec_activation_params": {"alpha": 1.0},
            "encdec_norm": "weight_norm",
            "encdec_norm_params": {},
            "encdec_kernel_size": 1,
            "encdec_residual_kernel_size": 1,
            "encdec_last_kernel_size": 1,
            "encdec_dilation_base": 2,
            "encdec_causal": False,
            "encdec_pad_mode": "reflect",
            "encdec_true_skip": False,
            "encdec_compress": 2,
            "encdec_lstm": 1,
            "decoder_trim_right_ratio": 1.0,
            "decoder_final_activation": None,
            "decoder_final_activation_params": None,
            "quantizer_codedim": 2,
            "quantizer_factor": 3,
        },
        discriminator_params={
            "msmpmb_discriminator_params": {
                "rates": [],
                "fft_sizes": [32, 16, 8],
                "sample_rate": 120,
                "periods": [2, 3, 5, 7, 11],
                "period_discriminator_params": {
                    "in_channels": 1,
                    "out_channels": 1,
                    "kernel_sizes": [5, 3],
                    "channels": 4,
                    "downsample_scales": [3, 1],
                    "max_downsample_channels": 16,
                    "bias": True,
                    "nonlinear_activation": "LeakyReLU",
                    "nonlinear_activation_params": {"negative_slope": 0.1},
                    "use_weight_norm": True,
                    "use_spectral_norm": False,
                },
                "band_discriminator_params": {
                    "hop_factor": 0.25,
                    "sample_rate": 120,
                    "bands": [
                        (0.0, 0.5),
                        (0.5, 1.0),
                    ],
                    "channel": 32,
                },
            },
        },
        use_dual_decoder=True,
        skip_quantizer_updates=0,
        lambda_quantization=1.0,
        lambda_reconstruct=1.0,
        lambda_adv=1.0,
        lambda_feat_match=2.0,
        lambda_mel=45.0,
    )
    default.update(kwargs)
    return default


def make_generator_args(**kwargs):
    default = dict(
        sample_rate=120,
        hidden_dim=2,
        encdec_channels=1,
        encdec_n_filters=32,
        encdec_n_residual_layers=1,
        encdec_ratios=[2, 2],
        encdec_activation="ELU",
        encdec_activation_params={"alpha": 1.0},
        encdec_norm="weight_norm",
        encdec_norm_params={},
        encdec_kernel_size=1,
        encdec_residual_kernel_size=1,
        encdec_last_kernel_size=1,
        encdec_dilation_base=2,
        encdec_causal=False,
        encdec_pad_mode="reflect",
        encdec_true_skip=False,
        encdec_compress=2,
        encdec_lstm=1,
        decoder_trim_right_ratio=1.0,
        decoder_final_activation=None,
        decoder_final_activation_params=None,
        quantizer_codedim=2,
        quantizer_factor=3,
    )
    default.update(kwargs)
    return default


def make_discriminator_args(**kwargs):
    defaults = dict(
        msmpmb_discriminator_params={
            "rates": [],
            "fft_sizes": [32, 16, 8],
            "sample_rate": 120,
            "periods": [2, 3, 5, 7, 11],
            "period_discriminator_params": {
                "in_channels": 1,
                "out_channels": 1,
                "kernel_sizes": [5, 3],
                "channels": 4,
                "downsample_scales": [3, 1],
                "max_downsample_channels": 16,
                "bias": True,
                "nonlinear_activation": "LeakyReLU",
                "nonlinear_activation_params": {"negative_slope": 0.1},
                "use_weight_norm": True,
                "use_spectral_norm": False,
            },
            "band_discriminator_params": {
                "hop_factor": 0.25,
                "sample_rate": 120,
                "bands": [
                    (0.0, 0.5),
                    (0.5, 1.0),
                ],
                "channel": 32,
            },
        },
    )
    defaults.update(kwargs)
    return defaults


def make_mel_loss_args(**kwargs):
    defaults = dict(
        fs=120,
        range_start=3,
        range_end=4,
        window="hann",
        n_mels=2,
        fmin=None,
        fmax=None,
        log_base=10.0,
    )
    defaults.update(kwargs)
    return defaults


@pytest.mark.skipif(
    "1.6" in torch.__version__,
    reason="group conv in pytorch 1.6 has an issue. "
    "See https://github.com/pytorch/pytorch/issues/42446.",
)
@pytest.mark.parametrize(
    "dict_fsq, dict_g, dict_d, dict_loss, average, include",
    [
        ({}, {}, {}, {}, True, True),
        ({}, {}, {}, {}, False, False),
        ({"use_dual_decoder": False}, {"quantizer_factor": 2}, {}, {}, False, True),
        ({"generator_params": {"encdec_true_skip": True}}, {}, {}, {}, True, True),
    ],
)
def test_fsqdac(dict_fsq, dict_g, dict_d, dict_loss, average, include):
    batch_size = 2
    batch_length = 256

    # Create audio tensor (different from original test: directly use (B, T) format)
    y = torch.randn(batch_size, batch_length)

    # Test FSQDAC as a full model
    args_fsq = make_fsqdac_args(**dict_fsq)
    model_fsq = FSQDAC(**args_fsq)

    # Test individual components
    args_g = make_generator_args(**dict_g)
    args_d = make_discriminator_args(**dict_d)
    args_loss = make_mel_loss_args(**dict_loss)

    model_g = DACGenerator(**args_g)
    model_d = DACDiscriminator(**args_d)

    aux_criterion = MultiScaleMelSpectrogramLoss(**args_loss)
    feat_match_criterion = FeatureMatchLoss(
        average_by_layers=average,
        average_by_discriminators=average,
        include_final_outputs=include,
    )
    gen_adv_criterion = GeneratorAdversarialLoss(
        average_by_discriminators=average,
    )
    dis_adv_criterion = DiscriminatorAdversarialLoss(
        average_by_discriminators=average,
    )

    # Test FSQDAC forward passes
    gen_output = model_fsq(y, forward_generator=True)
    dis_output = model_fsq(y, forward_generator=False)

    # Check if outputs contain expected keys
    assert "loss" in gen_output
    assert "stats" in gen_output
    assert "weight" in gen_output
    assert "optim_idx" in gen_output
    assert gen_output["optim_idx"] == 0

    assert "loss" in dis_output
    assert "stats" in dis_output
    assert "weight" in dis_output
    assert "optim_idx" in dis_output
    assert dis_output["optim_idx"] == 1

    # Test optimizers
    optimizer_g = torch.optim.AdamW(model_g.parameters())
    optimizer_d = torch.optim.AdamW(model_d.parameters())

    # Test generator trainable
    y_unsqueezed = y.unsqueeze(1)  # Add channel dimension for individual components
    y_hat, quant_loss, y_hat_real = model_g(y_unsqueezed, use_dual_decoder=True)
    p_hat = model_d(y_hat)
    aux_loss = aux_criterion(y_hat, y_unsqueezed)
    adv_loss = gen_adv_criterion(p_hat)

    with torch.no_grad():
        p = model_d(y_unsqueezed)

    fm_loss = feat_match_criterion(p_hat, p)
    loss_g = adv_loss + aux_loss + fm_loss

    optimizer_g.zero_grad()
    loss_g.backward()
    optimizer_g.step()

    # Test discriminator trainable
    p = model_d(y_unsqueezed)
    p_hat = model_d(y_hat.detach())
    real_loss, fake_loss = dis_adv_criterion(p_hat, p)
    loss_d = real_loss + fake_loss

    optimizer_d.zero_grad()
    loss_d.backward()
    optimizer_d.step()

    # Test inference methods
    # Test audio encoding/decoding roundtrip
    with torch.no_grad():
        # Test the FSQDAC model's inference method
        output = model_fsq.inference(y[0])
        assert "wav" in output
        assert "codec" in output

        # Test separate encode and decode methods
        codec = model_fsq.encode(y[0])
        wav = model_fsq.decode(codec)

        # Test roundtrip reconstruction
        assert wav.shape[-1] > 0
        assert codec.shape[-1] > 0

        # Test with batch input
        codec_batch = model_fsq.encode(y)
        wav_batch = model_fsq.decode(codec_batch)
        assert wav_batch.shape[0] == batch_size
