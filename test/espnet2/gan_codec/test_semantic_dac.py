# Copyright 2025 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Test code for SemanticDAC modules."""

import pytest
import torch

from espnet2.gan_codec.semantic_dac.semantic_dac import (
    SemanticDAC,
    SemanticDACGenerator,
)
from espnet2.gan_codec.shared.loss.freq_loss import MultiScaleMelSpectrogramLoss
from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
)

# Import the actual S3PRLUpstream for testing
from s3prl.nn import S3PRLUpstream

# Import the actual HubertTask for espnet semantic model testing
# Note: Actual import is only used if tests are run with semantic_type="espnet"
try:
    from espnet2.tasks.hubert import HubertTask
except ImportError:
    # For test environments without espnet2.tasks.hubert
    pass


def make_generator_args(**kwargs):
    """Create default generator arguments with optional overrides."""
    default = dict(
        sample_rate=120,
        hidden_dim=2,
        semantic_dim=240,
        semantic_type="s3prl",
        semantic_model="fbank",  # Using fbank for faster testing
        semantic_sample_rate=60,
        semantic_layer=0,
        semantic_loss="cosine",
        codebook_dim=2,
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
        quantizer_n_q=2,
        quantizer_bins=4,
        quantizer_decay=0.99,
        quantizer_kmeans_init=True,
        quantizer_kmeans_iters=1,
        quantizer_threshold_ema_dead_code=2,
        quantizer_target_bandwidth=[7.5, 15],
        quantizer_dropout=True,
    )
    default.update(kwargs)
    return default


def make_discriminator_args(**kwargs):
    """Create default discriminator arguments with optional overrides."""
    defaults = dict(
        scale_follow_official_norm=False,
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
    """Create default mel loss arguments with optional overrides."""
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
    "dict_g, dict_d, dict_loss, average, include",
    [
        ({}, {}, {}, True, True),
        ({}, {}, {}, False, False),
        ({"quantizer_kmeans_init": False}, {}, {}, False, True),
        ({"encdec_true_skip": True}, {}, {}, True, True),
        ({"semantic_loss": "L1"}, {}, {}, True, True),
        ({"semantic_loss": "L2"}, {}, {}, False, False),
    ],
)
def test_semantic_dac_generator(
    dict_g,
    dict_d,
    dict_loss,
    average,
    include,
):
    """Test SemanticDACGenerator under various configurations."""
    batch_size = 2
    batch_length = 128
    args_g = make_generator_args(**dict_g)

    # Create input tensor
    x = torch.randn(batch_size, 1, batch_length)

    # Initialize model
    model_g = SemanticDACGenerator(**args_g)

    # Test forward pass
    y_hat, commit_loss, quant_loss, semantic_loss, y_hat_real = model_g(
        x, use_dual_decoder=True
    )

    # Validate output shapes
    assert (
        y_hat.shape == x.shape
    ), f"Output shape {y_hat.shape} doesn't match input shape {x.shape}"
    assert (
        commit_loss.numel() == 1
    ), f"Commit loss should be a scalar, got shape {commit_loss.shape}"
    assert (
        quant_loss.numel() == 1
    ), f"Quantization loss should be a scalar, got shape {quant_loss.shape}"
    assert (
        semantic_loss.numel() == 1
    ), f"Semantic loss should be a scalar, got shape {semantic_loss.shape}"
    assert (
        y_hat_real.shape == x.shape
    ), f"Dual decoder output shape {y_hat_real.shape} "
       f"doesn't match input shape {x.shape}"

    # Test encode/decode
    codes = model_g.encode(x)
    x_hat = model_g.decode(codes)
    assert (
        x_hat.shape == x.shape
    ), f"Decoded shape {x_hat.shape} doesn't match input shape {x.shape}"


@pytest.mark.skipif(
    "1.6" in torch.__version__,
    reason="group conv in pytorch 1.6 has an issue. "
    "See https://github.com/pytorch/pytorch/issues/42446.",
)
@pytest.mark.parametrize(
    "dict_g, dict_d, dict_loss, average, include, cache_outputs",
    [
        ({}, {}, {}, True, True, True),
        ({}, {}, {}, False, False, False),
        ({"quantizer_kmeans_init": False}, {}, {}, False, True, True),
        ({"encdec_true_skip": True}, {}, {}, True, True, False),
        ({"semantic_loss": "L1"}, {}, {}, True, True, True),
        ({"semantic_loss": "L2"}, {}, {}, False, False, False),
    ],
)
def test_semantic_dac(
    dict_g,
    dict_d,
    dict_loss,
    average,
    include,
    cache_outputs,
):
    """Test the full SemanticDAC model under various configurations."""
    batch_size = 2
    batch_length = 128
    args_g = make_generator_args(**dict_g)
    args_d = make_discriminator_args(**dict_d)
    args_loss = make_mel_loss_args(**dict_loss)

    # Create input audio tensor
    y = torch.randn(batch_size, batch_length)

    # Initialize models and loss functions
    model = SemanticDAC(
        sampling_rate=120,
        generator_params=args_g,
        discriminator_params=args_d,
        generator_adv_loss_params={"average_by_discriminators": average},
        discriminator_adv_loss_params={"average_by_discriminators": average},
        use_feat_match_loss=True,
        feat_match_loss_params={
            "average_by_layers": average,
            "average_by_discriminators": average,
            "include_final_outputs": include,
        },
        use_mel_loss=True,
        mel_loss_params=args_loss,
        use_dual_decoder=True,
        lambda_quantization=1.0,
        lambda_reconstruct=1.0,
        lambda_commit=1.0,
        lambda_adv=1.0,
        lambda_feat_match=2.0,
        lambda_mel=45.0,
        lambda_semantic=1.0,
        cache_generator_outputs=cache_outputs,
    )

    # Create optimizers
    optimizer_g = torch.optim.AdamW(
        [p for p in model.generator.parameters() if p.requires_grad]
    )
    optimizer_d = torch.optim.AdamW(
        [p for p in model.discriminator.parameters() if p.requires_grad]
    )

    # Test generator forward path
    gen_output = model(y, forward_generator=True)

    # Validate generator output dictionary structure
    assert "loss" in gen_output, "Generator output should contain 'loss'"
    assert "stats" in gen_output, "Generator output should contain 'stats'"
    assert "weight" in gen_output, "Generator output should contain 'weight'"
    assert "optim_idx" in gen_output, "Generator output should contain 'optim_idx'"
    assert gen_output["optim_idx"] == 0, "Generator optim_idx should be 0"

    # Test generator optimization
    loss_g = gen_output["loss"]
    optimizer_g.zero_grad()
    loss_g.backward()
    optimizer_g.step()

    # Test discriminator forward path
    disc_output = model(y, forward_generator=False)

    # Validate discriminator output dictionary structure
    assert "loss" in disc_output, "Discriminator output should contain 'loss'"
    assert "stats" in disc_output, "Discriminator output should contain 'stats'"
    assert "weight" in disc_output, "Discriminator output should contain 'weight'"
    assert "optim_idx" in disc_output, "Discriminator output should contain 'optim_idx'"
    assert disc_output["optim_idx"] == 1, "Discriminator optim_idx should be 1"

    # Test discriminator optimization
    loss_d = disc_output["loss"]
    optimizer_d.zero_grad()
    loss_d.backward()
    optimizer_d.step()

    # Test inference methods
    x = torch.randn(batch_length).view(1, 1, -1)
    output = model.inference(x)

    assert "wav" in output, "Inference output should contain 'wav'"
    assert "codec" in output, "Inference output should contain 'codec'"

    # Test encode/decode methods
    codec = model.encode(x)
    wav = model.decode(codec)

    # Check model meta_info
    meta = model.meta_info()
    assert "fs" in meta, "Meta info should contain 'fs'"
    assert "num_streams" in meta, "Meta info should contain 'num_streams'"
    assert "frame_shift" in meta, "Meta info should contain 'frame_shift'"
    assert (
        "code_size_per_stream" in meta
    ), "Meta info should contain 'code_size_per_stream'"


@pytest.mark.skipif(
    "1.6" in torch.__version__,
    reason="group conv in pytorch 1.6 has an issue. "
    "See https://github.com/pytorch/pytorch/issues/42446.",
)
@pytest.mark.parametrize(
    "semantic_loss",
    [
        "cosine",
        "L1",
        "L2",
    ],
)
def test_semantic_loss_calculation(
    semantic_loss,
):
    """Test semantic loss calculation under different loss types."""
    batch_size = 2
    batch_length = 128

    # Create generator args with specific semantic config
    args_g = make_generator_args(
        semantic_loss=semantic_loss,
    )

    # Create input tensor
    x = torch.randn(batch_size, 1, batch_length)

    # Initialize model
    model_g = SemanticDACGenerator(**args_g)

    # Test forward pass
    _, _, _, semantic_loss_value, _ = model_g(x, use_dual_decoder=False)

    # Verify loss calculation worked
    assert semantic_loss_value.numel() == 1, "Semantic loss should be a scalar"
    assert not torch.isnan(semantic_loss_value), "Semantic loss should not be NaN"
    assert not torch.isinf(semantic_loss_value), "Semantic loss should not be infinity"
