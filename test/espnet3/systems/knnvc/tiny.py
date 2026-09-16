"""Tiny HiFi-GAN hyperparameters shared by the kNN-VC unit tests."""

TINY_GENERATOR = {
    "in_channels": 6,
    "projection_channels": 8,
    "channels": 16,
    "kernel_size": 3,
    "upsample_scales": [2, 2],
    "upsample_kernel_sizes": [4, 4],
    "resblock_kernel_sizes": [3],
    "resblock_dilations": [[1, 2]],
}
TINY_HOP = 4  # prod(upsample_scales)

TINY_DISCRIMINATOR = {
    "scales": 1,
    "scale_discriminator_params": {
        "in_channels": 1,
        "out_channels": 1,
        "kernel_sizes": [5, 3, 3, 3],
        "channels": 4,
        "max_downsample_channels": 8,
        "max_groups": 2,
        "bias": True,
        "downsample_scales": [2, 1],
        "nonlinear_activation": "LeakyReLU",
        "nonlinear_activation_params": {"negative_slope": 0.1},
    },
    "periods": [2],
    "period_discriminator_params": {
        "in_channels": 1,
        "out_channels": 1,
        "kernel_sizes": [3, 3],
        "channels": 4,
        "downsample_scales": [2, 1],
        "max_downsample_channels": 8,
        "bias": True,
        "nonlinear_activation": "LeakyReLU",
        "nonlinear_activation_params": {"negative_slope": 0.1},
        "use_weight_norm": True,
        "use_spectral_norm": False,
    },
}

TINY_MEL = {
    "fs": 16000,
    "n_fft": 16,
    "hop_length": TINY_HOP,
    "win_length": 16,
    "n_mels": 4,
    "fmin": 0,
    "fmax": 8000,
}
