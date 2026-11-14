"""Synthetic recordings."""

import numpy as np
from registry import corpus, mine


def coloured_noise(rng: np.random.RandomState, n: int, exponent: float) -> np.ndarray:
    """Noise with power spectrum 1/f^exponent (0 white, 1 pink, 2 brown), unit RMS."""
    spec = np.fft.rfft(rng.randn(n))
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1.0
    spec /= f ** (exponent / 2)
    x = np.fft.irfft(spec, n)
    return (x / (np.sqrt(np.mean(x**2)) + 1e-12)).astype(np.float32)


@corpus("synth_lownoise", "", "quiet_closemic", "D", "generated (no licence)")
def synth_lownoise(root, writers, shard: int, nshards: int, num=3000, dur=20.0):
    """Low-level noise without speech: the quiet room / mic self-noise of a laptop.

    Each recording is white, pink or brown noise, or digital silence with a
    dither-level noise floor, at an RMS of -90 to -40 dBFS (uniform in dB),
    with a slow gain drift of up to +-3 dB and, in a quarter of them, a mains
    hum (50 or 60 Hz and harmonics) 10-30 dB below the noise.
    """
    fs = writers["train"].fs
    n = int(dur * fs)
    t = np.arange(n) / fs
    kinds = [("white", 0.0), ("pink", 1.0), ("brown", 2.0), ("silence", None)]
    for i in range(num):
        if not mine(i, shard, nshards):
            continue
        rng = np.random.RandomState(i)
        kind, exponent = kinds[i % len(kinds)]
        if exponent is None:
            x = rng.randn(n).astype(np.float32)
            level_db = rng.uniform(-100, -85)  # about 16-bit dither
        else:
            x = coloured_noise(rng, n, exponent)
            level_db = rng.uniform(-90, -40)
        drift = 10 ** (
            rng.uniform(-3, 3) * np.sin(2 * np.pi * t / dur + rng.uniform(0, 6.3)) / 20
        )
        x = x * drift
        if rng.rand() < 0.25:
            mains = rng.choice([50.0, 60.0])
            hum = sum(
                np.sin(2 * np.pi * mains * h * t + rng.uniform(0, 6.3)) / h
                for h in (1, 2, 3)
            )
            hum = hum / np.sqrt(np.mean(hum**2)) * 10 ** (-rng.uniform(10, 30) / 20)
            x = x + hum
        x = x / np.sqrt(np.mean(x**2)) * 10 ** (level_db / 20)
        split = "dev" if i % 20 == 0 else "train"  # 5% for validation
        writers[split].add(
            f"{i:05d}",
            x.astype(np.float32),
            fs,
            negative_type=f"synthetic_{kind}",
            mic="synthetic",
        )
