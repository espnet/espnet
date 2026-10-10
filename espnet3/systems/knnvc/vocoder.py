"""HiFi-GAN vocoder on WavLM features: the trainable component of kNN-VC.

kNN-VC keeps the encoder (WavLM) frozen and the converter (kNN) is
non-parametric, so the vocoder is the only part that is trained. This module
provides the ``(loss, stats, weight)`` training model consumed by
:class:`espnet3.components.modeling.lightning_module.ESPnetLightningModule`
through its multi-optimizer path (``optimizers: {generator: ..., discriminator:
...}``), built from the HiFi-GAN modules that already ship in
``espnet2.gan_tts.hifigan``.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio

from espnet2.gan_tts.hifigan import (
    HiFiGANGenerator,
    HiFiGANMultiScaleMultiPeriodDiscriminator,
)
from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
)
from espnet3.components.modeling.optimization_spec import OptimizationStep

# HiFi-GAN "V1" generator adapted to 1024-dim WavLM features at a 320-sample
# hop (16 kHz), i.e. ``hifigan/config_v1_wavlm.json`` of the official kNN-VC
# repository expressed with :class:`KNNVCGenerator` argument names
# (``in_channels`` = ``hubert_dim``, ``projection_channels`` = ``hifi_dim``).
DEFAULT_GENERATOR_PARAMS: Dict[str, Any] = {
    "in_channels": 1024,
    "projection_channels": 512,
    "out_channels": 1,
    "channels": 512,
    "kernel_size": 7,
    "upsample_scales": [10, 8, 2, 2],
    "upsample_kernel_sizes": [20, 16, 4, 4],
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "use_additional_convs": True,
    "bias": True,
    "nonlinear_activation": "LeakyReLU",
    "nonlinear_activation_params": {"negative_slope": 0.1},
    "use_weight_norm": True,
}

# Mel-spectrogram settings of ``config_v1_wavlm.json``.
DEFAULT_MEL_PARAMS: Dict[str, Any] = {
    "fs": 16000,
    "n_fft": 1024,
    "hop_length": 320,
    "win_length": 1024,
    "n_mels": 80,
    "fmin": 0,
    "fmax": 8000,
}


def apply_official_period_output_kernel(
    discriminator: HiFiGANMultiScaleMultiPeriodDiscriminator,
) -> None:
    """Give the period discriminators kNN-VC's ``(3, 1)`` output convolution.

    ``espnet2.gan_tts.hifigan.HiFiGANPeriodDiscriminator`` derives that kernel
    as ``kernel_sizes[1] - 1`` and requires an odd ``kernel_sizes[1]``, so the
    official HiFi-GAN's ``(3, 1)`` is not reachable through configuration.
    Rebuilding ``output_conv`` makes each period discriminator identical to
    ``DiscriminatorP`` of the original HiFi-GAN
    (https://github.com/jik876/hifi-gan), which is what kNN-VC trains against.

    Args:
        discriminator: Discriminator whose ``mpd`` sub-discriminators are
            rebuilt in place.
    """
    for period_discriminator in discriminator.mpd.discriminators:
        conv = period_discriminator.output_conv
        spectral = hasattr(conv, "weight_orig")
        for remove in (
            torch.nn.utils.remove_weight_norm,
            torch.nn.utils.remove_spectral_norm,
        ):
            try:
                conv = remove(conv)
            except (ValueError, RuntimeError):
                pass
        replacement = torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            (3, 1),
            1,
            padding=(1, 0),
        )
        if spectral:
            replacement = torch.nn.utils.spectral_norm(replacement)
        else:
            replacement = torch.nn.utils.weight_norm(replacement)
        period_discriminator.output_conv = replacement


def build_generator_params(overrides: Optional[Dict[str, Any]] = None) -> Dict:
    """Merge user overrides into :data:`DEFAULT_GENERATOR_PARAMS`.

    Args:
        overrides: Subset of :class:`KNNVCGenerator` keyword arguments. ``None``
            or ``{}`` returns the kNN-VC defaults unchanged.

    Returns:
        A new dict of :class:`KNNVCGenerator` keyword arguments.
    """
    # Deep copy: the defaults hold nested lists (``upsample_scales``,
    # ``resblock_dilations``), and a shallow copy would hand every model the
    # module-level objects to mutate.
    params = copy.deepcopy(DEFAULT_GENERATOR_PARAMS)
    if overrides:
        params.update(copy.deepcopy(dict(overrides)))
    return params


class KNNVCGenerator(torch.nn.Module):
    """kNN-VC generator: linear feature projection + HiFi-GAN V1.

    The official kNN-VC ``Generator`` differs from stock HiFi-GAN by one
    ``Linear(hubert_dim, hifi_dim)`` applied to the WavLM features before the
    first convolution. This module reproduces that layout on top of
    ``espnet2.gan_tts.hifigan.HiFiGANGenerator`` so the released checkpoints
    can be loaded (see :mod:`~espnet3.systems.knnvc.checkpoint`) and
    so training uses the same network as the paper.

    Args:
        in_channels: Input feature dimension (WavLM-Large: ``1024``).
        projection_channels: Output size of the linear projection, which is
            also the HiFi-GAN input channel count (``512``).
        **hifigan_params: Remaining ``HiFiGANGenerator`` keyword arguments
            (``channels``, ``upsample_scales``, ...); ``in_channels`` of the
            HiFi-GAN is set to ``projection_channels``.
    """

    def __init__(
        self,
        in_channels: int = 1024,
        projection_channels: int = 512,
        **hifigan_params: Any,
    ) -> None:
        """Build the projection and the HiFi-GAN generator."""
        super().__init__()
        self.input_projection = torch.nn.Linear(in_channels, projection_channels)
        self.hifigan = HiFiGANGenerator(
            in_channels=projection_channels, **hifigan_params
        )

    @property
    def upsample_factor(self) -> int:
        """Return the number of waveform samples per input frame."""
        return int(self.hifigan.upsample_factor)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Generate waveforms.

        Args:
            feats: Features of shape ``(B, frames, in_channels)``.

        Returns:
            Waveform of shape ``(B, 1, frames * upsample_factor)``.
        """
        return self.hifigan(self.input_projection(feats).transpose(1, 2))

    def inference(self, feats: torch.Tensor) -> torch.Tensor:
        """Generate one waveform from ``(frames, in_channels)`` features.

        Returns:
            Waveform of shape ``(frames * upsample_factor,)``.
        """
        return self.forward(feats.unsqueeze(0)).squeeze(0).squeeze(0)

    def remove_weight_norm(self) -> None:
        """Remove weight normalization from the HiFi-GAN convolutions."""
        self.hifigan.remove_weight_norm()


class LogMelSpectrogram(torch.nn.Module):
    """Log-mel spectrogram matching the official kNN-VC HiFi-GAN training.

    Differences from ``espnet2.layers.log_mel.LogMel`` matter for reproducing
    the released vocoder: natural-log compression with a ``1e-5`` floor,
    magnitude (``power=1``) mel filterbank with Slaney normalization, and
    ``center=False`` with an explicit reflect pad of ``(n_fft - hop) / 2`` on
    both sides so that ``frames == samples / hop_length``.

    Args:
        fs: Sampling rate.
        n_fft: FFT size.
        hop_length: Hop size in samples.
        win_length: Window length in samples.
        n_mels: Number of mel bins.
        fmin: Lowest mel frequency.
        fmax: Highest mel frequency.
    """

    def __init__(
        self,
        fs: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 320,
        win_length: int = 1024,
        n_mels: int = 80,
        fmin: int = 0,
        fmax: Optional[int] = 8000,
    ) -> None:
        """Build the underlying torchaudio mel transform."""
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.melspectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=fs,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=False,
            power=1.0,
            norm="slaney",
            n_mels=n_mels,
            mel_scale="slaney",
            f_min=fmin,
            f_max=fmax,
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Compute log-mel features.

        Args:
            wav: Waveform of shape ``(B, samples)``.

        Returns:
            Log-mel tensor of shape ``(B, n_mels, frames)``.
        """
        pad = (self.n_fft - self.hop_length) // 2
        wav = F.pad(wav, (pad, pad), "reflect")
        mel = self.melspectrogram(wav)
        return torch.log(torch.clamp(mel, min=1e-5))


class KNNVCVocoderModel(torch.nn.Module):
    """HiFi-GAN generator + discriminator training model for kNN-VC.

    Forward returns the **discriminator** :class:`OptimizationStep` first and
    the **generator** one second, which is the order ESPnet3 backwards and
    steps them in. It matches the official ``hifigan/train.py`` and is what the
    espnet3 maintainers recommend for GAN models. The training config must
    declare optimizers under both names with matching ``params`` selectors; see
    the recipe ``training.yaml``.

    The discriminator scores the detached fake waveform for its own loss. For
    the generator's adversarial and feature-matching terms it runs with its
    parameters frozen, so the generator's graph does not depend on weights that
    the discriminator step has already updated in place -- without the freeze,
    that backward pass fails.

    The default loss weights are the official ones.

    Batch contract (what ``forward`` receives after ``CommonCollateFn``):

    - ``feats``: ``(B, frames, in_channels)`` encoder features, prematched
      or not (WavLM-Large layer 6 for kNN-VC);
    - ``speech``: ``(B, frames * hop_length)`` 16 kHz waveform aligned with
      ``feats``;
    - ``feats_lengths`` / ``speech_lengths``: accepted and ignored, because the
      recipe dataset already crops every training item to a fixed segment.

    Args:
        generator: Overrides for :data:`DEFAULT_GENERATOR_PARAMS`.
        discriminator: Keyword arguments for
            ``HiFiGANMultiScaleMultiPeriodDiscriminator`` (defaults are the
            official HiFi-GAN MSD + MPD).
        mel: Overrides for :data:`DEFAULT_MEL_PARAMS`.
        lambda_adv: Weight of the generator adversarial loss.
        lambda_feat_match: Weight of the feature matching loss.
        lambda_mel: Weight of the L1 log-mel loss.
        official_period_output_kernel: Rebuild the period discriminators'
            output convolution with the official ``(3, 1)`` kernel (see
            :func:`apply_official_period_output_kernel`). Set to ``False`` for
            the stock ``espnet2`` discriminator.
    """

    def __init__(
        self,
        generator: Optional[Dict[str, Any]] = None,
        discriminator: Optional[Dict[str, Any]] = None,
        mel: Optional[Dict[str, Any]] = None,
        lambda_adv: float = 1.0,
        lambda_feat_match: float = 2.0,
        lambda_mel: float = 45.0,
        official_period_output_kernel: bool = True,
    ) -> None:
        """Build generator, discriminator and loss modules."""
        super().__init__()
        self.generator = KNNVCGenerator(**build_generator_params(generator))
        self.discriminator = HiFiGANMultiScaleMultiPeriodDiscriminator(
            **(dict(discriminator) if discriminator else {})
        )
        if official_period_output_kernel:
            apply_official_period_output_kernel(self.discriminator)
        mel_params = dict(DEFAULT_MEL_PARAMS)
        if mel:
            mel_params.update(dict(mel))
        self.mel_spectrogram = LogMelSpectrogram(**mel_params)

        # Official HiFi-GAN sums (does not average) over discriminators and
        # layers, and its feature maps include the final discriminator output.
        self.generator_adv_loss = GeneratorAdversarialLoss(
            average_by_discriminators=False, loss_type="mse"
        )
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss(
            average_by_discriminators=False, loss_type="mse"
        )
        self.feat_match_loss = FeatureMatchLoss(
            average_by_layers=False,
            average_by_discriminators=False,
            include_final_outputs=True,
        )
        self.lambda_adv = float(lambda_adv)
        self.lambda_feat_match = float(lambda_feat_match)
        self.lambda_mel = float(lambda_mel)

    @property
    def hop_length(self) -> int:
        """Return the number of waveform samples produced per feature frame."""
        return int(self.generator.upsample_factor)

    def forward(
        self,
        feats: torch.Tensor,
        speech: torch.Tensor,
        feats_lengths: Optional[torch.Tensor] = None,
        speech_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[List[OptimizationStep], Dict[str, torch.Tensor], torch.Tensor]:
        """Compute generator and discriminator losses for one batch.

        Args:
            feats: Input features ``(B, frames, in_channels)``.
            speech: Target waveform ``(B, samples)``.
            feats_lengths: Ignored; present for the collate-function contract.
            speech_lengths: Ignored; present for the collate-function contract.
            **kwargs: Ignored extra batch fields.

        Returns:
            ``([discriminator_step, generator_step], stats, weight)`` where
            ``weight`` is the batch size.
        """
        y = speech.unsqueeze(1)
        y_hat = self.generator(feats)
        n_samples = min(y.size(-1), y_hat.size(-1))
        y = y[..., :n_samples]
        y_hat = y_hat[..., :n_samples]

        # ---- discriminator (scores the detached fake) ----
        p = self.discriminator(y)
        p_hat_detached = self.discriminator(y_hat.detach())
        real_loss, fake_loss = self.discriminator_adv_loss(p_hat_detached, p)
        discriminator_loss = real_loss + fake_loss

        # ---- generator (discriminator frozen: gradients flow through its
        # activations to the generator only) ----
        mel_loss = F.l1_loss(
            self.mel_spectrogram(y_hat.squeeze(1)),
            self.mel_spectrogram(y.squeeze(1)),
        )
        self._set_discriminator_requires_grad(False)
        try:
            p_hat = self.discriminator(y_hat)
            p_frozen = [[t.detach() for t in outs] for outs in p]
        finally:
            self._set_discriminator_requires_grad(True)
        adv_loss = self.generator_adv_loss(p_hat)
        feat_match_loss = self.feat_match_loss(p_hat, p_frozen)
        generator_loss = (
            self.lambda_adv * adv_loss
            + self.lambda_feat_match * feat_match_loss
            + self.lambda_mel * mel_loss
        )

        stats = {
            "mel_loss": mel_loss.detach(),
            "generator_adv_loss": adv_loss.detach(),
            "feat_match_loss": feat_match_loss.detach(),
            "generator_loss": generator_loss.detach(),
            "discriminator_real_loss": real_loss.detach(),
            "discriminator_fake_loss": fake_loss.detach(),
            "discriminator_loss": discriminator_loss.detach(),
        }
        weight = torch.tensor(feats.size(0), device=feats.device)
        return (
            [
                OptimizationStep(loss=discriminator_loss, name="discriminator"),
                OptimizationStep(loss=generator_loss, name="generator"),
            ],
            stats,
            weight,
        )

    def _set_discriminator_requires_grad(self, flag: bool) -> None:
        for param in self.discriminator.parameters():
            param.requires_grad_(flag)

    @torch.inference_mode()
    def inference(self, feats: torch.Tensor) -> torch.Tensor:
        """Vocode one feature sequence.

        Args:
            feats: Features of shape ``(frames, in_channels)``.

        Returns:
            Waveform of shape ``(frames * hop_length,)``.
        """
        return self.generator.inference(feats)
