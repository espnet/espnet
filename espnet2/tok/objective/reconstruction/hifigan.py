"""HiFi-GAN waveform reconstruction for speech-tokenizer training."""

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from espnet2.gan_tts.hifigan import (
    HiFiGANGenerator,
    HiFiGANMultiScaleMultiPeriodDiscriminator,
)
from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
    MelSpectrogramLoss,
)
from espnet2.gan_tts.utils import get_random_segments, get_segments
from espnet2.tok.modules.token_embedding import TokenEmbedding
from espnet2.tok.objective.reconstruction.abs_reconstruction import (
    AbsReconstructionObjective,
)
from espnet2.tok.output import SpeechTokenizerOutput
from espnet2.torch_utils.device_funcs import force_gatherable

DEFAULT_GENERATOR_PARAMS = {
    "out_channels": 1,
    "channels": 512,
    "global_channels": -1,
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

DEFAULT_DISCRIMINATOR_PARAMS = {
    "scales": 3,
    "scale_downsample_pooling": "AvgPool1d",
    "scale_downsample_pooling_params": {
        "kernel_size": 4,
        "stride": 2,
        "padding": 2,
    },
    "scale_discriminator_params": {
        "in_channels": 1,
        "out_channels": 1,
        "kernel_sizes": [15, 41, 5, 3],
        "channels": 128,
        "max_downsample_channels": 1024,
        "max_groups": 16,
        "bias": True,
        "downsample_scales": [4, 4, 4, 4, 1],
        "nonlinear_activation": "LeakyReLU",
        "nonlinear_activation_params": {"negative_slope": 0.1},
    },
    "follow_official_norm": True,
    "periods": [2, 3, 5, 7, 11],
    "period_discriminator_params": {
        "in_channels": 1,
        "out_channels": 1,
        "kernel_sizes": [5, 3],
        "channels": 32,
        "downsample_scales": [3, 3, 3, 3, 1],
        "max_downsample_channels": 1024,
        "bias": True,
        "nonlinear_activation": "LeakyReLU",
        "nonlinear_activation_params": {"negative_slope": 0.1},
        "use_weight_norm": True,
        "use_spectral_norm": False,
    },
}

DEFAULT_MEL_LOSS_PARAMS = {
    "fs": 16000,
    "n_fft": 1024,
    "hop_length": 256,
    "win_length": None,
    "window": "hann",
    "n_mels": 80,
    "fmin": 0,
    "fmax": 8000,
    "log_base": None,
}


class HiFiGANReconstructionObjective(AbsReconstructionObjective):
    """Reconstruct speech from differentiable tokens with HiFi-GAN.

    The objective will own its token embedding, speaker-conditioned generator,
    discriminators, reconstruction losses, and reusable generator-output cache.

    Token assignments are embedded differentiably.  A pretrained speaker
    embedding is repeated over token frames and concatenated with the token
    embedding before the standard ESPnet HiFi-GAN generator.  Defaults follow
    ParallelWaveGAN's 16-kHz VCTK HuBERT vocoder configuration.
    """

    def __init__(
        self,
        num_tokens: int,
        token_embed_dim: int = 512,
        speaker_embed_dim: int = 512,
        segment_size: int = 32,
        generator_params: Optional[Dict[str, Any]] = None,
        discriminator_params: Optional[Dict[str, Any]] = None,
        generator_adv_loss_params: Optional[Dict[str, Any]] = None,
        discriminator_adv_loss_params: Optional[Dict[str, Any]] = None,
        feat_match_loss_params: Optional[Dict[str, Any]] = None,
        mel_loss_params: Optional[Dict[str, Any]] = None,
        use_feat_match_loss: bool = True,
        use_mel_loss: bool = True,
        lambda_adv: float = 1.0,
        lambda_feat_match: float = 2.0,
        lambda_mel: float = 45.0,
        cache_generator_outputs: bool = True,
    ) -> None:
        super().__init__()
        if speaker_embed_dim <= 0:
            raise ValueError(f"speaker_embed_dim must be positive: {speaker_embed_dim}")
        if segment_size <= 0:
            raise ValueError(f"segment_size must be positive: {segment_size}")
        for name, value in (
            ("lambda_adv", lambda_adv),
            ("lambda_feat_match", lambda_feat_match),
            ("lambda_mel", lambda_mel),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative: {value}")

        self.embedding = TokenEmbedding(
            input_size=num_tokens,
            embed_dim=token_embed_dim,
            use_positional_encoding=False,
        )
        generator_conf = deepcopy(DEFAULT_GENERATOR_PARAMS)
        if generator_params is not None:
            generator_conf.update(generator_params)
        expected_channels = token_embed_dim + speaker_embed_dim
        configured_channels = generator_conf.pop("in_channels", expected_channels)
        if configured_channels != expected_channels:
            raise ValueError(
                "generator in_channels must equal token_embed_dim + "
                f"speaker_embed_dim: {configured_channels} != {expected_channels}"
            )
        if generator_conf.get("global_channels", -1) > 0:
            raise ValueError(
                "global_channels must be disabled because speaker embeddings are "
                "concatenated with token embeddings"
            )
        self.generator = HiFiGANGenerator(
            in_channels=expected_channels, **generator_conf
        )

        discriminator_conf = deepcopy(DEFAULT_DISCRIMINATOR_PARAMS)
        if discriminator_params is not None:
            discriminator_conf.update(discriminator_params)
        self.discriminator = HiFiGANMultiScaleMultiPeriodDiscriminator(
            **discriminator_conf
        )
        self.generator_adv_loss = GeneratorAdversarialLoss(
            **({"average_by_discriminators": False} | (generator_adv_loss_params or {}))
        )
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss(
            **(
                {"average_by_discriminators": False}
                | (discriminator_adv_loss_params or {})
            )
        )
        self.use_feat_match_loss = use_feat_match_loss
        if use_feat_match_loss:
            self.feat_match_loss = FeatureMatchLoss(
                **(
                    {
                        "average_by_discriminators": False,
                        "average_by_layers": False,
                        "include_final_outputs": True,
                    }
                    | (feat_match_loss_params or {})
                )
            )
        self.use_mel_loss = use_mel_loss
        if use_mel_loss:
            mel_conf = deepcopy(DEFAULT_MEL_LOSS_PARAMS)
            if mel_loss_params is not None:
                mel_conf.update(mel_loss_params)
            self.mel_loss = MelSpectrogramLoss(**mel_conf)

        self.speaker_embed_dim = speaker_embed_dim
        self.segment_size = segment_size
        self.lambda_adv = lambda_adv
        self.lambda_feat_match = lambda_feat_match
        self.lambda_mel = lambda_mel
        self.cache_generator_outputs = cache_generator_outputs
        self._cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    @property
    def is_adversarial(self) -> bool:
        """Return that this objective requires a discriminator optimizer."""
        return True

    @property
    def has_cached_generator_outputs(self) -> bool:
        """Return whether a discriminator turn can reuse generated waveforms."""
        return self._cache is not None

    @property
    def upsample_factor(self) -> int:
        """Return the number of waveform samples generated per token frame."""
        return self.generator.upsample_factor

    def _prepare_condition(
        self,
        tokenizer_output: SpeechTokenizerOutput,
        spembs: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate differentiable token and pretrained speaker embeddings."""
        token_features, _ = self.embedding(
            tokenizer_output.assignment, tokenizer_output.lengths
        )
        if spembs.dim() != 2:
            raise ValueError(
                f"spembs must have shape (B, D_spk), but got {tuple(spembs.shape)}"
            )
        if spembs.size(0) != token_features.size(0):
            raise ValueError(
                "Batch sizes of token features and spembs differ: "
                f"{token_features.size(0)} != {spembs.size(0)}"
            )
        if spembs.size(1) != self.speaker_embed_dim:
            raise ValueError(
                "Speaker embedding dimension differs from speaker_embed_dim: "
                f"{spembs.size(1)} != {self.speaker_embed_dim}"
            )
        speaker_features = spembs.to(dtype=token_features.dtype).unsqueeze(1)
        speaker_features = speaker_features.expand(-1, token_features.size(1), -1)
        return torch.cat([token_features, speaker_features], dim=-1).transpose(1, 2)

    @staticmethod
    def _as_waveform_channels(speech: torch.Tensor) -> torch.Tensor:
        """Convert waveform input to shape ``(B, 1, T)``."""
        if speech.dim() == 2:
            return speech.unsqueeze(1)
        if speech.dim() == 3 and speech.size(1) == 1:
            return speech
        raise ValueError(
            f"speech must have shape (B, T) or (B, 1, T), got {tuple(speech.shape)}"
        )

    @staticmethod
    def _pad_time(x: torch.Tensor, length: int) -> torch.Tensor:
        """Right-pad the time axis to at least ``length``."""
        return F.pad(x, (0, max(0, length - x.size(-1))))

    def _generate_segments(
        self,
        tokenizer_output: SpeechTokenizerOutput,
        speech: torch.Tensor,
        spembs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random fixed-size segments aligned with real waveform."""
        condition = self._prepare_condition(tokenizer_output, spembs)
        condition = self._pad_time(condition, self.segment_size)
        condition_segment, start_idxs = get_random_segments(
            condition, tokenizer_output.lengths, self.segment_size
        )
        speech_hat = self.generator(condition_segment)

        waveform = self._as_waveform_channels(speech)
        waveform_size = self.segment_size * self.upsample_factor
        required_waveform_size = max(
            waveform_size,
            int(tokenizer_output.lengths.max().item()) * self.upsample_factor,
        )
        waveform = self._pad_time(waveform, required_waveform_size)
        speech_segment = get_segments(
            waveform,
            start_idxs * self.upsample_factor,
            waveform_size,
        )
        return speech_hat, speech_segment

    def forward(
        self,
        tokenizer_output: SpeechTokenizerOutput,
        speech: torch.Tensor,
        spembs: torch.Tensor,
        **batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Compute the generator-side vocoder objective."""
        speech_hat, speech_real = self._generate_segments(
            tokenizer_output, speech, spembs
        )
        if self.training and self.cache_generator_outputs:
            self._cache = (speech_hat.detach(), speech_real.detach())

        fake_outputs = self.discriminator(speech_hat)
        with torch.no_grad():
            real_outputs = self.discriminator(speech_real)
        adv_loss = self.lambda_adv * self.generator_adv_loss(fake_outputs)
        loss = adv_loss
        stats: Dict[str, Any] = {"adv_loss": adv_loss.detach()}

        if self.use_feat_match_loss:
            feature_loss = self.lambda_feat_match * self.feat_match_loss(
                fake_outputs, real_outputs
            )
            loss = loss + feature_loss
            stats["feat_match_loss"] = feature_loss.detach()
        if self.use_mel_loss:
            mel_loss = self.lambda_mel * self.mel_loss(speech_hat, speech_real)
            loss = loss + mel_loss
            stats["mel_loss"] = mel_loss.detach()
        stats["loss"] = loss.detach()
        return force_gatherable((loss, stats, speech_hat.size(0)), device=loss.device)

    def forward_discriminator(
        self,
        tokenizer_output: Optional[SpeechTokenizerOutput],
        speech: torch.Tensor,
        spembs: torch.Tensor,
        **batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Compute discriminator loss, reusing cached generated waveform."""
        if self.cache_generator_outputs and self._cache is not None:
            speech_hat, speech_real = self._cache
        else:
            if tokenizer_output is None:
                raise RuntimeError(
                    "tokenizer_output is required when no generator cache exists"
                )
            with torch.no_grad():
                speech_hat, speech_real = self._generate_segments(
                    tokenizer_output, speech, spembs
                )
        fake_outputs = self.discriminator(speech_hat.detach())
        real_outputs = self.discriminator(speech_real)
        real_loss, fake_loss = self.discriminator_adv_loss(fake_outputs, real_outputs)
        loss = real_loss + fake_loss
        stats: Dict[str, Any] = {
            "loss": loss.detach(),
            "real_loss": real_loss.detach(),
            "fake_loss": fake_loss.detach(),
        }
        self._cache = None
        return force_gatherable((loss, stats, speech_hat.size(0)), device=loss.device)

    def synthesize(
        self,
        tokenizer_output: SpeechTokenizerOutput,
        spembs: torch.Tensor,
        **batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Synthesize full waveforms from tokens and speaker embeddings."""
        condition = self._prepare_condition(tokenizer_output, spembs)
        waveform = self.generator(condition)
        waveform_lengths = tokenizer_output.lengths * self.upsample_factor
        return waveform, waveform_lengths
