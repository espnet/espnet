# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""VITS: Variational Inference with adversarial learning for end-to-end Text-to-Speech.

This code is based on https://github.com/jaywalnut310/vits.

"""

from typing import Any
from typing import Dict
from typing import Optional

import torch

from typeguard import check_argument_types

from espnet2.gan_tts.abs_gan_tts import AbsGANTTS
from espnet2.gan_tts.vits.generator import VITSGenerator
from espnet2.gan_tts.vits.hifigan import HiFiGANMultiPeriodDiscriminator
from espnet2.gan_tts.vits.hifigan import HiFiGANMultiScaleDiscriminator
from espnet2.gan_tts.vits.hifigan import HiFiGANMultiScaleMultiPeriodDiscriminator
from espnet2.gan_tts.vits.hifigan import HiFiGANPeriodDiscriminator
from espnet2.gan_tts.vits.hifigan import HiFiGANScaleDiscriminator
from espnet2.gan_tts.vits.loss import DiscriminatorAdversarialLoss
from espnet2.gan_tts.vits.loss import FeatureMatchLoss
from espnet2.gan_tts.vits.loss import GeneratorAdversarialLoss
from espnet2.gan_tts.vits.loss import KLDivergenceLoss
from espnet2.gan_tts.vits.loss import MelSpectrogramLoss
from espnet2.torch_utils.device_funcs import force_gatherable


AVAILABLE_GENERATERS = {
    "vits_generator": VITSGenerator,
}
AVAILABLE_DISCRIMINATORS = {
    "hifigan_period_discriminator": HiFiGANPeriodDiscriminator,
    "hifigan_scale_discriminator": HiFiGANScaleDiscriminator,
    "hifigan_multi_period_discriminator": HiFiGANMultiPeriodDiscriminator,
    "hifigan_multi_scale_discriminator": HiFiGANMultiScaleDiscriminator,
    "hifigan_multi_scale_multi_period_discriminator": HiFiGANMultiScaleMultiPeriodDiscriminator,  # NOQA
}


class VITS(AbsGANTTS):
    """VITS module (generator + discriminator).

    This is a module of VITS described in `Conditional Variational Autoencoder
    with Adversarial Learning for End-to-End Text-to-Speech`_.

    .. _`Conditional Variational Autoencoder with Adversarial Learning for End-to-End
        Text-to-Speech`: https://arxiv.org/abs/2006.04558

    """

    def __init__(
        self,
        # generator related
        idim: int,
        odim: int,
        generator_type: str = "vits_generator",
        generator_params: Dict[str, Any] = {
            "aux_channels": 80,
            "hidden_channels": 192,
            "spks": -1,
            "global_channels": -1,
            "segment_size": 32,
            "text_encoder_attention_heads": 2,
            "text_encoder_attention_expand": 4,
            "text_encoder_blocks": 6,
            "text_encoder_kernel_size": 3,
            "text_encoder_dropout_rate": 0.1,
            "text_encoder_positional_dropout_rate": 0.0,
            "text_encoder_attention_dropout_rate": 0.0,
            "decoder_kernel_size": 7,
            "decoder_channels": 512,
            "decoder_upsample_scales": (8, 8, 2, 2),
            "decoder_upsample_kernel_sizes": (16, 16, 4, 4),
            "decoder_resblock_kernel_sizes": (3, 7, 11),
            "decoder_resblock_dilations": [(1, 3, 5), (1, 3, 5), (1, 3, 5)],
            "use_weight_norm_in_decoder": True,
            "posterior_encoder_kernel_size": 5,
            "posterior_encoder_layers": 16,
            "posterior_encoder_stacks": 1,
            "posterior_encoder_base_dilation": 1,
            "posterior_encoder_dropout_rate": 0.0,
            "use_weight_norm_in_posterior_encoder": True,
            "flow_flows": 4,
            "flow_kernel_size": 5,
            "flow_base_dilation": 1,
            "flow_layers": 5,
            "flow_dropout_rate": 0.0,
            "use_weight_norm_in_flow": True,
            "use_only_mean_in_flow": True,
            "stochastic_duration_predictor_kernel_size": 3,
            "stochastic_duration_predictor_dropout_rate": 0.5,
            "stochastic_duration_predictor_flows": 4,
            "stochastic_duration_predictor_dds_conv_layers": 3,
        },
        # discriminator related
        discriminator_type: str = "hifigan_multi_scale_multi_period_discriminator",
        discriminator_params: Dict[str, Any] = {
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
                "downsample_scales": [2, 2, 4, 4, 1],
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
        },
        # loss related
        generator_adv_loss_params: Dict[str, Any] = {
            "average_by_discriminators": False,
        },
        discriminator_adv_loss_params: Dict[str, Any] = {
            "average_by_discriminators": False,
        },
        feat_match_loss_params: Dict[str, Any] = {
            "average_by_discriminators": False,
            "average_by_layers": False,
            "include_final_outputs": True,
        },
        mel_loss_params: Dict[str, Any] = {
            "fs": 22050,
            "n_fft": 1024,
            "hop_length": 256,
            "win_length": None,
            "window": "hann",
            "n_mels": 80,
            "fmin": 0,
            "fmax": None,
            "log_base": None,
        },
        lambda_adv: float = 1.0,
        lambda_mel: float = 45.0,
        lambda_feat_match: float = 1.0,
        lambda_dur: float = 1.0,
        lambda_kl: float = 1.0,
        cache_generator_outputs: bool = True,
    ):
        """Initialize VITS module.

        Args:
            idim (int): Input dimension.
            odim (int): Output dimension (dummy for compatibility).
            generator_params (Dict[str, Any]): Generator parameter dict.
            discriminator_params (Dict[str, Any]): Discriminator parameter dict.
            generator_adv_loss_params (Dict[str, Any]): Generator adversarial loss
                parameter dict.
            discriminator_adv_loss_params (Dict[str, Any]): Discriminator adversarial
                loss parameter dict.
            feat_match_loss_params (Dict[str, Any]): Feature matching loss parameter
                dict.
            mel_loss_params (Dict[str, Any]): Mel spectrogram loss parameter dict.
            lambda_adv (float): Loss scaling coefficient for adversarial loss.
            lambda_mel (float): Loss scaling coefficient for mel spectrogram loss.
            lambda_feat_match (float): Loss scaling coefficient for feat match loss.
            lambda_dur (float): Loss scaling coefficient for duration loss.
            lambda_kl (float): Loss scaling coefficient for KL divergence loss.

        """
        assert check_argument_types()
        super().__init__()

        # define modules
        generator_params.update(idim=idim, odim=odim)
        generator_class = AVAILABLE_GENERATERS[generator_type]
        self.generator = generator_class(
            **generator_params,
        )
        discriminator_class = AVAILABLE_DISCRIMINATORS[discriminator_type]
        self.discriminator = discriminator_class(
            **discriminator_params,
        )
        self.generator_adv_loss = GeneratorAdversarialLoss(
            **generator_adv_loss_params,
        )
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss(
            **discriminator_adv_loss_params,
        )
        self.feat_match_loss = FeatureMatchLoss(
            **feat_match_loss_params,
        )
        self.mel_loss = MelSpectrogramLoss(
            **mel_loss_params,
        )
        self.kl_loss = KLDivergenceLoss()

        # coefficients
        self.lambda_adv = lambda_adv
        self.lambda_mel = lambda_mel
        self.lambda_kl = lambda_kl
        self.lambda_feat_match = lambda_feat_match
        self.lambda_dur = lambda_dur

        # cache
        self.cache_generator_outputs = cache_generator_outputs
        self._cache = None

    @property
    def require_raw_speech(self):
        """Return whether or not speech is required."""
        return True

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        sids: Optional[torch.Tensor] = None,
        forward_generator: bool = True,
    ):
        """Perform generator forward.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor): Feature length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            sids (Optional[Tensor]): Speaker index tensor (B,).
            forward_generator (bool): Whether to forward generator.

        Returns:
            Dict[str, Any]:
                - Tensor: Loss scalar tensor.
                - Dict[str, float]: Statistics to be monitored.
                - Tensor: Weight value.
                - int: Optimizer index (0 for generator and 1 for discriminator).

        """
        if forward_generator:
            return self._forward_generator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                speech=speech,
                speech_lengths=speech_lengths,
                sids=sids,
            )
        else:
            return self._forward_discrminator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                speech=speech,
                speech_lengths=speech_lengths,
                sids=sids,
            )

    def _forward_generator(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        sids: Optional[torch.Tensor] = None,
    ):
        """Perform generator forward.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor): Feature length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            sids (Optional[Tensor]): Speaker index tensor (B,).

        Returns:
            Dict[str, Any]:
                - Tensor: Loss scalar tensor.
                - Dict[str, float]: Statistics to be monitored.
                - Tensor: Weight value.
                - int: Optimizer index (0 for generator).

        """
        # setup
        batch_size = text.size(0)
        feats = feats.transpose(1, 2)
        speech = speech.unsqueeze(1)

        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            outs = self.generator(text, text_lengths, feats, feats_lengths, sids)
        else:
            outs = self._cache

        # store cache
        if self.training and self.cache_generator_outputs and not reuse_cache:
            self._cache = outs

        # parse outputs
        speech_hat_, dur_nll, _, start_idxs, _, z_mask, outs_ = outs
        _, z_p, m_p, logs_p, _, logs_q = outs_
        speech_ = self.generator.get_segments(
            x=speech,
            start_idxs=start_idxs * self.generator.upsample_factor,
            segment_size=self.generator.segment_size * self.generator.upsample_factor,
        )

        # calculate discriminator outputs
        p_hat = self.discriminator(speech_hat_)
        with torch.no_grad():
            # do not store discriminator gradient in generator turn
            p = self.discriminator(speech_)

        # calculate losses
        mel_loss = self.mel_loss(speech_hat_, speech_)
        kl_loss = self.kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
        dur_loss = torch.sum(dur_nll.float())
        adv_loss = self.generator_adv_loss(p_hat)
        feat_match_loss = self.feat_match_loss(p_hat, p)

        mel_loss = mel_loss * self.lambda_mel
        kl_loss = kl_loss * self.lambda_kl
        dur_loss = dur_loss * self.lambda_dur
        adv_loss = adv_loss * self.lambda_adv
        feat_match_loss = feat_match_loss * self.lambda_feat_match
        loss = mel_loss + kl_loss + dur_loss + adv_loss + feat_match_loss

        stats = dict(
            generator_loss=loss.item(),
            generator_mel_loss=mel_loss.item(),
            generator_kl_loss=kl_loss.item(),
            generator_dur_loss=dur_loss.item(),
            generator_adv_loss=adv_loss.item(),
            generator_feat_match_loss=feat_match_loss.item(),
        )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        # reset cache
        if reuse_cache or not self.training:
            self._cache = None

        return {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 0,  # needed for trainer
        }

    def _forward_discrminator(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        sids: Optional[torch.Tensor] = None,
    ):
        """Perform discriminator forward.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor): Feature length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            sids (Optional[Tensor]): Speaker index tensor (B,).

        Returns:
            Dict[str, Any]:
                - Tensor: Loss scalar tensor.
                - Dict[str, float]: Statistics to be monitored.
                - Tensor: Weight value.
                - int: Optimizer index (1 for discriminator).

        """
        # setup
        batch_size = text.size(0)
        feats = feats.transpose(1, 2)
        speech = speech.unsqueeze(1)

        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            outs = self.generator(text, text_lengths, feats, feats_lengths, sids)
        else:
            outs = self._cache

        # store cache
        if self.cache_generator_outputs and not reuse_cache:
            self._cache = outs

        # parse outputs
        speech_hat_, _, _, start_idxs, *_ = outs
        speech_ = self.generator.get_segments(
            x=speech,
            start_idxs=start_idxs * self.generator.upsample_factor,
            segment_size=self.generator.segment_size * self.generator.upsample_factor,
        )

        # calculate discriminator outputs
        p_hat = self.discriminator(speech_hat_.detach())
        p = self.discriminator(speech_)

        # calculate losses
        real_loss, fake_loss = self.discriminator_adv_loss(p_hat, p)
        loss = real_loss + fake_loss

        stats = dict(
            discriminator_loss=loss.item(),
            discriminator_real_loss=real_loss.item(),
            discriminator_fake_loss=fake_loss.item(),
        )
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        # reset cache
        if reuse_cache or not self.training:
            self._cache = None

        return {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 1,  # needed for trainer
        }

    def inference(
        self,
        text: torch.Tensor,
        sids: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
        noise_scale: float = 1.0,
        length_scale: float = 1.0,
        noise_scale_w: float = 1.0,
        max_len: Optional[int] = None,
        **kwargs,
    ):
        """Run inference.

        Args:
            text (Tensor): Input text index tensor (T_text,).
            sids (Tensor): Speaker index tensor (1,).
            durations (Tensor): Ground-truth duration tensor (T_text,).
            noise_scale (float): Noise scale value for flow.
            length_scale (float): Length scaling value.
            noise_scale_w (float): Noise scale value for duration predictor.
            max_len (Optional[int]): Maximum length.

        Returns:
            Tensor: Generated waveform tensor (T_wav).
            Tensor: Attention weight tensor (T_feats, T_text).
            None: Dummy outputs for compatibility.

        """
        # setup
        text = text[None]
        text_lengths = torch.tensor(
            [text.size(1)],
            dtype=torch.long,
            device=text.device,
        )
        if sids is not None:
            sids = sids.view(1, 1)
        if durations is not None:
            durations = durations.view(1, 1, -1)

        wav, att_w, _ = self.generator.inference(
            text=text,
            text_lengths=text_lengths,
            sids=sids,
            dur=durations,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            max_len=max_len,
        )
        return wav[0], None, att_w[0]
