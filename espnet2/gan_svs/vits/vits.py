# Copyright 2021 Tomoki Hayashi
# Copyright 2022 Yifeng Yu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""VITS module for GAN-SVS task."""

from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Any, Dict, Optional

import torch
from typeguard import check_argument_types

from espnet2.gan_svs.abs_gan_svs import AbsGANSVS
from espnet2.gan_tts.hifigan import (
    HiFiGANMultiPeriodDiscriminator,
    HiFiGANMultiScaleDiscriminator,
    HiFiGANMultiScaleMultiPeriodDiscriminator,
    HiFiGANPeriodDiscriminator,
    HiFiGANScaleDiscriminator,
)
from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
    MelSpectrogramLoss,
)
from espnet2.gan_tts.utils import get_segments
from espnet2.gan_svs.vits.generator import VITSGenerator
from espnet2.gan_tts.vits.loss import KLDivergenceLoss
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

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):  # NOQA
        yield


class VITS(AbsGANSVS):
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
        sampling_rate: int = 22050,
        generator_type: str = "vits_generator",
        use_visinger: bool = True,
        use_dp: bool = True,
        generator_params: Dict[str, Any] = {
            "midi_dim": 129,
            "midi_embed_integration_type": "add",
            "hidden_channels": 192,
            "spks": None,
            "langs": None,
            "spk_embed_dim": None,
            "global_channels": -1,
            "segment_size": 32,
            "text_encoder_attention_heads": 2,
            "text_encoder_ffn_expand": 4,
            "text_encoder_blocks": 6,
            "text_encoder_positionwise_layer_type": "conv1d",
            "text_encoder_positionwise_conv_kernel_size": 1,
            "text_encoder_positional_encoding_layer_type": "rel_pos",
            "text_encoder_self_attention_layer_type": "rel_selfattn",
            "text_encoder_activation_type": "swish",
            "text_encoder_normalize_before": True,
            "text_encoder_dropout_rate": 0.1,
            "text_encoder_positional_dropout_rate": 0.0,
            "text_encoder_attention_dropout_rate": 0.0,
            "text_encoder_conformer_kernel_size": 7,
            "use_macaron_style_in_text_encoder": True,
            "use_conformer_conv_in_text_encoder": True,
            "decoder_kernel_size": 7,
            "decoder_channels": 512,
            "decoder_upsample_scales": [8, 8, 2, 2],
            "decoder_upsample_kernel_sizes": [16, 16, 4, 4],
            "decoder_resblock_kernel_sizes": [3, 7, 11],
            "decoder_resblock_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
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
            "flow_layers": 4,
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
            "scales": 1,
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
                "use_weight_norm": True,
                "use_spectral_norm": False,
            },
            "follow_official_norm": False,
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
            "loss_type": "mse",
        },
        discriminator_adv_loss_params: Dict[str, Any] = {
            "average_by_discriminators": False,
            "loss_type": "mse",
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
        lambda_feat_match: float = 2.0,
        lambda_dur: float = 1.0,
        lambda_kl: float = 1.0,
        lambda_pitch: float = 1.0,
        lambda_phoneme: float = 1.0,
        cache_generator_outputs: bool = True,
    ):
        """Initialize VITS module.

        Args:
            idim (int): Input vocabrary size.
            odim (int): Acoustic feature dimension. The actual output channels will
                be 1 since VITS is the end-to-end text-to-wave model but for the
                compatibility odim is used to indicate the acoustic feature dimension.
            sampling_rate (int): Sampling rate, not used for the training but it will
                be referred in saving waveform during the inference.
            generator_type (str): Generator type.
            generator_params (Dict[str, Any]): Parameter dict for generator.
            discriminator_type (str): Discriminator type.
            discriminator_params (Dict[str, Any]): Parameter dict for discriminator.
            generator_adv_loss_params (Dict[str, Any]): Parameter dict for generator
                adversarial loss.
            discriminator_adv_loss_params (Dict[str, Any]): Parameter dict for
                discriminator adversarial loss.
            feat_match_loss_params (Dict[str, Any]): Parameter dict for feat match loss.
            mel_loss_params (Dict[str, Any]): Parameter dict for mel loss.
            lambda_adv (float): Loss scaling coefficient for adversarial loss.
            lambda_mel (float): Loss scaling coefficient for mel spectrogram loss.
            lambda_feat_match (float): Loss scaling coefficient for feat match loss.
            lambda_dur (float): Loss scaling coefficient for duration loss.
            lambda_kl (float): Loss scaling coefficient for KL divergence loss.
            cache_generator_outputs (bool): Whether to cache generator outputs.

        """
        assert check_argument_types()
        super().__init__()

        # define modules
        generator_class = AVAILABLE_GENERATERS[generator_type]
        if generator_type == "vits_generator":
            # NOTE(kan-bayashi): Update parameters for the compatibility.
            #   The idim and odim is automatically decided from input data,
            #   where idim represents #vocabularies and odim represents
            #   the input acoustic feature dimension.
            generator_params.update(vocabs=idim, aux_channels=odim)
        self.use_visinger = use_visinger
        self.use_dp = use_dp
        generator_params.update(use_visinger=self.use_visinger)
        generator_params.update(use_dp=self.use_dp)
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
        self.ctc_loss = torch.nn.CTCLoss(idim - 1, reduction="mean")
        self.mse_loss = torch.nn.MSELoss()

        # coefficients
        self.lambda_adv = lambda_adv
        self.lambda_mel = lambda_mel
        self.lambda_kl = lambda_kl
        self.lambda_feat_match = lambda_feat_match
        self.lambda_dur = lambda_dur
        self.lambda_pitch = lambda_pitch
        self.lambda_phoneme = lambda_phoneme

        # cache
        self.cache_generator_outputs = cache_generator_outputs
        self._cache = None

        # store sampling rate for saving wav file
        # (not used for the training)
        self.fs = sampling_rate

        # store parameters for test compatibility
        self.spks = self.generator.spks
        self.langs = self.generator.langs
        self.spk_embed_dim = self.generator.spk_embed_dim

    @property
    def require_raw_singing(self):
        """Return whether or not singing is required."""
        return True

    @property
    def require_vocoder(self):
        """Return whether or not vocoder is required."""
        return False

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        singing: torch.Tensor,
        singing_lengths: torch.Tensor,
        label: Optional[Dict[str, torch.Tensor]] = None,
        label_lengths: Optional[Dict[str, torch.Tensor]] = None,
        melody: Optional[Dict[str, torch.Tensor]] = None,
        melody_lengths: Optional[Dict[str, torch.Tensor]] = None,
        tempo: Optional[Dict[str, torch.Tensor]] = None,
        tempo_lengths: Optional[Dict[str, torch.Tensor]] = None,
        beat: Optional[Dict[str, torch.Tensor]] = None,
        beat_lengths: Optional[Dict[str, torch.Tensor]] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        duration: Optional[Dict[str, torch.Tensor]] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        flag_IsValid=False,
        forward_generator: bool = True,
    ) -> Dict[str, Any]:
        """Perform generator forward.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor): Feature length tensor (B,).
            singing (Tensor): Singing waveform tensor (B, T_wav).
            singing_lengths (Tensor): Singing length tensor (B,).
            sids (Optional[Tensor]): Speaker index tensor (B,) or (B, 1).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, spk_embed_dim).
            lids (Optional[Tensor]): Language index tensor (B,) or (B, 1).
            forward_generator (bool): Whether to forward generator.

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        duration = duration["phn"]
        label = label["lab"]
        label_lengths = label_lengths["lab"]
        melody = melody["lab"]
        melody_lengths = melody_lengths["lab"]
        beat = beat["score_syb"]
        beat_lengths = beat_lengths["score_syb"]
        # print("duration", duration)
        # print("label", label)
        # print("label_lengths", label_lengths)
        # print("text", text)
        # print("text_lengths", text_lengths)
        # print("feats", feats.shape)
        # print("feats_lengths", feats_lengths)
        # print("melody", melody)
        # print("melody_lengths", melody_lengths)
        # print("beat", beat)
        # print("beat_lengths", beat_lengths)
        # print("singing", singing)
        # print("singing_lengths", singing_lengths)
        # print("pitch", pitch.shape)
        # print("pitch_lengths", pitch_lengths)
        # raise ValueError
        if forward_generator:
            return self._forward_generator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                singing=singing,
                singing_lengths=singing_lengths,
                duration=duration,
                label=label,
                label_lengths=label_lengths,
                melody=melody,
                melody_lengths=melody_lengths,
                tempo=tempo,
                tempo_lengths=tempo_lengths,
                beat=beat,
                beat_lengths=beat_lengths,
                pitch=pitch,
                pitch_lengths=pitch_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
            )
        else:
            return self._forward_discrminator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                singing=singing,
                singing_lengths=singing_lengths,
                duration=duration,
                label=label,
                label_lengths=label_lengths,
                melody=melody,
                melody_lengths=melody_lengths,
                tempo=tempo,
                tempo_lengths=tempo_lengths,
                beat=beat,
                beat_lengths=beat_lengths,
                pitch=pitch,
                pitch_lengths=pitch_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
            )

    def _forward_generator(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        singing: torch.Tensor,
        singing_lengths: torch.Tensor,
        duration: torch.Tensor,
        label: Optional[Dict[str, torch.Tensor]] = None,
        label_lengths: Optional[Dict[str, torch.Tensor]] = None,
        melody: Optional[Dict[str, torch.Tensor]] = None,
        melody_lengths: Optional[Dict[str, torch.Tensor]] = None,
        tempo: Optional[Dict[str, torch.Tensor]] = None,
        tempo_lengths: Optional[Dict[str, torch.Tensor]] = None,
        beat: Optional[Dict[str, torch.Tensor]] = None,
        beat_lengths: Optional[Dict[str, torch.Tensor]] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Perform generator forward.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor): Feature length tensor (B,).
            singing (Tensor): Singing waveform tensor (B, T_wav).
            singing_lengths (Tensor): Singing length tensor (B,).
            sids (Optional[Tensor]): Speaker index tensor (B,) or (B, 1).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, spk_embed_dim).
            lids (Optional[Tensor]): Language index tensor (B,) or (B, 1).

        Returns:
            Dict[str, Any]:
                * loss (Tensor): Loss scalar tensor.
                * stats (Dict[str, float]): Statistics to be monitored.
                * weight (Tensor): Weight tensor to summarize losses.
                * optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        # setup
        batch_size = text.size(0)
        feats = feats.transpose(1, 2)
        singing = singing.unsqueeze(1)

        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            outs = self.generator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                duration=duration,
                label=label,
                label_lengths=label_lengths,
                melody=melody,
                melody_lengths=melody_lengths,
                tempo=tempo,
                tempo_lengths=tempo_lengths,
                beat=beat,
                beat_lengths=beat_lengths,
                pitch=pitch,
                pitch_lengths=pitch_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
            )
        else:
            outs = self._cache

        # store cache
        if self.training and self.cache_generator_outputs and not reuse_cache:
            self._cache = outs

        # parse outputs
        singing_hat_, start_idxs, _, z_mask, outs_ = outs
        if not self.use_visinger:
            _, z_p, m_p, logs_p, _, logs_q = outs_
        else:
            if self.use_dp:
                (
                    _,
                    z_p,
                    m_p,
                    logs_p,
                    _,
                    logs_q,
                    pred_pitch,
                    gt_pitch,
                    logw,
                    logw_gt,
                    log_probs,
                ) = outs_
            else:
                _, z_p, m_p, logs_p, _, logs_q, pred_pitch, gt_pitch = outs_
        singing_ = get_segments(
            x=singing,
            start_idxs=start_idxs * self.generator.upsample_factor,
            segment_size=self.generator.segment_size * self.generator.upsample_factor,
        )

        # calculate discriminator outputs
        p_hat = self.discriminator(singing_hat_)
        with torch.no_grad():
            # do not store discriminator gradient in generator turn
            p = self.discriminator(singing_)

        # calculate losses
        with autocast(enabled=False):
            mel_loss = self.mel_loss(singing_hat_, singing_)
            kl_loss = self.kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
            adv_loss = self.generator_adv_loss(p_hat)
            feat_match_loss = self.feat_match_loss(p_hat, p)

            if self.use_visinger:
                pitch_loss = self.mse_loss(pred_pitch, gt_pitch)
                if self.use_dp:
                    ctc_loss = self.ctc_loss(
                        log_probs, label, feats_lengths, label_lengths
                    )
                    dur_loss = self.mse_loss(logw, logw_gt)

            mel_loss = mel_loss * self.lambda_mel
            kl_loss = kl_loss * self.lambda_kl

            adv_loss = adv_loss * self.lambda_adv
            feat_match_loss = feat_match_loss * self.lambda_feat_match
            if self.use_visinger:
                pitch_loss = pitch_loss * self.lambda_pitch
                if self.use_dp:
                    ctc_loss = ctc_loss * self.lambda_phoneme
                    dur_loss = dur_loss * self.lambda_dur
            # loss = mel_loss + kl_loss + dur_loss + adv_loss + feat_match_loss

            loss = mel_loss + kl_loss + adv_loss + feat_match_loss
            if self.use_visinger:
                loss += pitch_loss
            if self.use_dp:
                loss += dur_loss
                loss += ctc_loss

        if self.use_visinger:
            if not self.use_dp:
                stats = dict(
                    generator_loss=loss.item(),
                    generator_mel_loss=mel_loss.item(),
                    generator_kl_loss=kl_loss.item(),
                    generator_adv_loss=adv_loss.item(),
                    generator_feat_match_loss=feat_match_loss.item(),
                    pitch_loss=pitch_loss.item(),
                )
            else:
                stats = dict(
                    generator_loss=loss.item(),
                    generator_mel_loss=mel_loss.item(),
                    generator_kl_loss=kl_loss.item(),
                    generator_dur_loss=dur_loss.item(),
                    generator_adv_loss=adv_loss.item(),
                    generator_feat_match_loss=feat_match_loss.item(),
                    pitch_loss=pitch_loss.item(),
                    ctc_loss=ctc_loss.item(),
                )
        else:
            stats = dict(
                generator_loss=loss.item(),
                generator_mel_loss=mel_loss.item(),
                generator_kl_loss=kl_loss.item(),
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
        singing: torch.Tensor,
        singing_lengths: torch.Tensor,
        duration: torch.Tensor,
        label: Optional[Dict[str, torch.Tensor]] = None,
        label_lengths: Optional[Dict[str, torch.Tensor]] = None,
        melody: Optional[Dict[str, torch.Tensor]] = None,
        melody_lengths: Optional[Dict[str, torch.Tensor]] = None,
        tempo: Optional[Dict[str, torch.Tensor]] = None,
        tempo_lengths: Optional[Dict[str, torch.Tensor]] = None,
        beat: Optional[Dict[str, torch.Tensor]] = None,
        beat_lengths: Optional[Dict[str, torch.Tensor]] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Perform discriminator forward.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor): Feature length tensor (B,).
            singing (Tensor): Singing waveform tensor (B, T_wav).
            singing_lengths (Tensor): Singing length tensor (B,).
            sids (Optional[Tensor]): Speaker index tensor (B,) or (B, 1).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, spk_embed_dim).
            lids (Optional[Tensor]): Language index tensor (B,) or (B, 1).

        Returns:
            Dict[str, Any]:
                * loss (Tensor): Loss scalar tensor.
                * stats (Dict[str, float]): Statistics to be monitored.
                * weight (Tensor): Weight tensor to summarize losses.
                * optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        # setup
        batch_size = text.size(0)
        feats = feats.transpose(1, 2)
        singing = singing.unsqueeze(1)

        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            outs = self.generator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                duration=duration,
                label=label,
                label_lengths=label_lengths,
                melody=melody,
                melody_lengths=melody_lengths,
                tempo=tempo,
                tempo_lengths=tempo_lengths,
                beat=beat,
                beat_lengths=beat_lengths,
                pitch=pitch,
                pitch_lengths=pitch_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
            )
        else:
            outs = self._cache

        # store cache
        if self.cache_generator_outputs and not reuse_cache:
            self._cache = outs

        # parse outputs
        # remove dp loss
        singing_hat_, start_idxs, *_ = outs
        singing_ = get_segments(
            x=singing,
            start_idxs=start_idxs * self.generator.upsample_factor,
            segment_size=self.generator.segment_size * self.generator.upsample_factor,
        )

        # calculate discriminator outputs
        p_hat = self.discriminator(singing_hat_.detach())
        p = self.discriminator(singing_)

        # calculate losses
        with autocast(enabled=False):
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
        feats: Optional[torch.Tensor] = None,
        label: Optional[Dict[str, torch.Tensor]] = None,
        melody: Optional[Dict[str, torch.Tensor]] = None,
        tempo: Optional[Dict[str, torch.Tensor]] = None,
        beat: Optional[Dict[str, torch.Tensor]] = None,
        pitch: Optional[torch.Tensor] = None,
        duration: Optional[Dict[str, torch.Tensor]] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        noise_scale: float = 0.667,
        noise_scale_dur: float = 0.8,
        alpha: float = 1.0,
        max_len: Optional[int] = None,
        use_teacher_forcing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            text (Tensor): Input text index tensor (T_text,).
            feats (Tensor): Feature tensor (T_feats, aux_channels).
            sids (Tensor): Speaker index tensor (1,).
            spembs (Optional[Tensor]): Speaker embedding tensor (spk_embed_dim,).
            lids (Tensor): Language index tensor (1,).
            durations (Tensor): Ground-truth duration tensor (T_text,).
            noise_scale (float): Noise scale value for flow.
            noise_scale_dur (float): Noise scale value for duration predictor.
            alpha (float): Alpha parameter to control the speed of generated singing.
            max_len (Optional[int]): Maximum length.
            use_teacher_forcing (bool): Whether to use teacher forcing.

        Returns:
            Dict[str, Tensor]:
                * wav (Tensor): Generated waveform tensor (T_wav,).
                * att_w (Tensor): Monotonic attention weight tensor (T_feats, T_text).
                * duration (Tensor): Predicted duration tensor (T_text,).

        """
        # setup
        duration = duration["phn"]
        label = label["lab"]
        melody = melody["lab"]
        beat = beat["score_syb"]
        text = text[None]
        text_lengths = torch.tensor(
            [text.size(1)],
            dtype=torch.long,
            device=text.device,
        )
        label_lengths = torch.tensor(
            [label.size(1)],
            dtype=torch.long,
            device=text.device,
        )

        if sids is not None:
            sids = sids.view(1)
        if lids is not None:
            lids = lids.view(1)
        # if durations is not None:
        #     durations = durations.view(1, 1, -1)

        # inference
        if use_teacher_forcing:
            assert feats is not None
            feats = feats[None].transpose(1, 2)
            feats_lengths = torch.tensor(
                [feats.size(2)],
                dtype=torch.long,
                device=feats.device,
            )
            wav, att_w, dur = self.generator.inference(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                label=label,
                label_lengths=label_lengths,
                melody=melody,
                beat=beat,
                sids=sids,
                spembs=spembs,
                lids=lids,
                max_len=max_len,
                use_teacher_forcing=use_teacher_forcing,
            )
        else:
            wav, att_w, dur = self.generator.inference(
                text=text,
                text_lengths=text_lengths,
                label=label,
                label_lengths=label_lengths,
                melody=melody,
                beat=beat,
                sids=sids,
                spembs=spembs,
                lids=lids,
                # dur=durations,
                noise_scale=noise_scale,
                noise_scale_dur=noise_scale_dur,
                alpha=alpha,
                max_len=max_len,
            )
        # return dict(wav=wav.view(-1), att_w=att_w[0], duration=dur[0])
        return dict(wav=wav.view(-1))
