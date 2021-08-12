# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""VITS: Variational Inference with adversarial learning for end-to-end Text-to-Speech.

This code is based on https://github.com/jaywalnut310/vits.

"""

import math

from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.gan_tts.abs_gan_tts import AbsGANTTS
from espnet2.gan_tts.vits.flow import ResidualAffineCouplingBlock
from espnet2.gan_tts.vits.hifigan import HiFiGANGenerator
from espnet2.gan_tts.vits.hifigan import HiFiGANMultiScaleMultiPeriodDiscriminator
from espnet2.gan_tts.vits.loss import DiscriminatorAdversarialLoss
from espnet2.gan_tts.vits.loss import FeatureMatchLoss
from espnet2.gan_tts.vits.loss import GeneratorAdversarialLoss
from espnet2.gan_tts.vits.loss import KLDivergenceLoss
from espnet2.gan_tts.vits.loss import MelSpectrogramLoss
from espnet2.gan_tts.vits.posterior_encoder import PosteriorEncoder
from espnet2.gan_tts.vits.stochastic_duration_predictor import StochasticDurationPredictor
from espnet2.gan_tts.vits.text_encoder import TextEncoder


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
        generator_params.update(idim=idim, odim=odim)
        self.generator = VITSGenerator(
            **generator_params,
        )
        self.discriminator = HiFiGANMultiScaleMultiPeriodDiscriminator(
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

        self.lambda_adv = lambda_adv
        self.lambda_mel = lambda_mel
        self.lambda_kl = lambda_kl
        self.lambda_feat_match = lambda_feat_match
        self.lambda_dur = lambda_dur

    def forward_generator(
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
            feats (Tensor): Feature tensor (B, aux_channels, T_feats).
            feats_lengths (Tensor): Feature length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, 1, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            sids (Optional[Tensor]): Speaker index tensor (B,).

        Returns:
            Dict[str, Any]:
                - Tensor: Loss scalar tensor.
                - Dict[str, float]: Statistics to be monitored.
                - Tensor: Weight value.
                - int: Optimizer index (0 for generator).

        """
        batch_size = text.size(0)
        (
            p_hat,
            p,
            speech_hat_,
            speech_,
            dur_nll,
            z_p,
            m_p,
            logs_p,
            logs_q,
            z_mask,
        ) = self.forward(
            text=text,
            text_lengths=text_lengths,
            feats=feats,
            feats_lengths=feats_lengths,
            speech=speech,
            speech_lengths=speech_lengths,
            sids=sids,
            is_generator=True,
        )
        mel_loss = self.mel_loss(speech_hat_, speech_)
        kl_loss = self.kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
        dur_loss = torch.sum(dur_nll.float())
        adv_loss = self.generator_adv_loss(p_hat)
        feat_match_loss = self.feat_match_loss(p_hat, p)

        mel_loss *= self.lambda_mel
        kl_loss *= self.lambda_kl
        dur_loss *= self.lambda_dur
        adv_loss *= self.lambda_adv
        feat_match_loss *= self.lambda_feat_match
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
        return {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 0,  # needed for trainer
        }

    def forward_discrminator(
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
            feats (Tensor): Feature tensor (B, aux_channels, T_feats).
            feats_lengths (Tensor): Feature length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, 1, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            sids (Optional[Tensor]): Speaker index tensor (B,).

        Returns:
            Dict[str, Any]:
                - Tensor: Loss scalar tensor.
                - Dict[str, float]: Statistics to be monitored.
                - Tensor: Weight value.
                - int: Optimizer index (1 for discriminator).

        """
        batch_size = text.size(0)
        p_hat, p = self.forward(
            text=text,
            text_lengths=text_lengths,
            feats=feats,
            feats_lengths=feats_lengths,
            speech=speech,
            speech_lengths=speech_lengths,
            sids=sids,
            is_generator=False,
        )
        real_loss, fake_loss = self.discriminator_adv_loss(p_hat, p)
        loss = real_loss + fake_loss

        stats = dict(
            discriminator_loss=loss.item(),
            discriminator_real_loss=real_loss.item(),
            discriminator_fake_loss=fake_loss.item(),
        )
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 1,  # needed for trainer
        }

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        sids: Optional[torch.Tensor] = None,
        is_generator: bool = True,
    ):
        """Perform forward calculation.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, aux_channels, T_feats).
            feats_lengths (Tensor): Feature length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, 1, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            sids (Optional[Tensor]): Speaker index tensor (B,).
            is_generator (bool): For generator forward or not.

        Returns:
            List[List[Tensor]]: List of list of discriminator outputs for fake samples.
            List[List[Tensor]]: List of list of discriminator outputs for real samples.
            Tensor: Fake waveform tensor (B, 1, segment_size * upsample_factor).
                Only provided for when is_generator = True.
            Tensor: Real waveform tensor (B, 1, segment_size * upsample_factor).
                Only provided for when is_generator = True.
            Tensor: Duration negative lower bound (B,).
                Only provided for when is_generator = True.
            Tensor: Flow hidden representation (B, H, T_feats).
                Only provided for when is_generator = True.
            Tensor: Expanded text encoder VAE mean (B, H, T_feats).
                Only provided for when is_generator = True.
            Tensor: Expanded text encoder VAE scale (B, H, T_feats).
                Only provided for when is_generator = True.
            Tensor: Posterior encoder VAE scale (B, H, T_feats).
                Only provided for when is_generator = True.
            Tensor: Feature mask tensor (B, 1, T_feats).
                Only provided for when is_generator = True.

        """
        if is_generator:
            outs = self.generator(text, text_lengths, feats, feats_lengths, sids)
            speech_hat_, dur_nll, _, start_idxs, _, z_mask, outs_ = outs
            _, z_p, m_p, logs_p, _, logs_q = outs_
            start_idxs *= self.generator.upsample_factor
            segment_size = self.generator.segment_size * self.generator.upsample_factor
            speech_ = self.generator.get_segments(
                x=speech,
                start_idxs=start_idxs,
                segment_size=segment_size,
            )
            p_hat = self.discriminator(speech_hat_)
            with torch.no_grad():
                # do not store discriminator gradient in generator turn
                p = self.discriminator(speech_)

            return (
                p_hat,
                p,
                speech_hat_,
                speech_,
                dur_nll,
                z_p,
                m_p,
                logs_p,
                logs_q,
                z_mask,
            )
        else:
            with torch.no_grad():
                # do not store generator gradient in generator turn
                speech_hat_, _, _, start_idxs, *_ = self.generator(
                    text, text_lengths, feats, feats_lengths, sids
                )
            start_idxs *= self.generator.upsample_factor
            segment_size = self.generator.segment_size * self.generator.upsample_factor
            speech_ = self.generator.get_segments(
                x=speech,
                start_idxs=start_idxs,
                segment_size=segment_size,
            )
            p_hat = self.discriminator(speech_hat_.detach())
            p = self.discriminator(speech_)

            return p_hat, p

    def inference(
        self,
        text,
        sids=None,
        durations=None,
        noise_scale=1.0,
        length_scale=1.0,
        noise_scale_w=1.0,
        max_len=None,
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


class VITSGenerator(torch.nn.Module):
    """VITS generator module.

    This is a module of VITS described in `Conditional Variational Autoencoder
    with Adversarial Learning for End-to-End Text-to-Speech`_.

    .. _`Conditional Variational Autoencoder with Adversarial Learning for End-to-End
        Text-to-Speech`: https://arxiv.org/abs/2006.04558

    """

    def __init__(
        self,
        idim,
        odim,
        aux_channels=80,
        hidden_channels=192,
        spks=-1,
        global_channels=-1,
        segment_size=32,
        text_encoder_attention_heads=2,
        text_encoder_attention_expand=4,
        text_encoder_blocks=6,
        text_encoder_kernel_size=3,
        text_encoder_dropout_rate=0.1,
        text_encoder_positional_dropout_rate=0.0,
        text_encoder_attention_dropout_rate=0.0,
        decoder_kernel_size=7,
        decoder_channels=512,
        decoder_upsample_scales=(8, 8, 2, 2),
        decoder_upsample_kernel_sizes=(16, 16, 4, 4),
        decoder_resblock_kernel_sizes=(3, 7, 11),
        decoder_resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        use_weight_norm_in_decoder=True,
        posterior_encoder_kernel_size=5,
        posterior_encoder_layers=16,
        posterior_encoder_stacks=1,
        posterior_encoder_base_dilation=1,
        posterior_encoder_dropout_rate=0.0,
        use_weight_norm_in_posterior_encoder=True,
        flow_flows=4,
        flow_kernel_size=5,
        flow_base_dilation=1,
        flow_layers=5,
        flow_dropout_rate=0.0,
        use_weight_norm_in_flow=True,
        use_only_mean_in_flow=True,
        stochastic_duration_predictor_kernel_size=3,
        stochastic_duration_predictor_dropout_rate=0.5,
        stochastic_duration_predictor_flows=4,
        stochastic_duration_predictor_dds_conv_layers=3,
    ):
        """Initialize VITS module.

        Args:
            idim (int): Input dimension.
            odim (int): Output dimension.
            aux_channels (int): Number of auxiliary feature channels.
            hidden_channels (int): Number of hidden channels.
            spks (int): Number of speakers.
            global_channels (int): Number of global conditioning channels.
            segment_size (int): Segment size for decoder.
            text_encoder_attention_heads (int): Number of heads in text encoder.
            text_encoder_attention_expand (int): Expansion number in text encoder.
            text_encoder_blocks (int): Number of blocks in text encoder.
            text_encoder_kernel_size (int): Kernel size in text encoder.
            text_encoder_dropout_rate (float): Dropout rate in text encoder.
            text_encoder_positional_dropout_rate (float): Dropout rate for positional
                encoding in text encoder.
            text_encoder_attention_dropout_rate (float): Dropout rate for attention in
                text encoder.
            decoder_kernel_size (int): Decoder kernel size.
            decoder_channels (int): Number of decoder initial channels.
            decoder_upsample_scales (list): List of upsampling scales in decoder.
            decoder_upsample_kernel_sizes (list): List of kernel size for upsampling
                layers in decoder.
            decoder_resblock_kernel_sizes (list): List of kernel size for resblocks in
                decoder.
            decoder_resblock_dilations (list): List of list of dilations for resblocks
                in decoder.
            use_weight_norm_in_decoder (bool): Whether to apply weight normalization in
                decoder.
            posterior_encoder_kernel_size (int): Posterior encoder kernel size.
            posterior_encoder_layers (int): Number of layers of posterior encoder.
            posterior_encoder_stacks (int): Number of stacks of posterior encoder.
            posterior_encoder_base_dilation (int): Base dilation of posterior encoder.
            posterior_encoder_dropout_rate (float): Dropout rate for posterior encoder.
            use_weight_norm_in_posterior_encoder (bool): Whether to apply weight
                normalization in posterior encoder.
            flow_flows (int): Number of flows in flow.
            flow_kernel_size (int): Kernel size in flow.
            flow_base_dilation (int): Base dilation in flow.
            flow_layers (int): Number of layers in flow.
            flow_dropout_rate (float): Dropout rate in flow
            use_weight_norm_in_flow (bool): Whether to apply weight normalization in
                flow.
            use_only_mean_in_flow (bool): Whether to use only mean in flow.
            stochastic_duration_predictor_kernel_size (int): Kernel size in stochastic
                duration predictor.
            stochastic_duration_predictor_dropout_rate (float): Dropout rate in
                stochastic duration predictor.
            stochastic_duration_predictor_flows (int): Number of flows in stochastic
                duration predictor.
            stochastic_duration_predictor_dds_conv_layers (int): Number of DDS conv
                layers in stochastic duration predictor.

        """
        super().__init__()
        self.segment_size = segment_size
        self.text_encoder = TextEncoder(
            vocabs=idim,
            attention_dim=hidden_channels,
            attention_heads=text_encoder_attention_heads,
            linear_units=hidden_channels * text_encoder_attention_expand,
            blocks=text_encoder_blocks,
            positionwise_conv_kernel_size=text_encoder_kernel_size,
            dropout_rate=text_encoder_dropout_rate,
            positional_dropout_rate=text_encoder_positional_dropout_rate,
            attention_dropout_rate=text_encoder_attention_dropout_rate,
        )
        self.decoder = HiFiGANGenerator(
            in_channels=hidden_channels,
            out_channels=1,
            channels=decoder_channels,
            kernel_size=decoder_kernel_size,
            upsample_scales=decoder_upsample_scales,
            upsample_kernal_sizes=decoder_upsample_kernel_sizes,
            resblock_kernel_sizes=decoder_resblock_kernel_sizes,
            resblock_dilations=decoder_resblock_dilations,
            use_weight_norm=use_weight_norm_in_decoder,
        )
        self.posterior_encoder = PosteriorEncoder(
            in_channels=aux_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=posterior_encoder_kernel_size,
            layers=posterior_encoder_layers,
            stacks=posterior_encoder_stacks,
            base_dilation=posterior_encoder_base_dilation,
            global_channels=global_channels,
            dropout_rate=posterior_encoder_dropout_rate,
            use_weight_norm=use_weight_norm_in_posterior_encoder,
        )
        self.flow = ResidualAffineCouplingBlock(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            flows=flow_flows,
            kernel_size=flow_kernel_size,
            base_dilation=flow_base_dilation,
            layers=flow_layers,
            global_channels=global_channels,
            dropout_rate=flow_dropout_rate,
            use_weight_norm=use_weight_norm_in_flow,
            use_only_mean=use_only_mean_in_flow,
        )
        self.duration_predictor = StochasticDurationPredictor(
            channels=hidden_channels,
            kernel_size=stochastic_duration_predictor_kernel_size,
            dropout_rate=stochastic_duration_predictor_dropout_rate,
            flows=stochastic_duration_predictor_flows,
            dds_conv_layers=stochastic_duration_predictor_dds_conv_layers,
            global_channels=global_channels,
        )

        self.upsample_factor = np.prod(decoder_upsample_scales)
        self.spks = spks
        if self.spks > 1:
            assert global_channels > 0
            self.global_emb = torch.nn.Embedding(spks, global_channels)

        # delayed import
        from espnet2.gan_tts.vits.monotonic_align import maximum_path

        self.maximum_path = maximum_path

    def forward(
        self,
        text,
        text_lengths,
        feats,
        feats_lengths,
        sids=None,
    ):
        """Calculate forward propagation.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, aux_channels, T_feats).
            feats_lengths (Tensor): Feature length tensor (B,).
            sids (Optional[Tensor]): Speaker index tensor (B,).

        Returns:
            Tensor: Waveform tensor (B, 1, segment_size * upsample_factor).
            Tensor: Duration negative lower bound tensor (B,).
            Tensor: Monotonic attention weight tensor (B, 1, T_feats, T_text).
            Tensor: Segments start index tensor (B,).
            Tensor: Text mask tensor (B, 1, T_text).
            Tensor: Feature mask tensor (B, 1, T_feats).
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
                - Tensor: Posterior encoder hidden representation (B, H, T_feats).
                - Tensor: Flow hidden representation (B, H, T_feats).
                - Tensor: Expanded text encoder VAE mean (B, H, T_feats).
                - Tensor: Expanded text encoder VAE scale (B, H, T_feats).
                - Tensor: Posterior encoder VAE mean (B, H, T_feats).
                - Tensor: Posterior encoder VAE scale (B, H, T_feats).

        """
        # forward text encoder
        x, m_p, logs_p, x_mask = self.text_encoder(text, text_lengths)

        # calculate global conditioning
        if self.spks > 0:
            g = self.global_enb(sids).unsqueeze(-1)  # (B, global_channels, 1)
        else:
            g = None

        # forward posterior encoder
        z, m_q, logs_q, y_mask = self.posterior_encoder(feats, feats_lengths, g=g)

        # forward flow
        z_p = self.flow(z, y_mask, g=g)  # (B, H, T_feats)

        # monotonic alignment search
        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # (B, H, T_text)
            # (B, 1, T_text)
            neg_x_ent_1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p,
                [1],
                keepdim=True,
            )
            # (B, T_feats, H) x (B, H, T_text) = (B, T_feats, T_text)
            neg_x_ent_2 = torch.matmul(
                -0.5 * (z_p ** 2).transpose(1, 2),
                s_p_sq_r,
            )
            # (B, T_feats, H) x (B, H, T_text) = (B, T_feats, T_text)
            neg_x_ent_3 = torch.matmul(
                z_p.transpose(1, 2),
                (m_p * s_p_sq_r),
            )
            # (B, 1, T_text)
            neg_x_ent_4 = torch.sum(
                -0.5 * (m_p ** 2) * s_p_sq_r,
                [1],
                keepdim=True,
            )
            # (B, T_feats, T_text)
            neg_x_ent = neg_x_ent_1 + neg_x_ent_2 + neg_x_ent_3 + neg_x_ent_4
            # (B, 1, T_feats, T_text)
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            # monotonic attention weight: (B, 1, T_feats, T_text)
            attn = (
                self.maximum_path(
                    neg_x_ent,
                    attn_mask.squeeze(1),
                )
                .unsqueeze(1)
                .detach()
            )

        # get durations
        w = attn.sum(2)
        dur_nll = self.duration_predictor(x, x_mask, w=w, g=g)
        dur_nll = dur_nll / torch.sum(x_mask)

        # expand the length to match with the feature sequence
        # (B, T_feats, T_text) x (B, T_text, H) -> (B, H, T_feats)
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        # (B, T_feats, T_text) x (B, T_text, H) -> (B, H, T_feats)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        # forward decoder with segments
        z_segments, z_start_idxs = self.get_random_segments(
            z,
            feats_lengths,
            self.segment_size,
        )
        wav = self.decoder(z_segments, g=g)

        return (
            wav,
            dur_nll,
            attn,
            z_start_idxs,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
        )

    def get_random_segments(self, x, x_lengths=None, segment_size=32):
        """Get random segments.

        Args:
            x (Tensor): Input tensor (B, C, T).
            x_lengths (Tensor): Length tensor (B,).
            segment_size (int): Segment size.

        Returns:
            Tensor: Segmented tensor (B, C, segment_size).
            Tensor: Start index tensor (B,).

        """
        b, c, t = x.size()
        if x_lengths is None:
            x_lengths = t
        max_start_idx = x_lengths - segment_size + 1
        start_idxs = (torch.rand([b]).to(x.device) * max_start_idx).to(
            dtype=torch.long,
        )
        segments = self.get_segments(x, start_idxs, segment_size)
        return segments, start_idxs

    def get_segments(self, x, start_idxs, segment_size=32):
        """Get segments.

        Args:
            x (Tensor): Input tensor (B, C, T).
            start_idxs (Tensor): Start index tensor (B,).
            segment_size (int): Segment size.

        Returns:
            Tensor: Segmented tensor (B, C, segment_size).

        """
        b, c, t = x.size()
        segments = x.new_zeros(b, c, segment_size)
        for i, start_idx in enumerate(start_idxs):
            segments[i] = x[i, :, start_idx : start_idx + segment_size]
        return segments

    def inference(
        self,
        text,
        text_lengths,
        sids=None,
        dur=None,
        noise_scale=1.0,
        length_scale=1.0,
        noise_scale_w=1.0,
        max_len=None,
    ):
        """Run inference.

        Args:
            text (Tensor): Input text index tensor (B, T_text,).
            text_lengths (Tensor): Text length tensor (B,).
            sid (Optional[Tensor]): Speaker index tensor (B,).
            dur (Optional[Tensor]): Ground-truth duration (B, T_text,). If provided,
                skip the prediction of durations (i.e., teacher forcing).
            noise_scale (float): Noise scale value for flow.
            length_scale (float): Length scaling value.
            noise_scale_w (float): Noise scale value for duration predictor.
            max_len (Optional[int]): Maximum length.

        Returns:
            Tensor: Generated waveform tensor (B, T_wav).
            Tensor: Attention weight tensor (B, T_feats, T_text).
            Tuple[Tensor, Tensor, Tensor, Tensor]:
                - Tensor: Flow-inversed hidden representation tensor (B, H, T_feats).
                - Tensor: Sampled hidden representation (B, H, T_feats).
                - Tensor: Expanded text encoder VAE mean (B, H, T_feats).
                - Tensor: Expanded text encoder VAE scale (B, H, T_feats).

        """
        # encoder
        x, m_p, logs_p, x_mask = self.text_encoder(text, text_lengths)
        if self.spks > 0:
            g = self.global_emb(sids).unsqueeze(-1)  # (B, global_channels, 1)
        else:
            g = None

        # duration
        if dur is None:
            logw = self.duration_predictor(
                x,
                x_mask,
                g=g,
                inverse=True,
                noise_scale=noise_scale_w,
            )
            w = torch.exp(logw) * x_mask * length_scale
            dur = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(dur, [1, 2]), 1).long()
        y_mask = make_non_pad_mask(y_lengths).unsqueeze(1).to(text.device)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = self._generate_path(dur, attn_mask)

        # expand the length to match with the feature sequence
        # (B, T_feats, T_text) x (B, T_text, H) -> (B, H, T_feats)
        m_p = torch.matmul(
            attn.squeeze(1),
            m_p.transpose(1, 2),
        ).transpose(1, 2)
        # (B, T_feats, T_text) x (B, T_text, H) -> (B, H, T_feats)
        logs_p = torch.matmul(
            attn.squeeze(1),
            logs_p.transpose(1, 2),
        ).transpose(1, 2)

        # decoder
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, inverse=True)
        wav = self.decoder((z * y_mask)[:, :, :max_len], g=g)

        return wav.squeeze(1), attn.squeeze(1), (z, z_p, m_p, logs_p)

    def _generate_path(self, dur, mask):
        """Generate path.

        Args:
            dur (Tensor): Duration tensor (B, 1, T_text).
            mask (Tensor): Attention mask tensor (B, 1, T_feats, T_text).

        Returns:
            Tensor: Path tensor (B, 1, T_feats, T_text).

        """
        b, _, t_y, t_x = mask.shape
        cum_dur = torch.cumsum(dur, -1)
        cum_dur_flat = cum_dur.view(b * t_x)
        path = torch.arange(t_y, dtype=dur.dtype, device=dur.device)
        path = path.unsqueeze(0) < cum_dur_flat.unsqueeze(1)
        path = path.view(b, t_x, t_y).to(dtype=mask.dtype)
        # path will be like (t_x = 3, t_y = 5):
        # [[[1., 1., 0., 0., 0.],      [[[1., 1., 0., 0., 0.],
        #   [1., 1., 1., 1., 0.],  -->   [0., 0., 1., 1., 0.],
        #   [1., 1., 1., 1., 1.]]]       [0., 0., 0., 0., 1.]]]
        path = path - F.pad(path, [0, 0, 1, 0, 0, 0])[:, :-1]
        return path.unsqueeze(1).transpose(2, 3) * mask
