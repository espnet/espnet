# Copyright 2022 Dan Lim
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""JETS module for GAN-TTS task."""

from typing import Any, Dict, Optional

import torch
from typeguard import check_argument_types

from espnet2.gan_tts.abs_gan_tts import AbsGANTTS
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
from espnet2.gan_tts.jets.generator import JETSGenerator
from espnet2.gan_tts.jets.loss import ForwardSumLoss, VarianceLoss
from espnet2.gan_tts.utils import get_segments
from espnet2.torch_utils.device_funcs import force_gatherable

AVAILABLE_GENERATERS = {
    "jets_generator": JETSGenerator,
}
AVAILABLE_DISCRIMINATORS = {
    "hifigan_period_discriminator": HiFiGANPeriodDiscriminator,
    "hifigan_scale_discriminator": HiFiGANScaleDiscriminator,
    "hifigan_multi_period_discriminator": HiFiGANMultiPeriodDiscriminator,
    "hifigan_multi_scale_discriminator": HiFiGANMultiScaleDiscriminator,
    "hifigan_multi_scale_multi_period_discriminator": HiFiGANMultiScaleMultiPeriodDiscriminator,  # NOQA
}


class JETS(AbsGANTTS):
    """JETS module (generator + discriminator).

    This is a module of JETS described in `JETS: Jointly Training FastSpeech2
    and HiFi-GAN for End to End Text to Speech'_.

    .. _`JETS: Jointly Training FastSpeech2 and HiFi-GAN for End to End Text to Speech`
        : https://arxiv.org/abs/2203.16852

    """

    def __init__(
        self,
        # generator related
        idim: int,
        odim: int,
        sampling_rate: int = 22050,
        generator_type: str = "jets_generator",
        generator_params: Dict[str, Any] = {
            "adim": 256,
            "aheads": 2,
            "elayers": 4,
            "eunits": 1024,
            "dlayers": 4,
            "dunits": 1024,
            "positionwise_layer_type": "conv1d",
            "positionwise_conv_kernel_size": 1,
            "use_scaled_pos_enc": True,
            "use_batch_norm": True,
            "encoder_normalize_before": True,
            "decoder_normalize_before": True,
            "encoder_concat_after": False,
            "decoder_concat_after": False,
            "reduction_factor": 1,
            "encoder_type": "transformer",
            "decoder_type": "transformer",
            "transformer_enc_dropout_rate": 0.1,
            "transformer_enc_positional_dropout_rate": 0.1,
            "transformer_enc_attn_dropout_rate": 0.1,
            "transformer_dec_dropout_rate": 0.1,
            "transformer_dec_positional_dropout_rate": 0.1,
            "transformer_dec_attn_dropout_rate": 0.1,
            "conformer_rel_pos_type": "latest",
            "conformer_pos_enc_layer_type": "rel_pos",
            "conformer_self_attn_layer_type": "rel_selfattn",
            "conformer_activation_type": "swish",
            "use_macaron_style_in_conformer": True,
            "use_cnn_in_conformer": True,
            "zero_triu": False,
            "conformer_enc_kernel_size": 7,
            "conformer_dec_kernel_size": 31,
            "duration_predictor_layers": 2,
            "duration_predictor_chans": 384,
            "duration_predictor_kernel_size": 3,
            "duration_predictor_dropout_rate": 0.1,
            "energy_predictor_layers": 2,
            "energy_predictor_chans": 384,
            "energy_predictor_kernel_size": 3,
            "energy_predictor_dropout": 0.5,
            "energy_embed_kernel_size": 1,
            "energy_embed_dropout": 0.5,
            "stop_gradient_from_energy_predictor": False,
            "pitch_predictor_layers": 5,
            "pitch_predictor_chans": 384,
            "pitch_predictor_kernel_size": 5,
            "pitch_predictor_dropout": 0.5,
            "pitch_embed_kernel_size": 1,
            "pitch_embed_dropout": 0.5,
            "stop_gradient_from_pitch_predictor": True,
            "generator_out_channels": 1,
            "generator_channels": 512,
            "generator_global_channels": -1,
            "generator_kernel_size": 7,
            "generator_upsample_scales": [8, 8, 2, 2],
            "generator_upsample_kernel_sizes": [16, 16, 4, 4],
            "generator_resblock_kernel_sizes": [3, 7, 11],
            "generator_resblock_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "generator_use_additional_convs": True,
            "generator_bias": True,
            "generator_nonlinear_activation": "LeakyReLU",
            "generator_nonlinear_activation_params": {"negative_slope": 0.1},
            "generator_use_weight_norm": True,
            "segment_size": 64,
            "spks": -1,
            "langs": -1,
            "spk_embed_dim": None,
            "spk_embed_integration_type": "add",
            "use_gst": False,
            "gst_tokens": 10,
            "gst_heads": 4,
            "gst_conv_layers": 6,
            "gst_conv_chans_list": [32, 32, 64, 64, 128, 128],
            "gst_conv_kernel_size": 3,
            "gst_conv_stride": 2,
            "gst_gru_layers": 1,
            "gst_gru_units": 128,
            "init_type": "xavier_uniform",
            "init_enc_alpha": 1.0,
            "init_dec_alpha": 1.0,
            "use_masking": False,
            "use_weighted_masking": False,
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
        lambda_var: float = 1.0,
        lambda_align: float = 2.0,
        cache_generator_outputs: bool = True,
        plot_pred_mos: bool = False,
        mos_pred_tool: str = "utmos",
    ):
        """Initialize JETS module.

        Args:
            idim (int): Input vocabrary size.
            odim (int): Acoustic feature dimension. The actual output channels will
                be 1 since JETS is the end-to-end text-to-wave model but for the
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
            lambda_var (float): Loss scaling coefficient for variance loss.
            lambda_align (float): Loss scaling coefficient for alignment loss.
            cache_generator_outputs (bool): Whether to cache generator outputs.
            plot_pred_mos (bool): Whether to plot predicted MOS during the training.
            mos_pred_tool (str): MOS prediction tool name.
        """
        assert check_argument_types()
        super().__init__()

        # define modules
        generator_class = AVAILABLE_GENERATERS[generator_type]
        generator_params.update(idim=idim, odim=odim)
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
        self.var_loss = VarianceLoss()
        self.forwardsum_loss = ForwardSumLoss()

        # coefficients
        self.lambda_adv = lambda_adv
        self.lambda_mel = lambda_mel
        self.lambda_feat_match = lambda_feat_match
        self.lambda_var = lambda_var
        self.lambda_align = lambda_align

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
        self.use_gst = getattr(self.generator, "use_gst", False)

        # plot pseudo mos during training
        self.plot_pred_mos = plot_pred_mos
        if plot_pred_mos:
            if mos_pred_tool == "utmos":
                # Load predictor for UTMOS22 (https://arxiv.org/abs/2204.02152)
                self.predictor = torch.hub.load(
                    "tarepan/SpeechMOS:v1.2.0", "utmos22_strong"
                )
            else:
                raise NotImplementedError(
                    f"Not supported mos_pred_tool: {mos_pred_tool}"
                )

    @property
    def require_raw_speech(self):
        """Return whether or not speech is required."""
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
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        sids: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        forward_generator: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform generator forward.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor): Feature length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
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
        if forward_generator:
            return self._forward_generator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                speech=speech,
                speech_lengths=speech_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
                **kwargs,
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
                spembs=spembs,
                lids=lids,
                **kwargs,
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
        spembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform generator forward.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor): Feature length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
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
        speech = speech.unsqueeze(1)

        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            outs = self.generator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
                **kwargs,
            )
        else:
            outs = self._cache

        # store cache
        if self.training and self.cache_generator_outputs and not reuse_cache:
            self._cache = outs

        # parse outputs
        (
            speech_hat_,
            bin_loss,
            log_p_attn,
            start_idxs,
            d_outs,
            ds,
            p_outs,
            ps,
            e_outs,
            es,
        ) = outs
        speech_ = get_segments(
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
        adv_loss = self.generator_adv_loss(p_hat)
        feat_match_loss = self.feat_match_loss(p_hat, p)
        dur_loss, pitch_loss, energy_loss = self.var_loss(
            d_outs, ds, p_outs, ps, e_outs, es, text_lengths
        )
        forwardsum_loss = self.forwardsum_loss(log_p_attn, text_lengths, feats_lengths)

        mel_loss = mel_loss * self.lambda_mel
        adv_loss = adv_loss * self.lambda_adv
        feat_match_loss = feat_match_loss * self.lambda_feat_match
        g_loss = mel_loss + adv_loss + feat_match_loss
        var_loss = (dur_loss + pitch_loss + energy_loss) * self.lambda_var
        align_loss = (forwardsum_loss + bin_loss) * self.lambda_align

        loss = g_loss + var_loss + align_loss

        stats = dict(
            generator_loss=loss.item(),
            generator_g_loss=g_loss.item(),
            generator_var_loss=var_loss.item(),
            generator_align_loss=align_loss.item(),
            generator_g_mel_loss=mel_loss.item(),
            generator_g_adv_loss=adv_loss.item(),
            generator_g_feat_match_loss=feat_match_loss.item(),
            generator_var_dur_loss=dur_loss.item(),
            generator_var_pitch_loss=pitch_loss.item(),
            generator_var_energy_loss=energy_loss.item(),
            generator_align_forwardsum_loss=forwardsum_loss.item(),
            generator_align_bin_loss=bin_loss.item(),
        )

        if self.plot_pred_mos:
            # Caltulate predicted MOS from generated speech waveform.
            with torch.no_grad():
                # speech_hat_: (B, 1, T)
                pmos = self.predictor(speech_hat_.squeeze(1), self.fs).mean()
            stats["generator_predicted_mos"] = pmos.item()

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
        spembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform discriminator forward.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor): Feature length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
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
        speech = speech.unsqueeze(1)

        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            outs = self.generator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
                **kwargs,
            )
        else:
            outs = self._cache

        # store cache
        if self.cache_generator_outputs and not reuse_cache:
            self._cache = outs

        # parse outputs
        speech_hat_, _, _, start_idxs, *_ = outs
        speech_ = get_segments(
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
        feats: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        use_teacher_forcing: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            text (Tensor): Input text index tensor (T_text,).
            feats (Tensor): Feature tensor (T_feats, aux_channels).
            pitch (Tensor): Pitch tensor (T_feats, 1).
            energy (Tensor): Energy tensor (T_feats, 1).
            use_teacher_forcing (bool): Whether to use teacher forcing.

        Returns:
            Dict[str, Tensor]:
                * wav (Tensor): Generated waveform tensor (T_wav,).
                * duration (Tensor): Predicted duration tensor (T_text,).

        """
        # setup
        text = text[None]
        text_lengths = torch.tensor(
            [text.size(1)],
            dtype=torch.long,
            device=text.device,
        )
        if "spembs" in kwargs:
            kwargs["spembs"] = kwargs["spembs"][None]
        if self.use_gst and "speech" in kwargs:
            # NOTE(kan-bayashi): Workaround for the use of GST
            kwargs.pop("speech")

        # inference
        if use_teacher_forcing:
            assert feats is not None
            feats = feats[None]
            feats_lengths = torch.tensor(
                [feats.size(1)],
                dtype=torch.long,
                device=feats.device,
            )
            assert pitch is not None
            pitch = pitch[None]
            assert energy is not None
            energy = energy[None]

            wav, dur = self.generator.inference(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                pitch=pitch,
                energy=energy,
                use_teacher_forcing=use_teacher_forcing,
                **kwargs,
            )
        else:
            wav, dur = self.generator.inference(
                text=text,
                text_lengths=text_lengths,
                feats=feats[None] if self.use_gst else None,
                **kwargs,
            )
        return dict(wav=wav.view(-1), duration=dur[0])
