# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Joint text-to-wav module for end-to-end training."""

from typing import Any, Dict

import torch
from typeguard import check_argument_types

from espnet2.gan_tts.abs_gan_tts import AbsGANTTS
from espnet2.gan_tts.hifigan import (
    HiFiGANGenerator,
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
from espnet2.gan_tts.melgan import MelGANGenerator, MelGANMultiScaleDiscriminator
from espnet2.gan_tts.melgan.pqmf import PQMF
from espnet2.gan_tts.parallel_wavegan import (
    ParallelWaveGANDiscriminator,
    ParallelWaveGANGenerator,
)
from espnet2.gan_tts.style_melgan import StyleMelGANDiscriminator, StyleMelGANGenerator
from espnet2.gan_tts.utils import get_random_segments, get_segments
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.tts.fastspeech import FastSpeech
from espnet2.tts.fastspeech2 import FastSpeech2
from espnet2.tts.tacotron2 import Tacotron2
from espnet2.tts.transformer import Transformer

AVAILABLE_TEXT2MEL = {
    "tacotron2": Tacotron2,
    "transformer": Transformer,
    "fastspeech": FastSpeech,
    "fastspeech2": FastSpeech2,
}
AVAILABLE_VOCODER = {
    "hifigan_generator": HiFiGANGenerator,
    "melgan_generator": MelGANGenerator,
    "parallel_wavegan_generator": ParallelWaveGANGenerator,
    "style_melgan_generator": StyleMelGANGenerator,
}
AVAILABLE_DISCRIMINATORS = {
    "hifigan_period_discriminator": HiFiGANPeriodDiscriminator,
    "hifigan_scale_discriminator": HiFiGANScaleDiscriminator,
    "hifigan_multi_period_discriminator": HiFiGANMultiPeriodDiscriminator,
    "hifigan_multi_scale_discriminator": HiFiGANMultiScaleDiscriminator,
    "hifigan_multi_scale_multi_period_discriminator": HiFiGANMultiScaleMultiPeriodDiscriminator,  # NOQA
    "melgan_multi_scale_discriminator": MelGANMultiScaleDiscriminator,
    "parallel_wavegan_discriminator": ParallelWaveGANDiscriminator,
    "style_melgan_discriminator": StyleMelGANDiscriminator,
}


class JointText2Wav(AbsGANTTS):
    """General class to jointly train text2mel and vocoder parts."""

    def __init__(
        self,
        # generator (text2mel + vocoder) related
        idim: int,
        odim: int,
        segment_size: int = 32,
        sampling_rate: int = 22050,
        text2mel_type: str = "fastspeech2",
        text2mel_params: Dict[str, Any] = {
            "adim": 384,
            "aheads": 2,
            "elayers": 4,
            "eunits": 1536,
            "dlayers": 4,
            "dunits": 1536,
            "postnet_layers": 5,
            "postnet_chans": 512,
            "postnet_filts": 5,
            "postnet_dropout_rate": 0.5,
            "positionwise_layer_type": "conv1d",
            "positionwise_conv_kernel_size": 1,
            "use_scaled_pos_enc": True,
            "use_batch_norm": True,
            "encoder_normalize_before": True,
            "decoder_normalize_before": True,
            "encoder_concat_after": False,
            "decoder_concat_after": False,
            "reduction_factor": 1,
            "encoder_type": "conformer",
            "decoder_type": "conformer",
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
        vocoder_type: str = "hifigan_generator",
        vocoder_params: Dict[str, Any] = {
            "out_channels": 1,
            "channels": 512,
            "global_channels": -1,
            "kernel_size": 7,
            "upsample_scales": [8, 8, 2, 2],
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "use_additional_convs": True,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
        },
        use_pqmf: bool = False,
        pqmf_params: Dict[str, Any] = {
            "subbands": 4,
            "taps": 62,
            "cutoff_ratio": 0.142,
            "beta": 9.0,
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
        use_feat_match_loss: bool = True,
        feat_match_loss_params: Dict[str, Any] = {
            "average_by_discriminators": False,
            "average_by_layers": False,
            "include_final_outputs": True,
        },
        use_mel_loss: bool = True,
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
        lambda_text2mel: float = 1.0,
        lambda_adv: float = 1.0,
        lambda_feat_match: float = 2.0,
        lambda_mel: float = 45.0,
        cache_generator_outputs: bool = False,
    ):
        """Initialize JointText2Wav module.

        Args:
            idim (int): Input vocabrary size.
            odim (int): Acoustic feature dimension. The actual output channels will
                be 1 since the model is the end-to-end text-to-wave model but for the
                compatibility odim is used to indicate the acoustic feature dimension.
            segment_size (int): Segment size for random windowed inputs.
            sampling_rate (int): Sampling rate, not used for the training but it will
                be referred in saving waveform during the inference.
            text2mel_type (str): The text2mel model type.
            text2mel_params (Dict[str, Any]): Parameter dict for text2mel model.
            use_pqmf (bool): Whether to use PQMF for multi-band vocoder.
            pqmf_params (Dict[str, Any]): Parameter dict for PQMF module.
            vocoder_type (str): The vocoder model type.
            vocoder_params (Dict[str, Any]): Parameter dict for vocoder model.
            discriminator_type (str): Discriminator type.
            discriminator_params (Dict[str, Any]): Parameter dict for discriminator.
            generator_adv_loss_params (Dict[str, Any]): Parameter dict for generator
                adversarial loss.
            discriminator_adv_loss_params (Dict[str, Any]): Parameter dict for
                discriminator adversarial loss.
            use_feat_match_loss (bool): Whether to use feat match loss.
            feat_match_loss_params (Dict[str, Any]): Parameter dict for feat match loss.
            use_mel_loss (bool): Whether to use mel loss.
            mel_loss_params (Dict[str, Any]): Parameter dict for mel loss.
            lambda_text2mel (float): Loss scaling coefficient for text2mel model loss.
            lambda_adv (float): Loss scaling coefficient for adversarial loss.
            lambda_feat_match (float): Loss scaling coefficient for feat match loss.
            lambda_mel (float): Loss scaling coefficient for mel loss.
            cache_generator_outputs (bool): Whether to cache generator outputs.

        """
        assert check_argument_types()
        super().__init__()
        self.segment_size = segment_size
        self.use_pqmf = use_pqmf

        # define modules
        self.generator = torch.nn.ModuleDict()
        text2mel_class = AVAILABLE_TEXT2MEL[text2mel_type]
        text2mel_params.update(idim=idim, odim=odim)
        self.generator["text2mel"] = text2mel_class(**text2mel_params,)
        vocoder_class = AVAILABLE_VOCODER[vocoder_type]
        if vocoder_type in ["hifigan_generator", "melgan_generator"]:
            vocoder_params.update(in_channels=odim)
        elif vocoder_type in ["parallel_wavegan_generator", "style_melgan_generator"]:
            vocoder_params.update(aux_channels=odim)
        self.generator["vocoder"] = vocoder_class(**vocoder_params,)
        if self.use_pqmf:
            self.pqmf = PQMF(**pqmf_params)
        discriminator_class = AVAILABLE_DISCRIMINATORS[discriminator_type]
        self.discriminator = discriminator_class(**discriminator_params,)
        self.generator_adv_loss = GeneratorAdversarialLoss(**generator_adv_loss_params,)
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss(
            **discriminator_adv_loss_params,
        )
        self.use_feat_match_loss = use_feat_match_loss
        if self.use_feat_match_loss:
            self.feat_match_loss = FeatureMatchLoss(**feat_match_loss_params,)
        self.use_mel_loss = use_mel_loss
        if self.use_mel_loss:
            self.mel_loss = MelSpectrogramLoss(**mel_loss_params,)

        # coefficients
        self.lambda_text2mel = lambda_text2mel
        self.lambda_adv = lambda_adv
        if self.use_feat_match_loss:
            self.lambda_feat_match = lambda_feat_match
        if self.use_mel_loss:
            self.lambda_mel = lambda_mel

        # cache
        self.cache_generator_outputs = cache_generator_outputs
        self._cache = None

        # store sampling rate for saving wav file
        # (not used for the training)
        self.fs = sampling_rate

        # store parameters for test compatibility
        self.spks = self.generator["text2mel"].spks
        self.langs = self.generator["text2mel"].langs
        self.spk_embed_dim = self.generator["text2mel"].spk_embed_dim

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
            # calculate text2mel outputs
            text2mel_loss, stats, feats_gen = self.generator["text2mel"](
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                joint_training=True,
                **kwargs,
            )
            # get random segments
            feats_gen_, start_idxs = get_random_segments(
                x=feats_gen.transpose(1, 2),
                x_lengths=feats_lengths,
                segment_size=self.segment_size,
            )
            # calculate vocoder outputs
            speech_hat_ = self.generator["vocoder"](feats_gen_)
            if self.use_pqmf:
                speech_hat_ = self.pqmf.synthesis(speech_hat_)
        else:
            text2mel_loss, stats, speech_hat_, start_idxs = self._cache

        # store cache
        if self.training and self.cache_generator_outputs and not reuse_cache:
            self._cache = (text2mel_loss, stats, speech_hat_, start_idxs)

        speech_ = get_segments(
            x=speech,
            start_idxs=start_idxs * self.generator["vocoder"].upsample_factor,
            segment_size=self.segment_size * self.generator["vocoder"].upsample_factor,
        )

        # calculate discriminator outputs
        p_hat = self.discriminator(speech_hat_)
        with torch.no_grad():
            # do not store discriminator gradient in generator turn
            p = self.discriminator(speech_)

        # calculate losses
        adv_loss = self.generator_adv_loss(p_hat)
        adv_loss = adv_loss * self.lambda_adv
        text2mel_loss = text2mel_loss * self.lambda_text2mel
        loss = adv_loss + text2mel_loss
        if self.use_feat_match_loss:
            feat_match_loss = self.feat_match_loss(p_hat, p)
            feat_match_loss = feat_match_loss * self.lambda_feat_match
            loss = loss + feat_match_loss
            stats.update(feat_match_loss=feat_match_loss.item())
        if self.use_mel_loss:
            mel_loss = self.mel_loss(speech_hat_, speech_)
            mel_loss = self.lambda_mel * mel_loss
            loss = loss + mel_loss
            stats.update(mel_loss=mel_loss.item())

        stats.update(
            adv_loss=adv_loss.item(),
            text2mel_loss=text2mel_loss.item(),
            loss=loss.item(),
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
            # calculate text2mel outputs
            text2mel_loss, stats, feats_gen = self.generator["text2mel"](
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                joint_training=True,
                **kwargs,
            )
            # get random segments
            feats_gen_, start_idxs = get_random_segments(
                x=feats_gen.transpose(1, 2),
                x_lengths=feats_lengths,
                segment_size=self.segment_size,
            )
            # calculate vocoder outputs
            speech_hat_ = self.generator["vocoder"](feats_gen_)
            if self.use_pqmf:
                speech_hat_ = self.pqmf.synthesis(speech_hat_)
        else:
            _, _, speech_hat_, start_idxs = self._cache

        # store cache
        if self.cache_generator_outputs and not reuse_cache:
            self._cache = (text2mel_loss, stats, speech_hat_, start_idxs)

        # parse outputs
        speech_ = get_segments(
            x=speech,
            start_idxs=start_idxs * self.generator["vocoder"].upsample_factor,
            segment_size=self.segment_size * self.generator["vocoder"].upsample_factor,
        )

        # calculate discriminator outputs
        p_hat = self.discriminator(speech_hat_.detach())
        p = self.discriminator(speech_)

        # calculate losses
        real_loss, fake_loss = self.discriminator_adv_loss(p_hat, p)
        loss = real_loss + fake_loss

        stats = dict(
            discriminator_loss=loss.item(),
            real_loss=real_loss.item(),
            fake_loss=fake_loss.item(),
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

    def inference(self, text: torch.Tensor, **kwargs,) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            text (Tensor): Input text index tensor (T_text,).

        Returns:
            Dict[str, Tensor]:
                * wav (Tensor): Generated waveform tensor (T_wav,).
                * feat_gan (Tensor): Generated feature tensor (T_text, C).

        """
        output_dict = self.generator["text2mel"].inference(text=text, **kwargs,)
        wav = self.generator["vocoder"].inference(output_dict["feat_gen"])
        if self.use_pqmf:
            wav = self.pqmf.synthesis(wav.unsqueeze(0).transpose(1, 2))
            wav = wav.squeeze(0).transpose(0, 1)
        output_dict.update(wav=wav)

        return output_dict
