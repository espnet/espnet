# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""VITS module for GAN-TTS task."""

from typing import Any
from typing import Dict
from typing import Tuple

import torch

from typeguard import check_argument_types

from espnet2.gan_tts.abs_gan_tts import AbsGANTTS
from espnet2.gan_tts.vits.hifigan import HiFiGANGenerator
from espnet2.gan_tts.vits.hifigan import HiFiGANMultiPeriodDiscriminator
from espnet2.gan_tts.vits.hifigan import HiFiGANMultiScaleDiscriminator
from espnet2.gan_tts.vits.hifigan import HiFiGANMultiScaleMultiPeriodDiscriminator
from espnet2.gan_tts.vits.hifigan import HiFiGANPeriodDiscriminator
from espnet2.gan_tts.vits.hifigan import HiFiGANScaleDiscriminator
from espnet2.gan_tts.vits.loss import DiscriminatorAdversarialLoss
from espnet2.gan_tts.vits.loss import FeatureMatchLoss
from espnet2.gan_tts.vits.loss import GeneratorAdversarialLoss
from espnet2.gan_tts.vits.loss import MelSpectrogramLoss
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.tts.fastspeech import FastSpeech
from espnet2.tts.tacotron2 import Tacotron2
from espnet2.tts.transformer import Transformer

AVAILABLE_TEXT2MEL = {
    "tacotron2": Tacotron2,
    "transformer": Transformer,
    "fastspeech": FastSpeech,
}
AVAILABLE_VOCODER = {
    "hifigan_generator": HiFiGANGenerator,
}
AVAILABLE_DISCRIMINATORS = {
    "hifigan_period_discriminator": HiFiGANPeriodDiscriminator,
    "hifigan_scale_discriminator": HiFiGANScaleDiscriminator,
    "hifigan_multi_period_discriminator": HiFiGANMultiPeriodDiscriminator,
    "hifigan_multi_scale_discriminator": HiFiGANMultiScaleDiscriminator,
    "hifigan_multi_scale_multi_period_discriminator": HiFiGANMultiScaleMultiPeriodDiscriminator,  # NOQA
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
        text2mel_type: str = "tacotron2",
        text2mel_params: Dict[str, Any] = {
            "embed_dim": 512,
            "elayers": 1,
            "eunits": 512,
            "econv_layers": 3,
            "econv_chans": 512,
            "econv_filts": 5,
            "atype": "location",
            "adim": 512,
            "aconv_chans": 32,
            "aconv_filts": 15,
            "cumulate_att_w": True,
            "dlayers": 2,
            "dunits": 1024,
            "prenet_layers": 2,
            "prenet_units": 256,
            "postnet_layers": 5,
            "postnet_chans": 512,
            "postnet_filts": 5,
            "output_activation": None,
            "use_batch_norm": True,
            "use_concate": True,
            "use_residual": False,
            "reduction_factor": 1,
            "spk_embed_dim": None,
            "spk_embed_integration_type": "concat",
            "use_gst": False,
            "gst_tokens": 10,
            "gst_heads": 4,
            "gst_conv_layers": 6,
            "gst_conv_chans_list": [32, 32, 64, 64, 128, 128],
            "gst_conv_kernel_size": 3,
            "gst_conv_stride": 2,
            "gst_gru_layers": 1,
            "gst_gru_units": 128,
            "dropout_rate": 0.5,
            "zoneout_rate": 0.1,
            "use_masking": True,
            "use_weighted_masking": False,
            "bce_pos_weight": 5.0,
            "loss_type": "L1+L2",
            "use_guided_attn_loss": True,
            "guided_attn_loss_sigma": 0.4,
            "guided_attn_loss_lambda": 1.0,
        },
        vocoder_type: str = "hifigan_generator",
        vocoder_params: Dict[str, Any] = {
            "out_channels": 1,
            "channels": 512,
            "global_channels": -1,
            "kernel_size": 7,
            "upsample_scales": [8, 8, 2, 2],
            "upsample_kernal_sizes": [16, 16, 4, 4],
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "use_additional_convs": True,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
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
        cache_generator_outputs: bool = True,
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

        # define modules
        self.generator = torch.nn.ModuleDict()
        text2mel_class = AVAILABLE_TEXT2MEL[text2mel_type]
        text2mel_params.update(idim=idim, odim=odim)
        self.generator["text2mel"] = text2mel_class(
            **text2mel_params,
        )
        vocoder_class = AVAILABLE_VOCODER[vocoder_type]
        if vocoder_type == "hifigan_generator":
            vocoder_params.update(in_channels=odim)
        self.generator["vocoder"] = vocoder_class(
            **vocoder_params,
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
        self.use_feat_match_loss = use_feat_match_loss
        if self.use_feat_match_loss:
            self.feat_match_loss = FeatureMatchLoss(
                **feat_match_loss_params,
            )
        self.use_mel_loss = use_mel_loss
        if self.use_mel_loss:
            self.mel_loss = MelSpectrogramLoss(
                **mel_loss_params,
            )

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
                speech=feats,
                speech_lengths=feats_lengths,
                joint_training=True,
                **kwargs,
            )
            # get random segments
            feats_gen_, start_idxs = self.get_random_segments(
                x=feats_gen.transpose(1, 2),
                x_lengths=feats_lengths,
                segment_size=self.segment_size,
            )
            # calculate vocoder outputs
            speech_hat_ = self.generator["vocoder"](feats_gen_)
        else:
            text2mel_loss, stats, speech_hat_, start_idxs = self._cache

        # store cache
        if self.training and self.cache_generator_outputs and not reuse_cache:
            self._cache = (text2mel_loss, stats, speech_hat_, start_idxs)

        speech_ = self.get_segments(
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
                speech=feats,
                speech_lengths=feats_lengths,
                joint_training=True,
                **kwargs,
            )
            # get random segments
            feats_gen_, start_idxs = self.get_random_segments(
                x=feats_gen.transpose(1, 2),
                x_lengths=feats_lengths,
                segment_size=self.segment_size,
            )
            # calculate vocoder outputs
            speech_hat_ = self.generator["vocoder"](feats_gen_)
        else:
            _, _, speech_hat_, start_idxs = self._cache

        # store cache
        if self.cache_generator_outputs and not reuse_cache:
            self._cache = (text2mel_loss, stats, speech_hat_, start_idxs)

        # parse outputs
        speech_ = self.get_segments(
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

    def get_random_segments(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        segment_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        max_start_idx = x_lengths - segment_size
        start_idxs = (torch.rand([b]).to(x.device) * max_start_idx).to(
            dtype=torch.long,
        )
        segments = self.get_segments(x, start_idxs, segment_size)
        return segments, start_idxs

    def get_segments(
        self,
        x: torch.Tensor,
        start_idxs: torch.Tensor,
        segment_size: int,
    ) -> torch.Tensor:
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
        text: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            text (Tensor): Input text index tensor (T_text,).

        Returns:
            Dict[str, Tensor]:
                * wav (Tensor): Generated waveform tensor (T_wav,).
                * feat_gan (Tensor): Generated feature tensor (T_text, C).

        """
        output_dict = self.generator["text2mel"].inference(
            text=text,
            **kwargs,
        )
        wav = self.generator["vocoder"].inference(output_dict["feat_gen"])
        output_dict.update(wav=wav)

        return output_dict
