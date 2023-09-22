# Copyright 2021 Tomoki Hayashi
# Copyright 2023 Renmin University of China (Yuning Wu)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Joint score-to-wav module for end-to-end training."""

from typing import Any, Dict, Optional

import torch
from typeguard import check_argument_types

from espnet2.gan_svs.abs_gan_svs import AbsGANSVS
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
from espnet2.svs.naive_rnn.naive_rnn_dp import NaiveRNNDP
from espnet2.svs.xiaoice.XiaoiceSing import XiaoiceSing
from espnet2.torch_utils.device_funcs import force_gatherable

AVAILABLE_SCORE2MEL = {
    "xiaoice": XiaoiceSing,
    "naive_rnn_dp": NaiveRNNDP,
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


class JointScore2Wav(AbsGANSVS):
    """General class to jointly train score2mel and vocoder parts."""

    def __init__(
        self,
        # generator (score2mel + vocoder) related
        idim: int,
        odim: int,
        segment_size: int = 32,
        sampling_rate: int = 22050,
        score2mel_type: str = "xiaoice",
        score2mel_params: Dict[str, Any] = {
            "midi_dim": 129,
            "tempo_dim": 500,
            "adim": 384,
            "aheads": 4,
            "elayers": 6,
            "eunits": 1536,
            "dlayers": 6,
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
            "duration_predictor_layers": 2,
            "duration_predictor_chans": 384,
            "duration_predictor_kernel_size": 3,
            "duration_predictor_dropout_rate": 0.1,
            "reduction_factor": 1,
            "encoder_type": "transformer",
            "decoder_type": "transformer",
            "transformer_enc_dropout_rate": 0.1,
            "transformer_enc_positional_dropout_rate": 0.1,
            "transformer_enc_attn_dropout_rate": 0.1,
            "transformer_dec_dropout_rate": 0.1,
            "transformer_dec_positional_dropout_rate": 0.1,
            "transformer_dec_attn_dropout_rate": 0.1,
            # only for conformer
            "conformer_rel_pos_type": "latest",
            "conformer_pos_enc_layer_type": "rel_pos",
            "conformer_self_attn_layer_type": "rel_selfattn",
            "conformer_activation_type": "swish",
            "use_macaron_style_in_conformer": True,
            "use_cnn_in_conformer": True,
            "zero_triu": False,
            "conformer_enc_kernel_size": 7,
            "conformer_dec_kernel_size": 31,
            # extra embedding related
            "spks": None,
            "langs": None,
            "spk_embed_dim": None,
            "spk_embed_integration_type": "add",
            # training related
            "init_type": "xavier_uniform",
            "init_enc_alpha": 1.0,
            "init_dec_alpha": 1.0,
            "use_masking": False,
            "use_weighted_masking": False,
            "loss_function": "XiaoiceSing2",
            "loss_type": "L1",
            "lambda_mel": 1,
            "lambda_dur": 0.1,
            "lambda_pitch": 0.01,
            "lambda_vuv": 0.01,
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
        lambda_score2mel: float = 1.0,
        lambda_adv: float = 1.0,
        lambda_feat_match: float = 2.0,
        lambda_mel: float = 45.0,
        cache_generator_outputs: bool = False,
    ):
        """Initialize JointScore2Wav module.

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
        score2mel_class = AVAILABLE_SCORE2MEL[score2mel_type]
        score2mel_params.update(idim=idim, odim=odim)
        self.generator["score2mel"] = score2mel_class(
            **score2mel_params,
        )
        vocoder_class = AVAILABLE_VOCODER[vocoder_type]
        if vocoder_type in ["hifigan_generator", "melgan_generator"]:
            vocoder_params.update(in_channels=odim)
        elif vocoder_type in ["parallel_wavegan_generator", "style_melgan_generator"]:
            vocoder_params.update(aux_channels=odim)
        self.generator["vocoder"] = vocoder_class(
            **vocoder_params,
        )
        if self.use_pqmf:
            self.pqmf = PQMF(**pqmf_params)
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
        self.lambda_score2mel = lambda_score2mel
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
        self.spks = self.generator["score2mel"].spks
        self.langs = self.generator["score2mel"].langs
        self.spk_embed_dim = self.generator["score2mel"].spk_embed_dim

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
        pitch: torch.LongTensor = None,
        duration: Optional[Dict[str, torch.Tensor]] = None,
        slur: torch.LongTensor = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        forward_generator: bool = True,
    ) -> Dict[str, Any]:
        """Perform generator forward.

        Args:
            text (LongTensor): Batch of padded character ids (B, Tmax).
            text_lengths (LongTensor): Batch of lengths of each input batch (B,).
            feats (Tensor): Batch of padded target features (B, Lmax, odim).
            feats_lengths (LongTensor): Batch of the lengths of each target (B,).
            singing (Tensor): Singing waveform tensor (B, T_wav).
            singing_lengths (Tensor): Singing length tensor (B,).
            label (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded label ids (B, Tmax).
            label_lengths (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of the lengths of padded label ids (B, ).
            melody (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded melody (B, Tmax).
            pitch (FloatTensor): Batch of padded f0 (B, Tmax).
            duration (Optional[Dict]): key is "lab", "score_phn" or "score_syb";
                value (LongTensor): Batch of padded duration (B, Tmax).
            slur (FloatTensor): Batch of padded slur (B, Tmax).
            spembs (Optional[Tensor]): Batch of speaker embeddings (B, spk_embed_dim).
            sids (Optional[Tensor]): Batch of speaker IDs (B, 1).
            lids (Optional[Tensor]): Batch of language IDs (B, 1).
            forward_generator (bool): Whether to forward generator.

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """

        label = label["score"]
        label_lengths = label_lengths["score"]
        melody = melody["score"]
        duration = duration["lab"]

        if forward_generator:
            return self._forward_generator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                singing=singing,
                singing_lengths=singing_lengths,
                label=label,
                label_lengths=label_lengths,
                melody=melody,
                duration=duration,
                slur=slur,
                pitch=pitch,
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
                label=label,
                label_lengths=label_lengths,
                melody=melody,
                duration=duration,
                slur=slur,
                pitch=pitch,
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
        label: Optional[Dict[str, torch.Tensor]] = None,
        label_lengths: Optional[Dict[str, torch.Tensor]] = None,
        melody: Optional[Dict[str, torch.Tensor]] = None,
        duration: Optional[Dict[str, torch.Tensor]] = None,
        slur: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
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
            label (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded label ids (B, Tmax).
            label_lengths (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of the lengths of padded label ids (B, ).
            melody (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded melody (B, Tmax).
            duration (Optional[Dict]): key is "lab", "score_phn" or "score_syb";
                value (LongTensor): Batch of padded duration (B, Tmax).
            slur (FloatTensor): Batch of padded slur (B, T_max).
            pitch (FloatTensor): Batch of padded f0 (B, Tmax).
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
        singing = singing.unsqueeze(1)

        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            # calculate text2mel outputs
            score2mel_loss, stats, feats_gen = self.generator["score2mel"](
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                label=label,
                label_lengths=label_lengths,
                melody=melody,
                melody_lengths=label_lengths,
                duration=duration,
                duration_lengths=label_lengths,
                pitch=pitch,
                pitch_lengths=feats_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
                joint_training=True,
            )
            # get random segments
            feats_gen_, start_idxs = get_random_segments(
                x=feats_gen.transpose(1, 2),
                x_lengths=feats_lengths,
                segment_size=self.segment_size,
            )
            # calculate vocoder outputs
            singing_hat_ = self.generator["vocoder"](feats_gen_)
            if self.use_pqmf:
                singing_hat_ = self.pqmf.synthesis(singing_hat_)
        else:
            score2mel_loss, stats, singing_hat_, start_idxs = self._cache

        # store cache
        if self.training and self.cache_generator_outputs and not reuse_cache:
            self._cache = (score2mel_loss, stats, singing_hat_, start_idxs)

        singing_ = get_segments(
            x=singing,
            start_idxs=start_idxs * self.generator["vocoder"].upsample_factor,
            segment_size=self.segment_size * self.generator["vocoder"].upsample_factor,
        )

        # calculate discriminator outputs
        p_hat = self.discriminator(singing_hat_)
        with torch.no_grad():
            # do not store discriminator gradient in generator turn
            p = self.discriminator(singing_)

        # calculate losses
        adv_loss = self.generator_adv_loss(p_hat)
        adv_loss = adv_loss * self.lambda_adv
        score2mel_loss = score2mel_loss * self.lambda_score2mel
        loss = adv_loss + score2mel_loss
        if self.use_feat_match_loss:
            feat_match_loss = self.feat_match_loss(p_hat, p)
            feat_match_loss = feat_match_loss * self.lambda_feat_match
            loss = loss + feat_match_loss
            stats.update(feat_match_loss=feat_match_loss.item())
        if self.use_mel_loss:
            mel_loss = self.mel_loss(singing_hat_, singing_)
            mel_loss = self.lambda_mel * mel_loss
            loss = loss + mel_loss
            stats.update(mel_loss=mel_loss.item())

        stats.update(
            adv_loss=adv_loss.item(),
            score2mel_loss=score2mel_loss.item(),
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
        singing: torch.Tensor,
        singing_lengths: torch.Tensor,
        label: Optional[Dict[str, torch.Tensor]] = None,
        label_lengths: Optional[Dict[str, torch.Tensor]] = None,
        melody: Optional[Dict[str, torch.Tensor]] = None,
        duration: Optional[Dict[str, torch.Tensor]] = None,
        slur: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
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
            label (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded label ids (B, Tmax).
            label_lengths (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of the lengths of padded label ids (B, ).
            melody (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded melody (B, Tmax).
            duration (Optional[Dict]): key is "lab", "score_phn" or "score_syb";
                value (LongTensor): Batch of padded duration (B, Tmax).
            slur (FloatTensor): Batch of padded slur (B, T_max).
            pitch (FloatTensor): Batch of padded f0 (B, Tmax).
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
        singing = singing.unsqueeze(1)

        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            # calculate score2mel outputs
            score2mel_loss, stats, feats_gen = self.generator["score2mel"](
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                label=label,
                label_lengths=label_lengths,
                melody=melody,
                melody_lengths=label_lengths,
                duration=duration,
                duration_lengths=label_lengths,
                pitch=pitch,
                pitch_lengths=feats_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
                joint_training=True,
            )
            # get random segments
            feats_gen_, start_idxs = get_random_segments(
                x=feats_gen.transpose(1, 2),
                x_lengths=feats_lengths,
                segment_size=self.segment_size,
            )
            # calculate vocoder outputs
            singing_hat_ = self.generator["vocoder"](feats_gen_)
            if self.use_pqmf:
                singing_hat_ = self.pqmf.synthesis(singing_hat_)
        else:
            _, _, singing_hat_, start_idxs = self._cache

        # store cache
        if self.cache_generator_outputs and not reuse_cache:
            self._cache = (score2mel_loss, stats, singing_hat_, start_idxs)

        # parse outputs
        singing_ = get_segments(
            x=singing,
            start_idxs=start_idxs * self.generator["vocoder"].upsample_factor,
            segment_size=self.segment_size * self.generator["vocoder"].upsample_factor,
        )

        # calculate discriminator outputs
        p_hat = self.discriminator(singing_hat_.detach())
        p = self.discriminator(singing_)

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

    def inference(
        self,
        text: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        label: Optional[Dict[str, torch.Tensor]] = None,
        melody: Optional[Dict[str, torch.Tensor]] = None,
        pitch: Optional[torch.Tensor] = None,
        duration: Optional[Dict[str, torch.Tensor]] = None,
        slur: Optional[Dict[str, torch.Tensor]] = None,
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
            label (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded label ids (B, Tmax).
            melody (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded melody (B, Tmax).
            duration (Optional[Dict]): key is "lab", "score_phn" or "score_syb";
                value (LongTensor): Batch of padded duration (B, Tmax).
            pitch (FloatTensor): Batch of padded f0 (B, Tmax).
            slur (LongTensor): Batch of padded slur (B, Tmax).
            sids (Tensor): Speaker index tensor (1,).
            spembs (Optional[Tensor]): Speaker embedding tensor (spk_embed_dim,).
            lids (Tensor): Language index tensor (1,).
            noise_scale (float): Noise scale value for flow.
            noise_scale_dur (float): Noise scale value for duration predictor.
            alpha (float): Alpha parameter to control the speed of generated singing.
            max_len (Optional[int]): Maximum length.
            use_teacher_forcing (bool): Whether to use teacher forcing.

        Returns:
            Dict[str, Tensor]:
                * wav (Tensor): Generated waveform tensor (T_wav,).
                * feat_gan (Tensor): Generated feature tensor (T_text, C).

        """
        output_dict = self.generator["score2mel"].inference(
            text=text,
            feats=feats,
            label=label,
            melody=melody,
            duration=duration,
            pitch=pitch,
            sids=sids,
            spembs=spembs,
            lids=lids,
            joint_training=True,
        )
        wav = self.generator["vocoder"].inference(output_dict["feat_gen"])
        if self.use_pqmf:
            wav = self.pqmf.synthesis(wav.unsqueeze(0).transpose(1, 2))
            wav = wav.squeeze(0).transpose(0, 1)
        output_dict.update(wav=wav)

        return output_dict
