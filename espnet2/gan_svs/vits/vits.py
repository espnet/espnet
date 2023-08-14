# Copyright 2021 Tomoki Hayashi
# Copyright 2022 Yifeng Yu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""VITS/VISinger module for GAN-SVS task."""

from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Any, Dict, Optional

import torch
from torch.nn import functional as F
from typeguard import check_argument_types

from espnet2.gan_svs.abs_gan_svs import AbsGANSVS
from espnet2.gan_svs.avocodo.avocodo import (
    SBD,
    AvocodoDiscriminator,
    AvocodoDiscriminatorPlus,
    CoMBD,
)
from espnet2.gan_svs.visinger2.visinger2_vocoder import VISinger2Discriminator
from espnet2.gan_svs.vits.generator import VISingerGenerator
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
from espnet2.gan_tts.vits.loss import KLDivergenceLoss, KLDivergenceLossWithoutFlow
from espnet2.torch_utils.device_funcs import force_gatherable

AVAILABLE_GENERATERS = {
    "visinger": VISingerGenerator,
    # TODO(yifeng): add more generators
    "visinger2": VISingerGenerator,
    # "pisinger": PISingerGenerator,
}
AVAILABLE_DISCRIMINATORS = {
    "hifigan_period_discriminator": HiFiGANPeriodDiscriminator,
    "hifigan_scale_discriminator": HiFiGANScaleDiscriminator,
    "hifigan_multi_period_discriminator": HiFiGANMultiPeriodDiscriminator,
    "hifigan_multi_scale_discriminator": HiFiGANMultiScaleDiscriminator,
    "hifigan_multi_scale_multi_period_discriminator": HiFiGANMultiScaleMultiPeriodDiscriminator,  # NOQA
    "combd": CoMBD,
    "sbd": SBD,
    "avocodo": AvocodoDiscriminator,
    "visinger2": VISinger2Discriminator,
    "avocodo_plus": AvocodoDiscriminatorPlus,
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
        generator_type: str = "visinger",
        vocoder_generator_type: str = "hifigan",
        generator_params: Dict[str, Any] = {
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
            "projection_filters": [0, 1, 1, 1],
            "projection_kernels": [0, 5, 7, 11],
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
            "expand_f0_method": "repeat",
            "use_phoneme_predictor": False,
        },
        # discriminator related
        discriminator_type: str = "hifigan_multi_scale_multi_period_discriminator",
        discriminator_params: Dict[str, Any] = {
            "hifigan_multi_scale_multi_period_discriminator": {
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
            "avocodo": {
                "combd": {
                    "combd_h_u": [
                        [16, 64, 256, 1024, 1024, 1024],
                        [16, 64, 256, 1024, 1024, 1024],
                        [16, 64, 256, 1024, 1024, 1024],
                    ],
                    "combd_d_k": [
                        [7, 11, 11, 11, 11, 5],
                        [11, 21, 21, 21, 21, 5],
                        [15, 41, 41, 41, 41, 5],
                    ],
                    "combd_d_s": [
                        [1, 1, 4, 4, 4, 1],
                        [1, 1, 4, 4, 4, 1],
                        [1, 1, 4, 4, 4, 1],
                    ],
                    "combd_d_d": [
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                    ],
                    "combd_d_g": [
                        [1, 4, 16, 64, 256, 1],
                        [1, 4, 16, 64, 256, 1],
                        [1, 4, 16, 64, 256, 1],
                    ],
                    "combd_d_p": [
                        [3, 5, 5, 5, 5, 2],
                        [5, 10, 10, 10, 10, 2],
                        [7, 20, 20, 20, 20, 2],
                    ],
                    "combd_op_f": [1, 1, 1],
                    "combd_op_k": [3, 3, 3],
                    "combd_op_g": [1, 1, 1],
                },
                "sbd": {
                    "use_sbd": True,
                    "sbd_filters": [
                        [64, 128, 256, 256, 256],
                        [64, 128, 256, 256, 256],
                        [64, 128, 256, 256, 256],
                        [32, 64, 128, 128, 128],
                    ],
                    "sbd_strides": [
                        [1, 1, 3, 3, 1],
                        [1, 1, 3, 3, 1],
                        [1, 1, 3, 3, 1],
                        [1, 1, 3, 3, 1],
                    ],
                    "sbd_kernel_sizes": [
                        [[7, 7, 7], [7, 7, 7], [7, 7, 7], [7, 7, 7], [7, 7, 7]],
                        [[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]],
                        [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                        [[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]],
                    ],
                    "sbd_dilations": [
                        [[5, 7, 11], [5, 7, 11], [5, 7, 11], [5, 7, 11], [5, 7, 11]],
                        [[3, 5, 7], [3, 5, 7], [3, 5, 7], [3, 5, 7], [3, 5, 7]],
                        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
                        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [2, 3, 5], [2, 3, 5]],
                    ],
                    "sbd_band_ranges": [[0, 6], [0, 11], [0, 16], [0, 64]],
                    "sbd_transpose": [False, False, False, True],
                    "pqmf_config": {
                        "sbd": [16, 256, 0.03, 10.0],
                        "fsbd": [64, 256, 0.1, 9.0],
                    },
                },
                "pqmf_config": {
                    "lv1": [2, 256, 0.25, 10.0],
                    "lv2": [4, 192, 0.13, 10.0],
                },
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
        lambda_dur: float = 0.1,
        lambda_kl: float = 1.0,
        lambda_pitch: float = 10.0,
        lambda_phoneme: float = 1.0,
        lambda_c_yin: float = 45.0,
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
            vocoder_generator_type (str): Type of vocoder generator to use in the model.
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
            lambda_pitch (float): Loss scaling coefficient for pitch loss.
            lambda_phoneme (float): Loss scaling coefficient for phoneme loss.
            lambda_c_yin (float): Loss scaling coefficient for yin loss.
            cache_generator_outputs (bool): Whether to cache generator outputs.

        """
        assert check_argument_types()
        super().__init__()

        # define modules
        generator_class = AVAILABLE_GENERATERS[generator_type]
        if "visinger" in generator_type or "pisinger" in generator_type:
            # NOTE(kan-bayashi): Update parameters for the compatibility.
            #   The idim and odim is automatically decided from input data,
            #   where idim represents #vocabularies and odim represents
            #   the input acoustic feature dimension.
            generator_params.update(vocabs=idim, aux_channels=odim)
        self.generator_type = generator_type
        self.use_flow = True if generator_params["flow_flows"] > 0 else False
        self.use_phoneme_predictor = generator_params["use_phoneme_predictor"]
        self.discriminator_type = discriminator_type
        if "avocodo" in discriminator_type:
            use_avocodo = True
            vocoder_generator_type = "avocodo"
        else:
            use_avocodo = False
        self.use_avocodo = use_avocodo
        self.vocoder_generator_type = vocoder_generator_type
        generator_params.update(generator_type=generator_type)
        generator_params.update(vocoder_generator_type=vocoder_generator_type)
        generator_params.update(fs=mel_loss_params["fs"])
        generator_params.update(hop_length=mel_loss_params["hop_length"])
        generator_params.update(win_length=mel_loss_params["win_length"])
        generator_params.update(n_fft=mel_loss_params["n_fft"])
        if vocoder_generator_type == "uhifigan" and use_avocodo:
            generator_params.update(use_avocodo=use_avocodo)
        self.generator = generator_class(
            **generator_params,
        )

        discriminator_class = AVAILABLE_DISCRIMINATORS[self.discriminator_type]
        if use_avocodo:
            discriminator_params.update(
                projection_filters=generator_params["projection_filters"]
            )
            discriminator_params["sbd"].update(
                segment_size=generator_params["segment_size"]
                * mel_loss_params["hop_length"]
            )
        if "visinger2" in discriminator_type:
            discriminator_params["multi_freq_disc_params"].update(
                sample_rate=sampling_rate
            )

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
        if self.use_flow:
            self.kl_loss = KLDivergenceLoss()
        else:
            self.kl_loss = KLDivergenceLossWithoutFlow()

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
        self.lambda_c_yin = lambda_c_yin

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
        pitch: torch.LongTensor = None,
        ying: torch.Tensor = None,
        duration: Optional[Dict[str, torch.Tensor]] = None,
        slur: torch.LongTensor = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        forward_generator: bool = True,
    ) -> Dict[str, Any]:
        """Perform generator forward.

        Args:
            text (LongTensor): Batch of padded character ids (B, T_text).
            text_lengths (LongTensor): Batch of lengths of each input batch (B,).
            feats (Tensor): Batch of padded target features (B, Lmax, odim).
            feats_lengths (LongTensor): Batch of the lengths of each target (B,).
            singing (Tensor): Singing waveform tensor (B, T_wav).
            singing_lengths (Tensor): Singing length tensor (B,).
            label (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded label ids (B, T_text).
            label_lengths (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of the lengths of padded label ids (B, ).
            melody (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded melody (B, T_text).
            pitch (FloatTensor): Batch of padded f0 (B, T_feats).
            duration (Optional[Dict]): key is "lab", "score_phn" or "score_syb";
                value (LongTensor): Batch of padded duration (B, T_text).
            slur (FloatTensor): Batch of padded slur (B, T_text).
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
        score_dur = duration["score_syb"]
        gt_dur = duration["lab"]
        label = label["lab"]
        label_lengths = label_lengths["lab"]
        melody = melody["lab"]

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
                gt_dur=gt_dur,
                score_dur=score_dur,
                slur=slur,
                pitch=pitch,
                ying=ying,
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
                gt_dur=gt_dur,
                score_dur=score_dur,
                slur=slur,
                pitch=pitch,
                ying=ying,
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
        label: torch.Tensor = None,
        label_lengths: torch.Tensor = None,
        melody: torch.Tensor = None,
        gt_dur: torch.Tensor = None,
        score_dur: torch.Tensor = None,
        slur: torch.Tensor = None,
        pitch: torch.Tensor = None,
        ying: Optional[torch.Tensor] = None,
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
            label (Tensor): Label index tensor (B, T_text).
            label_lengths (Tensor): Label length tensor (B,).
            melody (Tensor): Melody index tensor (B, T_text).
            gt_dur (Tensor): Groundtruth duration tensor (B, T_text).
            score_dur (Tensor): Score duration tensor (B, T_text).
            slur (Tensor): Slur index tensor (B, T_text).
            pitch (FloatTensor): Batch of padded f0 (B, T_feats).
            ying (Optional[Tensor]): Yin pitch tensor (B, T_feats).
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
                label=label,
                label_lengths=label_lengths,
                melody=melody,
                gt_dur=gt_dur,
                score_dur=score_dur,
                slur=slur,
                pitch=pitch,
                ying=ying,
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
        if "visinger" in self.generator_type:
            singing_hat_, start_idxs, _, z_mask, outs_, *extra_outs = outs

            if (
                self.vocoder_generator_type == "visinger2"
                and self.generator_type == "visinger2"
            ):
                singing_hat_ddsp_, predict_mel = extra_outs
            elif self.vocoder_generator_type == "visinger2":
                singing_hat_ddsp_ = extra_outs[0]
            elif self.generator_type == "visinger2":
                predict_mel = extra_outs[0]
        elif "pisinger" in self.generator_type:
            if self.vocoder_generator_type == "visinger2":
                (
                    singing_hat_,
                    start_idxs,
                    _,
                    z_mask,
                    outs_,
                    singing_hat_ddsp_,
                    outs2_,
                ) = outs
            else:
                singing_hat_, start_idxs, _, z_mask, outs_, outs2_ = outs
            (
                yin_gt_crop,
                yin_gt_shifted_crop,
                yin_dec_crop,
                z_yin_crop_shifted,
                scope_shift,
            ) = outs2_

        (
            _,
            z_p,
            m_p,
            logs_p,
            m_q,
            logs_q,
            pred_pitch,
            gt_pitch,
            pred_dur,
            gt_dur,
            log_probs,
        ) = outs_

        singing_ = get_segments(
            x=singing,
            start_idxs=start_idxs * self.generator.upsample_factor,
            segment_size=self.generator.segment_size * self.generator.upsample_factor,
        )

        # calculate discriminator outputs
        if "avocodo" in self.discriminator_type:
            p, p_hat, fmaps_real, fmaps_fake = self.discriminator(
                singing_, singing_hat_
            )
        else:
            p_hat = self.discriminator(singing_hat_)
            with torch.no_grad():
                # do not store discriminator gradient in generator turn
                p = self.discriminator(singing_)

        # calculate losses
        with autocast(enabled=False):
            if "pisinger" in self.generator_type:
                yin_dec_loss = (
                    F.l1_loss(yin_gt_shifted_crop, yin_dec_crop) * self.lambda_c_yin
                )
                # TODO(yifeng): add yin shift loss later
                # loss_yin_shift = (
                #     F.l1_loss(torch.exp(-yin_gt_crop), torch.exp(-yin_hat_crop))
                #     * self.lambda_c_yin
                #     + F.l1_loss(
                #         torch.exp(-yin_hat_shifted),
                #         torch.exp(-(torch.chunk(yin_hat_crop, 2, dim=0)[1])),
                #     )
                #     * self.lambda_c_yin
                # )
            if self.use_avocodo:
                mel_loss = self.mel_loss(singing_hat_[-1], singing_)
            elif self.vocoder_generator_type == "visinger2":
                mel_loss = self.mel_loss(singing_hat_, singing_)
                ddsp_mel_loss = self.mel_loss(singing_hat_ddsp_, singing_)
            else:
                mel_loss = self.mel_loss(singing_hat_, singing_)
            if self.use_flow:
                kl_loss = self.kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
            else:
                kl_loss = self.kl_loss(m_q, logs_q, m_p, logs_p)

            if "avocodo" in self.discriminator_type:
                adv_loss = self.generator_adv_loss(p_hat)
                feat_match_loss = self.feat_match_loss(fmaps_fake, fmaps_real)
            else:
                adv_loss = self.generator_adv_loss(p_hat)
                feat_match_loss = self.feat_match_loss(p_hat, p)

            pitch_loss = self.mse_loss(pred_pitch, gt_pitch)

            phoneme_dur_loss = self.mse_loss(
                pred_dur[:, 0, :].squeeze(1), gt_dur.float()
            )
            score_dur_loss = self.mse_loss(pred_dur[:, 1, :].squeeze(1), gt_dur.float())

            if self.use_phoneme_predictor:
                ctc_loss = self.ctc_loss(log_probs, label, feats_lengths, label_lengths)

            mel_loss = mel_loss * self.lambda_mel
            kl_loss = kl_loss * self.lambda_kl

            adv_loss = adv_loss * self.lambda_adv
            feat_match_loss = feat_match_loss * self.lambda_feat_match

            pitch_loss = pitch_loss * self.lambda_pitch
            phoneme_dur_loss = phoneme_dur_loss * self.lambda_dur
            score_dur_loss = score_dur_loss * self.lambda_dur
            if self.use_phoneme_predictor:
                ctc_loss = ctc_loss * self.lambda_phoneme

            loss = mel_loss + kl_loss + adv_loss + feat_match_loss
            if self.vocoder_generator_type == "visinger2":
                ddsp_mel_loss = ddsp_mel_loss * self.lambda_mel
                loss = loss + ddsp_mel_loss
            if self.generator_type == "visinger2":
                loss_mel_am = self.mse_loss(feats * z_mask, predict_mel * z_mask)
                loss = loss + loss_mel_am

            loss = loss + pitch_loss
            loss = loss + phoneme_dur_loss
            loss = loss + score_dur_loss
            if self.use_phoneme_predictor:
                loss = loss + ctc_loss
            if "pisinger" in self.generator_type:
                loss = loss + yin_dec_loss

        stats = dict(
            generator_loss=loss.item(),
            generator_mel_loss=mel_loss.item(),
            generator_phn_dur_loss=phoneme_dur_loss.item(),
            generator_score_dur_loss=score_dur_loss.item(),
            generator_adv_loss=adv_loss.item(),
            generator_feat_match_loss=feat_match_loss.item(),
            generator_pitch_loss=pitch_loss.item(),
            generator_kl_loss=kl_loss.item(),
        )

        if self.use_phoneme_predictor:
            stats.update(
                dict(
                    generator_phoneme_loss=ctc_loss.item(),
                )
            )

        if self.vocoder_generator_type == "visinger2":
            stats.update(
                dict(
                    generator_mel_ddsp_loss=ddsp_mel_loss.item(),
                )
            )
        if self.generator_type == "visinger2":
            stats.update(
                dict(
                    generator_mel_am_loss=loss_mel_am.item(),
                )
            )
        if "pisinger" in self.generator_type:
            stats.update(
                dict(
                    generator_yin_dec_loss=yin_dec_loss.item(),
                )
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
        label: torch.Tensor = None,
        label_lengths: torch.Tensor = None,
        melody: torch.Tensor = None,
        gt_dur: torch.Tensor = None,
        score_dur: torch.Tensor = None,
        slur: torch.Tensor = None,
        pitch: torch.Tensor = None,
        ying: Optional[torch.Tensor] = None,
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
            label (Tensor): Label index tensor (B, T_text).
            label_lengths (Tensor): Label length tensor (B,).
            melody (Tensor): Melody index tensor (B, T_text).
            gt_dur (Tensor): Groundtruth duration tensor (B, T_text).
            score_dur (Tensor): Score duration tensor (B, T_text).
            slur (Tensor): Slur index tensor (B, T_text).
            pitch (FloatTensor): Batch of padded f0 (B, T_feats).
            ying (Optional[Tensor]): Yin pitch tensor (B, T_feats).
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
                gt_dur=gt_dur,
                label=label,
                label_lengths=label_lengths,
                melody=melody,
                score_dur=score_dur,
                slur=slur,
                pitch=pitch,
                ying=ying,
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
        if "avocodo" in self.discriminator_type:
            detached_singing_hat_ = [x.detach() for x in singing_hat_]
            p, p_hat, fmaps_real, fmaps_fake = self.discriminator(
                singing_, detached_singing_hat_
            )
        else:
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
                value (LongTensor): Batch of padded label ids (B, T_text).
            melody (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded melody (B, T_text).
            pitch (FloatTensor): Batch of padded f0 (B, T_feats).
            slur (LongTensor): Batch of padded slur (B, T_text).
            sids (Tensor): Speaker index tensor (1,).
            spembs (Optional[Tensor]): Speaker embedding tensor (spk_embed_dim,).
            lids (Tensor): Language index tensor (1,).
            noise_scale (float): Noise scale value for flow.
            noise_scale_dur (float): Noise scale value for duration predictor.
            alpha (float): Alpha parameter to control the speed of generated singing.
            max_len (Optional[int]): Maximum length.
            use_teacher_forcing (bool): Whether to use teacher forcing.
            duration (Optional[Dict]): key is "lab", "score_phn" or "score_syb";
                value (LongTensor): Batch of padded duration (B, T_text).

        Returns:
            Dict[str, Tensor]:
                * wav (Tensor): Generated waveform tensor (T_wav,).

        """
        # setup
        label = label["lab"]
        melody = melody["lab"]
        score_dur = duration["score_syb"]
        gt_dur = duration["lab"]
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

        # inference
        if use_teacher_forcing:
            assert feats is not None
            assert pitch is not None
            feats = feats[None].transpose(1, 2)
            feats_lengths = torch.tensor(
                [feats.size(2)],
                dtype=torch.long,
                device=feats.device,
            )
            wav = self.generator.inference(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                label=label,
                label_lengths=label_lengths,
                melody=melody,
                score_dur=score_dur,
                slur=slur,
                gt_dur=gt_dur,
                pitch=pitch,
                sids=sids,
                spembs=spembs,
                lids=lids,
                noise_scale=noise_scale,
                noise_scale_dur=noise_scale_dur,
                alpha=alpha,
                max_len=max_len,
                use_teacher_forcing=use_teacher_forcing,
            )
        else:
            wav = self.generator.inference(
                text=text,
                text_lengths=text_lengths,
                label=label,
                label_lengths=label_lengths,
                melody=melody,
                score_dur=score_dur,
                slur=slur,
                sids=sids,
                spembs=spembs,
                lids=lids,
                noise_scale=noise_scale,
                noise_scale_dur=noise_scale_dur,
                alpha=alpha,
                max_len=max_len,
            )
        return dict(wav=wav.view(-1))
