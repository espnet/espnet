# Copyright 2021 Tomoki Hayashi
# Copyright 2022 Yifeng Yu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Generator module in VISinger.

This code is based on https://github.com/jaywalnut310/vits.

    This is a module of VISinger described in `VISinger: Variational Inference
      with Adversarial Learning for End-to-End Singing Voice Synthesis`_.

    .. _`VISinger: Variational Inference with Adversarial Learning for
      End-to-End Singing Voice Synthesis`: https://arxiv.org/abs/2110.08813

"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.gan_svs.avocodo import AvocodoGenerator
from espnet2.gan_svs.uhifigan import UHiFiGANGenerator
from espnet2.gan_svs.uhifigan.sine_generator import SineGen
from espnet2.gan_svs.utils.expand_f0 import expand_f0
from espnet2.gan_svs.visinger2 import (
    Generator_Harm,
    Generator_Noise,
    VISinger2VocoderGenerator,
)
from espnet2.gan_svs.visinger2.ddsp import upsample
from espnet2.gan_svs.vits.duration_predictor import DurationPredictor
from espnet2.gan_svs.vits.length_regulator import LengthRegulator
from espnet2.gan_svs.vits.phoneme_predictor import PhonemePredictor
from espnet2.gan_svs.vits.pitch_predictor import Decoder
from espnet2.gan_svs.vits.prior_decoder import PriorDecoder
from espnet2.gan_svs.vits.text_encoder import TextEncoder
from espnet2.gan_tts.hifigan import HiFiGANGenerator
from espnet2.gan_tts.utils import get_random_segments, get_segments
from espnet2.gan_tts.vits.posterior_encoder import PosteriorEncoder
from espnet2.gan_tts.vits.residual_coupling import ResidualAffineCouplingBlock


class VISingerGenerator(torch.nn.Module):
    """Generator module in VISinger."""

    @typechecked
    def __init__(
        self,
        vocabs: int,
        aux_channels: int = 513,
        hidden_channels: int = 192,
        spks: Optional[int] = None,
        langs: Optional[int] = None,
        spk_embed_dim: Optional[int] = None,
        global_channels: int = -1,
        segment_size: int = 32,
        text_encoder_attention_heads: int = 2,
        text_encoder_ffn_expand: int = 4,
        text_encoder_blocks: int = 6,
        text_encoder_positionwise_layer_type: str = "conv1d",
        text_encoder_positionwise_conv_kernel_size: int = 1,
        text_encoder_positional_encoding_layer_type: str = "rel_pos",
        text_encoder_self_attention_layer_type: str = "rel_selfattn",
        text_encoder_activation_type: str = "swish",
        text_encoder_normalize_before: bool = True,
        text_encoder_dropout_rate: float = 0.1,
        text_encoder_positional_dropout_rate: float = 0.0,
        text_encoder_attention_dropout_rate: float = 0.0,
        text_encoder_conformer_kernel_size: int = 7,
        use_macaron_style_in_text_encoder: bool = True,
        use_conformer_conv_in_text_encoder: bool = True,
        decoder_kernel_size: int = 7,
        decoder_channels: int = 512,
        decoder_downsample_scales: List[int] = [2, 2, 8, 8],
        decoder_downsample_kernel_sizes: List[int] = [4, 4, 16, 16],
        decoder_upsample_scales: List[int] = [8, 8, 2, 2],
        decoder_upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        decoder_resblock_kernel_sizes: List[int] = [3, 7, 11],
        decoder_resblock_dilations: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        # avocodo
        use_avocodo=False,
        projection_filters: List[int] = [0, 1, 1, 1],
        projection_kernels: List[int] = [0, 5, 7, 11],
        # visinger 2
        n_harmonic: int = 64,
        use_weight_norm_in_decoder: bool = True,
        posterior_encoder_kernel_size: int = 5,
        posterior_encoder_layers: int = 16,
        posterior_encoder_stacks: int = 1,
        posterior_encoder_base_dilation: int = 1,
        posterior_encoder_dropout_rate: float = 0.0,
        use_weight_norm_in_posterior_encoder: bool = True,
        flow_flows: int = 4,
        flow_kernel_size: int = 5,
        flow_base_dilation: int = 1,
        flow_layers: int = 4,
        flow_dropout_rate: float = 0.0,
        use_weight_norm_in_flow: bool = True,
        use_only_mean_in_flow: bool = True,
        generator_type: str = "visinger",
        vocoder_generator_type: str = "hifigan",
        fs: int = 22050,
        hop_length: int = 256,
        win_length: Optional[int] = 1024,
        n_fft: int = 1024,
        use_phoneme_predictor: bool = False,
        expand_f0_method: str = "repeat",
        # hubert
        hubert_channels: Union[int, None] = 0,
    ):
        """Initialize VITS generator module.

        Args:
            vocabs (int): Input vocabulary size.
            aux_channels (int): Number of acoustic feature channels.
            hidden_channels (int): Number of hidden channels.
            spks (Optional[int]): Number of speakers. If set to > 1, assume that the
                sids will be provided as the input and use sid embedding layer.
            langs (Optional[int]): Number of languages. If set to > 1, assume that the
                lids will be provided as the input and use sid embedding layer.
            spk_embed_dim (Optional[int]): Speaker embedding dimension. If set to > 0,
                assume that spembs will be provided as the input.
            global_channels (int): Number of global conditioning channels.
            segment_size (int): Segment size for decoder.
            text_encoder_attention_heads (int): Number of heads in conformer block
                of text encoder.
            text_encoder_ffn_expand (int): Expansion ratio of FFN in conformer block
                of text encoder.
            text_encoder_blocks (int): Number of conformer blocks in text encoder.
            text_encoder_positionwise_layer_type (str): Position-wise layer type in
                conformer block of text encoder.
            text_encoder_positionwise_conv_kernel_size (int): Position-wise convolution
                kernel size in conformer block of text encoder. Only used when the
                above layer type is conv1d or conv1d-linear.
            text_encoder_positional_encoding_layer_type (str): Positional encoding layer
                type in conformer block of text encoder.
            text_encoder_self_attention_layer_type (str): Self-attention layer type in
                conformer block of text encoder.
            text_encoder_activation_type (str): Activation function type in conformer
                block of text encoder.
            text_encoder_normalize_before (bool): Whether to apply layer norm before
                self-attention in conformer block of text encoder.
            text_encoder_dropout_rate (float): Dropout rate in conformer block of
                text encoder.
            text_encoder_positional_dropout_rate (float): Dropout rate for positional
                encoding in conformer block of text encoder.
            text_encoder_attention_dropout_rate (float): Dropout rate for attention in
                conformer block of text encoder.
            text_encoder_conformer_kernel_size (int): Conformer conv kernel size. It
                will be used when only use_conformer_conv_in_text_encoder = True.
            use_macaron_style_in_text_encoder (bool): Whether to use macaron style FFN
                in conformer block of text encoder.
            use_conformer_conv_in_text_encoder (bool): Whether to use covolution in
                conformer block of text encoder.
            decoder_kernel_size (int): Decoder kernel size.
            decoder_channels (int): Number of decoder initial channels.
            decoder_downsample_scales (List[int]): List of downsampling scales in
                decoder.
            decoder_downsample_kernel_sizes (List[int]): List of kernel sizes for
                downsampling layers in decoder.
            decoder_upsample_scales (List[int]): List of upsampling scales in decoder.
            decoder_upsample_kernel_sizes (List[int]): List of kernel sizes for
                upsampling layers in decoder.
            decoder_resblock_kernel_sizes (List[int]): List of kernel sizes for
                resblocks in decoder.
            decoder_resblock_dilations (List[List[int]]): List of list of dilations for
                resblocks in decoder.
            use_avocodo (bool): Whether to use Avocodo model in the generator.
            projection_filters (List[int]): List of projection filter sizes.
            projection_kernels (List[int]): List of projection kernel sizes.
            n_harmonic (int): Number of harmonic components.
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
            generator_type (str): Type of generator to use for the model.
            vocoder_generator_type (str): Type of vocoder generator to use for the
                model.
            fs (int): Sample rate of the audio.
            hop_length (int): Number of samples between successive frames in STFT.
            win_length (int): Window size of the STFT.
            n_fft (int): Length of the FFT window to be used.
            use_phoneme_predictor (bool): Whether to use phoneme predictor in the model.
            expand_f0_method (str): The method used to expand F0. Use "repeat" or
                "interpolation".
            hubert_channels (int): Number of channels in the Hubert model.
                This is used in VISinger2 Plus.
        """
        super().__init__()
        self.aux_channels = aux_channels
        self.hidden_channels = hidden_channels
        self.generator_type = generator_type
        self.segment_size = segment_size
        self.sample_rate = fs
        self.hop_length = hop_length
        self.use_avocodo = use_avocodo
        self.use_flow = True if flow_flows > 0 else False
        self.use_phoneme_predictor = use_phoneme_predictor
        self.text_encoder = TextEncoder(
            vocabs=vocabs,
            attention_dim=hidden_channels,
            attention_heads=text_encoder_attention_heads,
            linear_units=hidden_channels * text_encoder_ffn_expand,
            blocks=text_encoder_blocks,
            positionwise_layer_type=text_encoder_positionwise_layer_type,
            positionwise_conv_kernel_size=text_encoder_positionwise_conv_kernel_size,
            positional_encoding_layer_type=text_encoder_positional_encoding_layer_type,
            self_attention_layer_type=text_encoder_self_attention_layer_type,
            activation_type=text_encoder_activation_type,
            normalize_before=text_encoder_normalize_before,
            dropout_rate=text_encoder_dropout_rate,
            positional_dropout_rate=text_encoder_positional_dropout_rate,
            attention_dropout_rate=text_encoder_attention_dropout_rate,
            conformer_kernel_size=text_encoder_conformer_kernel_size,
            use_macaron_style=use_macaron_style_in_text_encoder,
            use_conformer_conv=use_conformer_conv_in_text_encoder,
        )
        if vocoder_generator_type == "uhifigan":
            self.decoder = UHiFiGANGenerator(
                in_channels=hidden_channels,
                out_channels=1,
                channels=decoder_channels,
                global_channels=global_channels,
                kernel_size=decoder_kernel_size,
                downsample_scales=decoder_downsample_scales,
                downsample_kernel_sizes=decoder_downsample_kernel_sizes,
                upsample_scales=decoder_upsample_scales,
                upsample_kernel_sizes=decoder_upsample_kernel_sizes,
                resblock_kernel_sizes=decoder_resblock_kernel_sizes,
                resblock_dilations=decoder_resblock_dilations,
                use_weight_norm=use_weight_norm_in_decoder,
                use_avocodo=use_avocodo,
            )
            self.sine_generator = SineGen(
                sample_rate=fs,
            )
        elif vocoder_generator_type == "hifigan":
            self.decoder = HiFiGANGenerator(
                in_channels=hidden_channels,
                out_channels=1,
                channels=decoder_channels,
                global_channels=global_channels,
                kernel_size=decoder_kernel_size,
                upsample_scales=decoder_upsample_scales,
                upsample_kernel_sizes=decoder_upsample_kernel_sizes,
                resblock_kernel_sizes=decoder_resblock_kernel_sizes,
                resblock_dilations=decoder_resblock_dilations,
                use_weight_norm=use_weight_norm_in_decoder,
            )
        elif vocoder_generator_type == "avocodo":
            self.decoder = AvocodoGenerator(
                in_channels=hidden_channels,
                out_channels=1,
                channels=decoder_channels,
                global_channels=global_channels,
                kernel_size=decoder_kernel_size,
                upsample_scales=decoder_upsample_scales,
                upsample_kernel_sizes=decoder_upsample_kernel_sizes,
                resblock_kernel_sizes=decoder_resblock_kernel_sizes,
                resblock_dilations=decoder_resblock_dilations,
                projection_filters=projection_filters,
                projection_kernels=projection_kernels,
                use_weight_norm=use_weight_norm_in_decoder,
            )
        elif vocoder_generator_type == "visinger2":
            self.decoder = VISinger2VocoderGenerator(
                in_channels=hidden_channels,
                out_channels=1,
                channels=decoder_channels,
                global_channels=global_channels,
                kernel_size=decoder_kernel_size,
                upsample_scales=decoder_upsample_scales,
                upsample_kernel_sizes=decoder_upsample_kernel_sizes,
                resblock_kernel_sizes=decoder_resblock_kernel_sizes,
                resblock_dilations=decoder_resblock_dilations,
                use_weight_norm=use_weight_norm_in_decoder,
                n_harmonic=n_harmonic,
            )
            self.dec_harm = Generator_Harm(
                hidden_channels=hidden_channels,
                n_harmonic=n_harmonic,
                kernel_size=3,
                padding=1,
                dropout_rate=0.1,
                sample_rate=fs,
                hop_size=hop_length,
            )
            self.dec_noise = Generator_Noise(
                win_length=win_length,
                hop_length=hop_length,
                n_fft=n_fft,
                hidden_channels=hidden_channels,
                kernel_size=3,
                padding=1,
                dropout_rate=0.1,
            )
            self.sin_prenet = torch.nn.Conv1d(1, n_harmonic + 2, 3, padding=1)
        else:
            raise ValueError(
                f"Not supported vocoder generator type: {vocoder_generator_type}"
            )
        if hubert_channels is None:
            hubert_channels = 0
        self.posterior_encoder = PosteriorEncoder(
            in_channels=hubert_channels + aux_channels,
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
        if self.use_flow:
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

        self.f0_prenet = torch.nn.Conv1d(1, hidden_channels + 2, 3, padding=1)

        if generator_type == "visinger2":
            self.energy_prenet = torch.nn.Conv1d(1, hidden_channels + 2, 3, padding=1)
            self.mel_prenet = torch.nn.Conv1d(
                aux_channels, hidden_channels + 2, 3, padding=1
            )

        # TODO(kan-bayashi): Add deterministic version as an option

        self.duration_predictor = DurationPredictor(
            channels=hidden_channels,
            filter_channels=256,
            kernel_size=3,
            dropout_rate=0.5,
            global_channels=global_channels,
        )

        self.lr = LengthRegulator()

        if self.use_phoneme_predictor:
            self.phoneme_predictor = PhonemePredictor(
                vocabs=vocabs,
                hidden_channels=hidden_channels,
                attention_dim=hidden_channels,
                blocks=2,
            )

        self.f0_decoder = Decoder(
            1,
            attention_dim=hidden_channels,
            attention_heads=text_encoder_attention_heads,
            linear_units=hidden_channels * text_encoder_ffn_expand,
            blocks=text_encoder_blocks,
            pw_layer_type=text_encoder_positionwise_layer_type,
            pw_conv_kernel_size=text_encoder_positionwise_conv_kernel_size,
            pos_enc_layer_type=text_encoder_positional_encoding_layer_type,
            self_attention_layer_type=text_encoder_self_attention_layer_type,
            activation_type=text_encoder_activation_type,
            normalize_before=text_encoder_normalize_before,
            dropout_rate=text_encoder_dropout_rate,
            positional_dropout_rate=text_encoder_positional_dropout_rate,
            attention_dropout_rate=text_encoder_attention_dropout_rate,
            conformer_kernel_size=text_encoder_conformer_kernel_size,
            use_macaron_style=use_macaron_style_in_text_encoder,
            use_conformer_conv=use_conformer_conv_in_text_encoder,
            global_channels=global_channels,
        )

        if self.generator_type == "visinger2":
            self.mel_decoder = Decoder(
                out_channels=aux_channels,
                attention_dim=hidden_channels,
                attention_heads=text_encoder_attention_heads,
                linear_units=hidden_channels * text_encoder_ffn_expand,
                blocks=text_encoder_blocks,
                pw_layer_type=text_encoder_positionwise_layer_type,
                pw_conv_kernel_size=text_encoder_positionwise_conv_kernel_size,
                pos_enc_layer_type=text_encoder_positional_encoding_layer_type,
                self_attention_layer_type=text_encoder_self_attention_layer_type,
                activation_type=text_encoder_activation_type,
                normalize_before=text_encoder_normalize_before,
                dropout_rate=text_encoder_dropout_rate,
                positional_dropout_rate=text_encoder_positional_dropout_rate,
                attention_dropout_rate=text_encoder_attention_dropout_rate,
                conformer_kernel_size=text_encoder_conformer_kernel_size,
                use_macaron_style=use_macaron_style_in_text_encoder,
                use_conformer_conv=use_conformer_conv_in_text_encoder,
                global_channels=global_channels,
            )

        self.prior_decoder = PriorDecoder(
            out_channels=hidden_channels * 2,
            attention_dim=hidden_channels,
            attention_heads=text_encoder_attention_heads,
            linear_units=hidden_channels * text_encoder_ffn_expand,
            blocks=text_encoder_blocks,
            positionwise_layer_type=text_encoder_positionwise_layer_type,
            positionwise_conv_kernel_size=text_encoder_positionwise_conv_kernel_size,
            positional_encoding_layer_type=text_encoder_positional_encoding_layer_type,
            self_attention_layer_type=text_encoder_self_attention_layer_type,
            activation_type=text_encoder_activation_type,
            normalize_before=text_encoder_normalize_before,
            dropout_rate=text_encoder_dropout_rate,
            positional_dropout_rate=text_encoder_positional_dropout_rate,
            attention_dropout_rate=text_encoder_attention_dropout_rate,
            conformer_kernel_size=text_encoder_conformer_kernel_size,
            use_macaron_style=use_macaron_style_in_text_encoder,
            use_conformer_conv=use_conformer_conv_in_text_encoder,
            global_channels=global_channels,
        )

        self.upsample_factor = int(np.prod(decoder_upsample_scales))
        self.spks = None

        if spks is not None and spks > 1:
            assert global_channels > 0
            self.spks = spks
            self.global_emb = torch.nn.Embedding(spks, global_channels)
        self.spk_embed_dim = None
        if spk_embed_dim is not None and spk_embed_dim > 0:
            assert global_channels > 0
            self.spk_embed_dim = spk_embed_dim
            self.spemb_proj = torch.nn.Linear(spk_embed_dim, global_channels)
        self.langs = None
        if langs is not None and langs > 1:
            assert global_channels > 0
            self.langs = langs
            self.lang_emb = torch.nn.Embedding(langs, global_channels)

        self.vocoder_generator_type = vocoder_generator_type
        self.dropout = torch.nn.Dropout(0.2)
        self.expand_f0_method = expand_f0_method

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
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
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
        """Calculate forward propagation.

        Args:
            text (LongTensor): Batch of padded character ids (B, Tmax).
            text_lengths (LongTensor): Batch of lengths of each input batch (B,).
            feats (Tensor): Batch of padded target features (B, Lmax, odim).
            feats_lengths (LongTensor): Batch of the lengths of each target (B,).
            label (LongTensor): Batch of padded label ids (B, Tmax).
            label_lengths (LongTensor): Batch of the lengths of padded label ids (B, ).
            melody (LongTensor): Batch of padded midi (B, Tmax).
            gt_dur (LongTensor): Batch of padded ground truth duration (B, Tmax).
            score_dur (LongTensor): Batch of padded score duration (B, Tmax).
            pitch (FloatTensor): Batch of padded f0 (B, Tmax).
            ying (Optional[Tensor]): Batch of padded ying (B, Tmax).
            spembs (Optional[Tensor]): Batch of speaker embeddings (B, spk_embed_dim).
            sids (Optional[Tensor]): Batch of speaker IDs (B, 1).
            lids (Optional[Tensor]): Batch of language IDs (B, 1).

        Returns:
            Tensor: Waveform tensor (B, 1, segment_size * upsample_factor).
            Tensor: Duration negative log-likelihood (NLL) tensor (B,).
            Tensor: Monotonic attention weight tensor (B, 1, T_feats, T_text).
            Tensor: Segments start index tensor (B,).
            Tensor: Text mask tensor (B, 1, T_text).
            Tensor: Feature mask tensor (B, 1, T_feats).
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
                - Tensor: Posterior encoder hidden representation (B, H, T_feats).
                - Tensor: Flow hidden representation (B, H, T_feats).
                - Tensor: Expanded text encoder projected mean (B, H, T_feats).
                - Tensor: Expanded text encoder projected scale (B, H, T_feats).
                - Tensor: Posterior encoder projected mean (B, H, T_feats).
                - Tensor: Posterior encoder projected scale (B, H, T_feats).

        """
        # calculate global conditioning
        g = None
        if self.spks is not None:
            # speaker one-hot vector embedding: (B, global_channels, 1)
            g = self.global_emb(sids.view(-1)).unsqueeze(-1)
        if self.spk_embed_dim is not None:
            # pretreined speaker embedding, e.g., X-vector (B, global_channels, 1)
            g_ = self.spemb_proj(F.normalize(spembs)).unsqueeze(-1)
            if g is None:
                g = g_
            else:
                g = g + g_
        if self.langs is not None:
            # language one-hot vector embedding: (B, global_channels, 1)
            g_ = self.lang_emb(lids.view(-1)).unsqueeze(-1)
            if g is None:
                g = g_
            else:
                g = g + g_

        # forward text encoder

        # Encoder
        x, x_mask, dur_input, x_pitch = self.text_encoder(
            label, label_lengths, melody, score_dur, slur
        )

        # dur
        # Note this is different, we use frame level duration not time level
        # but it has no big difference on performance
        predict_dur = self.duration_predictor(dur_input, x_mask, g=g)
        predict_dur = (torch.exp(predict_dur) - 1) * x_mask
        predict_dur = predict_dur * self.sample_rate / self.hop_length

        # LR
        decoder_input, mel_len = self.lr(x, gt_dur, use_state_info=True)  # noqa
        decoder_input_pitch, mel_len = self.lr(  # noqa
            x_pitch, gt_dur, use_state_info=True
        )  # noqa

        LF0 = 2595.0 * torch.log10(1.0 + pitch / 700.0)
        LF0 = LF0 / 500
        LF0 = LF0.transpose(1, 2)

        predict_lf0, predict_bn_mask = self.f0_decoder(
            decoder_input + decoder_input_pitch, feats_lengths, g=g
        )
        predict_lf0 = torch.max(
            predict_lf0, torch.zeros_like(predict_lf0).to(predict_lf0)
        )

        if self.generator_type == "visinger2":
            predict_mel, predict_bn_mask = self.mel_decoder(
                decoder_input + self.f0_prenet(LF0), feats_lengths, g=g
            )

            predict_energy = (
                predict_mel.detach().sum(1).unsqueeze(1) / self.aux_channels
            )

        decoder_input = decoder_input + self.f0_prenet(LF0)
        if self.generator_type == "visinger2":
            decoder_input = (
                decoder_input
                + self.energy_prenet(predict_energy)
                + self.mel_prenet(predict_mel.detach())
            )
        decoder_output, predict_bn_mask = self.prior_decoder(
            decoder_input, feats_lengths, g=g
        )

        prior_info = decoder_output
        prior_mean = prior_info[:, : self.hidden_channels, :]
        prior_logstd = prior_info[:, self.hidden_channels :, :]

        # forward posterior encoder
        posterior_z, posterior_mean, posterior_logstd, y_mask = self.posterior_encoder(
            feats, feats_lengths, g=g
        )

        if self.use_flow:
            z_flow = self.flow(posterior_z, y_mask, g=g)
        else:
            z_flow = None

        # phoneme predictor
        if self.use_phoneme_predictor:
            log_probs = self.phoneme_predictor(posterior_z, y_mask)
        else:
            log_probs = None

        p_z = posterior_z
        p_z = self.dropout(p_z)

        # get random segments
        z_segments, z_start_idxs = get_random_segments(
            p_z, feats_lengths, self.segment_size
        )

        if self.vocoder_generator_type == "uhifigan":
            # get sine wave

            # def plot_sine_waves(sine_waves, name):
            #     import matplotlib.pyplot as plt

            #     sine_waves_np = sine_waves[0].detach().cpu().numpy()
            #     plt.plot(sine_waves_np)
            #     plt.xlabel("Time (samples)")
            #     plt.ylabel("Amplitude")
            #     plt.title("Sine Wave")
            #     plt.savefig(name + ".png")
            #     plt.close()

            # plot_sine_waves(pitch_segments[0], "pitch_segments")

            pitch_segments = get_segments(
                pitch.transpose(1, 2), z_start_idxs, self.segment_size
            )
            pitch_segments_expended = expand_f0(
                pitch_segments, self.hop_length, method="repeat"
            )

            # plot_sine_waves(
            #     pitch_segments_expended[0].unsqueeze(0), "pitch_segments_expended"
            # )
            pitch_segments_expended = pitch_segments_expended.reshape(
                -1, pitch_segments_expended.shape[-1], 1
            )

            sine_waves, uv, noise = self.sine_generator(pitch_segments_expended)  # noqa

            sine_waves = sine_waves.transpose(1, 2)

            wav = self.decoder(z_segments, excitation=sine_waves, g=g)
        elif self.vocoder_generator_type == "visinger2":
            pitch_ = upsample(pitch, self.hop_length)
            omega = torch.cumsum(2 * math.pi * pitch_ / self.sample_rate, 1)
            sin = torch.sin(omega).transpose(1, 2)

            # dsp synthesize
            pitch = pitch.transpose(1, 2)
            noise_x = self.dec_noise(posterior_z, y_mask)
            harm_x = self.dec_harm(pitch, posterior_z, y_mask)

            # dsp waveform
            dsp_o = torch.cat([harm_x, noise_x], axis=1)

            # decoder_condition = torch.cat([harm_x, noise_x, sin], axis=1)
            decoder_condition = self.sin_prenet(sin)

            # dsp based HiFiGAN vocoder
            dsp_slice = get_segments(
                dsp_o,
                z_start_idxs * self.hop_length,
                self.segment_size * self.hop_length,
            )

            condition_slice = get_segments(
                decoder_condition,
                z_start_idxs * self.hop_length,
                self.segment_size * self.hop_length,
            )
            wav = self.decoder(z_segments, condition_slice, g=g)
        else:
            wav = self.decoder(z_segments, g=g)

            # wav = dsp_slice.sum(1, keepdim=True)

        common_tuple = (
            posterior_z,
            z_flow,
            prior_mean,
            prior_logstd,
            posterior_mean,
            posterior_logstd,
            predict_lf0,
            LF0 * predict_bn_mask,
            predict_dur,
            gt_dur,
            log_probs,
        )

        output = (wav, z_start_idxs, x_mask, y_mask, common_tuple)

        if self.vocoder_generator_type == "visinger2":
            output = output + (dsp_slice.sum(1),)
        if self.generator_type == "visinger2":
            output = output + (predict_mel,)
        return output

    def inference(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        feats_lengths: Optional[torch.Tensor] = None,
        label: torch.Tensor = None,
        label_lengths: torch.Tensor = None,
        melody: torch.Tensor = None,
        score_dur: torch.Tensor = None,
        slur: torch.Tensor = None,
        gt_dur: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        noise_scale: float = 0.667,
        noise_scale_dur: float = 0.8,
        alpha: float = 1.0,
        max_len: Optional[int] = None,
        use_teacher_forcing: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run inference.

        Args:
            text (LongTensor): Batch of padded character ids (B, Tmax).
            text_lengths (LongTensor): Batch of lengths of each input batch (B,).
            feats (Tensor): Batch of padded target features (B, Lmax, odim).
            feats_lengths (LongTensor): Batch of the lengths of each target (B,).
            label (LongTensor): Batch of padded label ids (B, Tmax).
            label_lengths (LongTensor): Batch of the lengths of padded label ids (B, ).
            melody (LongTensor): Batch of padded midi (B, Tmax).
            gt_dur (LongTensor): Batch of padded ground truth duration (B, Tmax).
            score_dur (LongTensor): Batch of padded score duration (B, Tmax).
            pitch (FloatTensor): Batch of padded f0 (B, Tmax).
            ying (Optional[Tensor]): Batch of padded ying (B, Tmax).
            spembs (Optional[Tensor]): Batch of speaker embeddings (B, spk_embed_dim).
            sids (Optional[Tensor]): Batch of speaker IDs (B, 1).
            lids (Optional[Tensor]): Batch of language IDs (B, 1).
            noise_scale (float): Noise scale parameter for flow.
            noise_scale_dur (float): Noise scale parameter for duration predictor.
            alpha (float): Alpha parameter to control the speed of generated speech.
            max_len (Optional[int]): Maximum length of acoustic feature sequence.
            use_teacher_forcing (bool): Whether to use teacher forcing.

        Returns:
            Tensor: Generated waveform tensor (B, T_wav).

        """
        # encoder
        x, x_mask, dur_input, x_pitch = self.text_encoder(
            label, label_lengths, melody, score_dur, slur
        )
        g = None
        if self.spks is not None:
            # (B, global_channels, 1)
            g = self.global_emb(sids.view(-1)).unsqueeze(-1)
        if self.spk_embed_dim is not None:
            # (B, global_channels, 1)
            g_ = self.spemb_proj(F.normalize(spembs.unsqueeze(0))).unsqueeze(-1)
            if g is None:
                g = g_
            else:
                g = g + g_
        if self.langs is not None:
            # (B, global_channels, 1)
            g_ = self.lang_emb(lids.view(-1)).unsqueeze(-1)
            if g is None:
                g = g_
            else:
                g = g + g_

        if use_teacher_forcing:
            # forward posterior encoder
            z, m_q, logs_q, y_mask = self.posterior_encoder(  # noqa
                feats, feats_lengths, g=g
            )  # noqa

            # forward flow
            if self.use_flow:
                z_p = self.flow(z, y_mask, g=g)  # (B, H, T_feats)

            # decoder
            pitch = pitch.transpose(0, 1).reshape(1, 1, -1)

            if self.vocoder_generator_type == "uhifigan":
                pitch_segments_expended = expand_f0(
                    pitch, self.hop_length, method=self.expand_f0_method
                )
                pitch_segments_expended = pitch_segments_expended.reshape(
                    -1, pitch_segments_expended.shape[-1], 1
                )
                sine_waves, _, _ = self.sine_generator(pitch_segments_expended)
                sine_waves = sine_waves.transpose(1, 2)
                wav = self.decoder(
                    (z * y_mask)[:, :, :max_len], excitation=sine_waves, g=g
                )
            elif self.vocoder_generator_type == "avocodo":
                wav = self.decoder((z * y_mask)[:, :, :max_len], g=g)[-1]
            elif self.vocoder_generator_type == "visinger2":
                pitch_ = upsample(pitch.transpose(1, 2), self.hop_length)
                omega = torch.cumsum(2 * math.pi * pitch_ / self.sample_rate, 1)
                sin = torch.sin(omega).transpose(1, 2)

                # dsp synthesize
                noise_x = self.dec_noise(z, y_mask)
                harm_x = self.dec_harm(pitch, z, y_mask)

                # dsp waveform
                dsp_o = torch.cat([harm_x, noise_x], axis=1)  # noqa

                # decoder_condition = torch.cat([harm_x, noise_x, sin], axis=1)
                decoder_condition = self.sin_prenet(sin)

                # dsp based HiFiGAN vocoder
                wav = self.decoder((z * y_mask)[:, :, :max_len], decoder_condition, g=g)
                # wav = dsp_o.sum(1)
                # wav = noise_x
                # wav = harm_x.sum(1)
            else:
                wav = self.decoder((z * y_mask)[:, :, :max_len], g=g)
        else:
            # dur
            predict_dur = self.duration_predictor(dur_input, x_mask, g=g)
            predict_dur = (torch.exp(predict_dur) - 1) * x_mask
            predict_dur = predict_dur * self.sample_rate / self.hop_length

            predict_dur = torch.max(predict_dur, torch.ones_like(predict_dur).to(x))
            predict_dur = torch.ceil(predict_dur).long()
            predict_dur = predict_dur[:, 0, :]
            y_lengths = torch.clamp_min(torch.sum(predict_dur, [1]), 1).long()

            # LR
            decoder_input, mel_len = self.lr(
                x, predict_dur, use_state_info=True
            )  # noqa
            decoder_input_pitch, mel_len = self.lr(  # noqa
                x_pitch, predict_dur, use_state_info=True
            )  # noqa

            # aam
            predict_lf0, predict_bn_mask = self.f0_decoder(  # noqa
                decoder_input + decoder_input_pitch, y_lengths, g=g
            )  # noqa

            if self.generator_type == "visinger2":
                predict_mel, predict_bn_mask = self.mel_decoder(  # noqa
                    decoder_input + self.f0_prenet(predict_lf0),
                    y_lengths,
                    g=g,
                )
                predict_energy = predict_mel.sum(1).unsqueeze(1) / self.aux_channels

            predict_lf0 = torch.max(
                predict_lf0, torch.zeros_like(predict_lf0).to(predict_lf0)
            )

            decoder_input = decoder_input + self.f0_prenet(predict_lf0)
            if self.generator_type == "visinger2":
                decoder_input = (
                    decoder_input
                    + self.energy_prenet(predict_energy)
                    + self.mel_prenet(predict_mel)
                )

            decoder_output, y_mask = self.prior_decoder(decoder_input, y_lengths, g=g)

            prior_info = decoder_output
            m_p = prior_info[:, : self.hidden_channels, :]
            logs_p = prior_info[:, self.hidden_channels :, :]

            # decoder
            z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

            if self.use_flow:
                z = self.flow(z_p, y_mask, g=g, inverse=True)
            else:
                z = z_p

            F0_std = 500
            F0 = predict_lf0 * F0_std
            F0 = F0 / 2595
            F0 = torch.pow(10, F0)
            F0 = (F0 - 1) * 700.0

            if self.vocoder_generator_type == "uhifigan":
                pitch_segments_expended = expand_f0(
                    F0, self.hop_length, method=self.expand_f0_method
                )
                pitch_segments_expended = pitch_segments_expended.reshape(
                    -1, pitch_segments_expended.shape[-1], 1
                )
                sine_waves, uv, noise = self.sine_generator(pitch_segments_expended)
                sine_waves = sine_waves.transpose(1, 2)
                wav = self.decoder(
                    (z * y_mask)[:, :, :max_len], excitation=sine_waves, g=g
                )
            elif self.vocoder_generator_type == "avocodo":
                wav = self.decoder((z * y_mask)[:, :, :max_len], g=g)[-1]
            elif self.vocoder_generator_type == "visinger2":
                pitch_ = upsample(F0.transpose(1, 2), self.hop_length)
                omega = torch.cumsum(2 * math.pi * pitch_ / self.sample_rate, 1)
                sin = torch.sin(omega).transpose(1, 2)

                # dsp synthesize
                noise_x = self.dec_noise(z, y_mask)
                harm_x = self.dec_harm(F0, z, y_mask)

                # dsp waveform
                dsp_o = torch.cat([harm_x, noise_x], axis=1)  # noqa

                # decoder_condition = torch.cat([harm_x, noise_x, sin], axis=1)
                decoder_condition = self.sin_prenet(sin)

                # dsp based HiFiGAN vocoder
                wav = self.decoder((z * y_mask)[:, :, :max_len], decoder_condition, g=g)
                # wav = dsp_o.sum(1)
                # wav = noise_x
                # wav = harm_x.sum(1)
            else:
                wav = self.decoder((z * y_mask)[:, :, :max_len], g=g)

        return wav.squeeze(1)
