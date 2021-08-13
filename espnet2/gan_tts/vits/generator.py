# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Generator module in VITS.

This code is based on https://github.com/jaywalnut310/vits.

"""

import math

from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet2.gan_tts.vits.duration_predictor import StochasticDurationPredictor
from espnet2.gan_tts.vits.hifigan import HiFiGANGenerator
from espnet2.gan_tts.vits.posterior_encoder import PosteriorEncoder
from espnet2.gan_tts.vits.residual_coupling import ResidualAffineCouplingBlock
from espnet2.gan_tts.vits.text_encoder import TextEncoder


class VITSGenerator(torch.nn.Module):
    """Generator module in VITS.

    This is a module of VITS described in `Conditional Variational Autoencoder
    with Adversarial Learning for End-to-End Text-to-Speech`_.

    As text encoder, we use conformer architecture instead of the relative positional
    Transformer, which contains additional convolution layers.

    .. _`Conditional Variational Autoencoder with Adversarial Learning for End-to-End
        Text-to-Speech`: https://arxiv.org/abs/2006.04558

    """

    def __init__(
        self,
        vocabs: int,
        aux_channels: int = 513,
        hidden_channels: int = 192,
        spks: int = -1,
        global_channels: int = -1,
        segment_size: int = 32,
        text_encoder_attention_heads: int = 2,
        text_encoder_ffn_expand: int = 4,
        text_encoder_blocks: int = 6,
        text_encoder_positionwise_layer_type: str = "conv1d",
        text_encoder_positionwise_conv_kernel_size: int = 1,
        text_encoder_normalize_before: bool = True,
        text_encoder_dropout_rate: float = 0.1,
        text_encoder_positional_dropout_rate: float = 0.0,
        text_encoder_attention_dropout_rate: float = 0.0,
        text_encoder_conformer_kernel_size: int = 7,
        use_macaron_style_in_text_encoder: bool = True,
        use_conformer_conv_in_text_encoder: bool = True,
        decoder_kernel_size: int = 7,
        decoder_channels: int = 512,
        decoder_upsample_scales: List[int] = [8, 8, 2, 2],
        decoder_upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        decoder_resblock_kernel_sizes: List[int] = [3, 7, 11],
        decoder_resblock_dilations: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
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
        stochastic_duration_predictor_kernel_size: int = 3,
        stochastic_duration_predictor_dropout_rate: float = 0.5,
        stochastic_duration_predictor_flows: int = 4,
        stochastic_duration_predictor_dds_conv_layers: int = 3,
    ):
        """Initialize VITS generator module.

        Args:
            vocabs (int): Input vocabulary size.
            aux_channels (int): Number of acoustic feature channels.
            hidden_channels (int): Number of hidden channels.
            spks (int): Number of speakers.
            global_channels (int): Number of global conditioning channels.
            segment_size (int): Segment size for decoder.
            text_encoder_attention_heads (int): Number of heads in text encoder.
            text_encoder_ffn_expand (int): Expansion ratio of FFN in text encoder.
            text_encoder_blocks (int): Number of blocks in text encoder.
            text_encoder_positionwise_layer_type (str): Position-wise layer type.
            text_encoder_positionwise_conv_kernel_size (int): Position-wise convolution
                kernal size. Only used when the above layer is conv1d or conv1d-linear.
            text_encoder_normalize_before (bool): Whether to appli layer norm before
                self-attention in conformer block.
            text_encoder_dropout_rate (float): Dropout rate in text encoder.
            text_encoder_positional_dropout_rate (float): Dropout rate for positional
                encoding in text encoder.
            text_encoder_attention_dropout_rate (float): Dropout rate for attention in
                text encoder.
            text_encoder_conformer_kernel_size (int): Conformer conv kernal size in
                text encoder. Only used when use_conformer_conv_in_text_encoder = True.
            use_macaron_style_in_text_encoder (bool): Whether to use macaron style FFN
                in conformer block of text encoder.
            use_conformer_conv_in_text_encoder (bool): Whether to use covolution in
                conformer block of text encoder.
            decoder_kernel_size (int): Decoder kernel size.
            decoder_channels (int): Number of decoder initial channels.
            decoder_upsample_scales (List[int]): List of upsampling scales in decoder.
            decoder_upsample_kernel_sizes (List[int]): List of kernel size for
                upsampling layers in decoder.
            decoder_resblock_kernel_sizes (List[int]): List of kernel size for resblocks
                in decoder.
            decoder_resblock_dilations (List[List[int]]): List of list of dilations for
                resblocks in decoder.
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
            vocabs=vocabs,
            attention_dim=hidden_channels,
            attention_heads=text_encoder_attention_heads,
            linear_units=hidden_channels * text_encoder_ffn_expand,
            blocks=text_encoder_blocks,
            positionwise_layer_type=text_encoder_positionwise_layer_type,
            positionwise_conv_kernel_size=text_encoder_positionwise_conv_kernel_size,
            normalize_before=text_encoder_normalize_before,
            dropout_rate=text_encoder_dropout_rate,
            positional_dropout_rate=text_encoder_positional_dropout_rate,
            attention_dropout_rate=text_encoder_attention_dropout_rate,
            conformer_kernel_size=text_encoder_conformer_kernel_size,
            use_macaron_style=use_macaron_style_in_text_encoder,
            use_conformer_conv=use_conformer_conv_in_text_encoder,
        )
        self.decoder = HiFiGANGenerator(
            in_channels=hidden_channels,
            out_channels=1,
            channels=decoder_channels,
            global_channels=global_channels,
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

        self.upsample_factor = int(np.prod(decoder_upsample_scales))
        self.spks = spks
        if self.spks > 1:
            assert global_channels > 0
            self.global_emb = torch.nn.Embedding(spks, global_channels)

        # delayed import
        from espnet2.gan_tts.vits.monotonic_align import maximum_path

        self.maximum_path = maximum_path

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        sids: Optional[torch.Tensor] = None,
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
        g = None
        if self.spks > 0:
            # (B, global_channels, 1)
            g = self.global_emb(sids).unsqueeze(-1)

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

        # forward duration predictor
        w = attn.sum(2)  # (B, 1, T_text)
        dur_nll = self.duration_predictor(x, x_mask, w=w, g=g)
        dur_nll = dur_nll / torch.sum(x_mask)

        # expand the length to match with the feature sequence
        # (B, T_feats, T_text) x (B, T_text, H) -> (B, H, T_feats)
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        # (B, T_feats, T_text) x (B, T_text, H) -> (B, H, T_feats)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        # get random segments
        z_segments, z_start_idxs = self.get_random_segments(
            z,
            feats_lengths,
            self.segment_size,
        )

        # forward decoder with random segments
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
        text_lengths: torch.Tensor,
        sids: Optional[torch.Tensor] = None,
        dur: Optional[torch.Tensor] = None,
        noise_scale: float = 1.0,
        length_scale: float = 1.0,
        noise_scale_w: float = 1.0,
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            Tensor: Monotonic attention weight tensor (B, T_feats, T_text).
            Tensor: Duration tensor (B, T_text).

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

        return wav.squeeze(1), attn.squeeze(1), dur.squeeze(1)

    def _generate_path(self, dur: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Generate path a.k.a. monotonic attention.

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
