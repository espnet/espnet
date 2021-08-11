# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""VITS: Variational Inference with adversarial learning for end-to-end Text-to-Speech.

This code is based on the official implementation:
- https://github.com/jaywalnut310/vits

"""

import math

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet2.tts.vits.flow import ResidualAffineCouplingBlock
from espnet2.tts.vits.hifigan import HiFiGANGenerator
from espnet2.tts.vits.posterior_encoder import PosteriorEncoder
from espnet2.tts.vits.stochastic_duration_predictor import StochasticDurationPredictor
from espnet2.tts.vits.text_encoder import TextEncoder


class VITS(torch.nn.Module):
    """VITS module.

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
        """Initilize VITS module.

        Args:
            idim (int): Input dimension.
            odim (int): Output dimenstion.
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
            use_weight_norm_in_decoder (bool): Whether to apply weight normlization in
                decoder.
            posterior_encoder_kernel_size (int): Posterior encoder kernel size.
            posterior_encoder_layers (int): Number of layers of posterior encoder.
            posterior_encoder_stacks (int): Number of stacks of posterior encoder.
            posterior_encoder_base_dilation (int): Base dilation of posterior encoder.
            posterior_encoder_dropout_rate (float): Dropout rate for posterior encoder.
            use_weight_norm_in_posterior_encoder (bool): Whether to apply weight
                normlization in posterior encoder.
            flow_flows (int): Number of flows in flow.
            flow_kernel_size (int): Kernel sizein flow.
            flow_base_dilation (int): Base dilation in flow.
            flow_layers (int): Number of layers in flow.
            flow_dropout_rate (float): Dropout rate in flow
            use_weight_norm_in_flow (bool): Whether to apply weight normlization in
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

        self.spks = spks
        if self.spks > 1:
            assert global_channels > 0
            self.global_emb = torch.nn.Embedding(spks, global_channels)

        # delayed import
        from espnet2.tts.vits.monotonic_align import maximum_path

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
            feats_lengths (Tensor): Feature length tensor (B,)
            sids (Optional[Tensor]): Speaker index tensor (B,).

        Returns:
            Tensor: Waveform tensor (B, 1, segment_size * upsample_factor).
            Tensor: Duration tensor (B, 1, T).
            Tensor: Attention weight tensor (B, T_feat, T_text).
            Tensor: Segments start index tensor (B,).
            Tensor: Text mask tensor (B, 1, T_text).
            Tensor: Feature mask tensor (B, 1, T_feats).
            tuple[Tensor]: Tuple of tensors.

        """
        # forward text encoder
        x, m_p, logs_p, x_mask = self.text_encoder(text, text_lengths)

        # calcualte global conditioning
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
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            # (B, T_feats, T_text)
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
        dur = self.duration_predictor(x, x_mask, w=w, g=g)
        dur = dur / torch.sum(x_mask)

        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        # forward decoder with segments
        z_segments, z_start_idxs = self._get_random_segments(
            z,
            feats_lengths,
            self.segment_size,
        )
        wav = self.decoder(z_segments, g=g)

        return (
            wav,
            dur,
            attn,
            z_start_idxs,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
        )

    def _get_random_segments(self, x, x_lengths=None, segment_size=32):
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
        segments = self._get_segments(x, start_idxs, segment_size)
        return segments, start_idxs

    def _get_segments(self, x, start_idxs, segment_size=32):
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
        sid=None,
        noise_scale=1.0,
        length_scale=1.0,
        noise_scale_w=1.0,
        max_len=None,
    ):
        """Run inference.

        Args:
            text (Tensor): Input text index tensor (T_text,).
            sid (Tensor): Speaker index tensor (1,)
            noise_scale (float): Noise scale value for flow.
            length_scale (float): Length scaling value.
            noise_scale_w (float): Noise scale value for duration predictor.
            max_len (Optional[int]): Maximum length.

        Returns:
            Tensor: Generated waveform tensor (1, 1, T_wave).
            Tensor: Attention weight tensor (1, T_feats, T_text).
            Tensor: Feature mask tensor (1, 1, T_feats).
            List[Tensor]: List of tensors.

        """
        # setup
        text = text[None]
        text_lengths = torch.tensor(
            [text.size(1)],
            dtype=torch.long,
            device=text.device,
        )

        # encoder
        x, m_p, logs_p, x_mask = self.text_encoder(text, text_lengths)
        if self.spks > 0:
            g = self.global_emb(sid.view(1)).unsqueeze(-1)  # (B, global_channels, 1)
        else:
            g = None

        # duration
        logw = self.duration_predictor(
            x,
            x_mask,
            g=g,
            reverse=True,
            noise_scale=noise_scale_w,
        )
        w = torch.exp(logw) * x_mask * length_scale
        dur = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(dur, [1, 2]), 1).long()
        y_mask = make_non_pad_mask(y_lengths).unsqueeze(1).to(text.device)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = self._generate_path(dur, attn_mask)

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
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        wav = self.decoder((z * y_mask)[:, :, :max_len], g=g)

        return wav, attn, y_mask, (z, z_p, m_p, logs_p)

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
        path = path.view(b, t_x, t_y)
        path = (
            path
            - F.pad(path, self._convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
        )
        path = path.unsqueeze(1).transpose(2, 3) * mask
        return path

    def _convert_pad_shape(self, pad_shape):
        l_ = pad_shape[::-1]
        pad_shape = [item for sublist in l_ for item in sublist]
        return pad_shape
