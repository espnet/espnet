import argparse
import math
import random
import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.tts.abs_tts import AbsTTS
from espnet2.tts.gradtts import monotonic_align
from espnet2.tts.gradtts.denoiser import Diffusion
from espnet2.tts.gradtts.text import cmudict, text_to_sequence
from espnet2.tts.gradtts.text.symbols import symbols
from espnet2.tts.gradtts.text_encoder import TextEncoder
from espnet2.tts.gradtts.utils import (
    convert_pad_shape,
    duration_loss,
    fix_len_compatibility,
    generate_path,
    intersperse,
    sequence_mask,
)
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import DurationPredictor
from espnet.nets.pytorch_backend.fastspeech.length_regulator import LengthRegulator
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.encoder import (
    Encoder as TransformerEncoder,
)


class GradTTS(AbsTTS):
    def __init__(
        self,
        # network structure related
        idim=149,
        n_spks=1,
        spk_emb_dim=64,
        n_enc_channels=192,
        filter_channels=768,
        filter_channels_dp=256,
        n_heads=2,
        n_enc_layers=6,
        enc_kernel=3,
        enc_dropout=0.1,
        window_size=4,
        odim=80,
        # decoder parameters
        dec_dim=64,
        beta_min=0.05,
        beta_max=20.0,
        pe_scale=1000,
        # encoder parameters
        # encoder_type: str = "transformer",
        # decoder_type: str = "diffusion",
        # transformer_enc_dropout_rate: float = 0.1,
        # transformer_enc_positional_dropout_rate: float = 0.1,
        # transformer_enc_attn_dropout_rate: float = 0.1,
        # adim: int = 384,
        # aheads: int = 2,
        # elayers: int = 4,
        # eunits: int = 1024,
        # encoder_normalize_before: bool = True,
        # encoder_concat_after: bool = False,
        # positionwise_layer_type: str = "conv1d-linear",
        # positionwise_conv_kernel_size: int = 1,
        # use_scaled_pos_enc: bool = True,
        # reduction_factor: int = 1,
    ):
        super(GradTTS, self).__init__()
        self.n_vocab = idim
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_enc_channels = n_enc_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.enc_kernel = enc_kernel
        self.enc_dropout = enc_dropout
        self.window_size = window_size
        self.n_feats = odim
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        # self.encoder_type = encoder_type
        # self.decoder_type = decoder_type
        # use idx 0 as padding idx
        self.padding_idx = 0
        self.eos = idim - 1

        # self.use_scaled_pos_enc = use_scaled_pos_enc
        # self.reduction_factor = reduction_factor

        # get positional encoding class
        # pos_enc_class = (
        #     ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding
        # )

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        # self.langs = None
        self.encoder = TextEncoder(
            idim,
            odim,
            n_enc_channels,
            filter_channels,
            filter_channels_dp,
            n_heads,
            n_enc_layers,
            enc_kernel,
            enc_dropout,
            window_size,
        )
        self.decoder = Diffusion(
            odim, dec_dim, n_spks, spk_emb_dim, beta_min, beta_max, pe_scale
        )

    def relocate_input(self, x: list):
        """
        Relocates provided tensors to the same device set for the module.
        """
        device = next(self.parameters()).device
        for i in range(len(x)):
            if isinstance(x[i], torch.Tensor) and x[i].device != device:
                x[i] = x[i].to(device)
        return x

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        joint_training: bool = False,
        sids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_size = text.size(0)
        text = text[:, : text_lengths.max()]  # for data-parallel
        feats = feats[:, : feats_lengths.max()]  # for data-parallel

        x_lengths, attn_dur, x_mask, logw, y, y_mask, mu_y = self._forward(
            text,
            text_lengths,
            feats,
            feats_lengths,
            n_timesteps=100,
            temperature=1.0,
            stoc=False,
            n_spk=None,
            sids=None,
            length_scale=1.0,
            out_size=fix_len_compatibility(22050 // 256),
        )

        # calculate loss
        dur_loss, prior_loss, diff_loss = self.compute_loss(
            x_lengths,
            attn_dur,
            x_mask,
            logw,
            y,
            y_mask,
            mu_y,
        )
        loss = dur_loss + prior_loss + diff_loss

        stats = dict(
            dur_loss=dur_loss.item(),
            prior_loss=prior_loss.item(),
            diff_loss=diff_loss.item(),
        )

        if not joint_training:
            stats.update(loss=loss.item())
            loss, stats, weight = force_gatherable(
                (loss, stats, batch_size), loss.device
            )
            return loss, stats, weight
        # else:
        #     return loss, stats, after_outs if after_outs is not None else before_outs

    def _forward(
        self,
        x,
        x_lengths,
        y,
        y_lengths,
        n_timesteps,
        temperature,
        stoc,
        n_spk=None,
        sids: Optional[torch.Tensor] = None,
        length_scale=1.0,
        out_size=None,
    ):
        x, x_lengths, y, y_lengths = self.relocate_input([x, x_lengths, y, y_lengths])
        y = y.transpose(1, 2)
        print("y:", y.shape)

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(n_spk)
        else:
            spk = None

        start_time = time.time()
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)
        y_max_length = y.shape[-1]
        encoder_process_time = time.time() - start_time
        print(f"encoder process time: {encoder_process_time} seconds")

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        start_time = time.time()
        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad():
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            # print(factor.shape)
            # print(y.shape)
            # print((y ** 2).shape)
            y_square = torch.matmul(factor.transpose(1, 2), y**2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
            attn_dur = attn.detach()
            attn = attn.detach()

        MAS_time = time.time() - start_time
        print(f"MAS process time: {MAS_time} seconds")

        start_time = time.time()
        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(
                zip([0] * max_offset.shape[0], max_offset.cpu().numpy())
            )
            out_offset = torch.LongTensor(
                [
                    torch.tensor(random.choice(range(start, end)) if end > start else 0)
                    for start, end in offset_ranges
                ]
            ).to(y_lengths)

            attn_cut = torch.zeros(
                attn.shape[0],
                attn.shape[1],
                out_size,
                dtype=attn.dtype,
                device=attn.device,
            )
            y_cut = torch.zeros(
                y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device
            )
            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        align_and_get_muy_time = time.time() - start_time
        print(f"Align and get mu_y process: {align_and_get_muy_time} seconds")

        return x_lengths, attn_dur, x_mask, logw, y, y_mask, mu_y

    def compute_loss(
        self, x_lengths, attn_dur, x_mask, logw, y, y_mask, mu_y, n_spk=None
    ):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        logw_ = torch.log(1e-8 + torch.sum(attn_dur.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Compute loss of score-based decoder
        start_time = time.time()
        diff_loss, xt = self.decoder.compute_loss(y, y_mask, mu_y, n_spk)
        diff_loss_process_time = time.time() - start_time
        # record time of computing diff loss time
        print(
            f"diffusion stage and compute diff loss time: {diff_loss_process_time} seconds"
        )

        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)

        return dur_loss, prior_loss, diff_loss

    @torch.no_grad()
    def inference(
        self,
        text,
        n_timesteps=100,
        temperature=1.0,
        stoc=False,
        n_spks=None,
        sids: Optional[torch.Tensor] = None,
        length_scale=1.0,
        use_teacher_forcing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        cmu = cmudict.CMUDict(
            "/home/lsh/content/espnet/espnet2/tts/gradtts/cmu_dictionary"
        )
        # print('text:')
        # print(text)
        text = text.tolist()
        text = text[:64]
        print("text:")
        print(text)
        x = torch.LongTensor(text).cpu()[None]
        print("x:")
        print(x)
        print(x.shape)
        # x = F.pad(text, [0, 1], "constant", self.eos)
        # x_lengths = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        x_lengths = torch.LongTensor([x.shape[-1]]).cpu()
        # x, x_lengths = self.relocate_input([x, x_lengths])

        x, x_lengths = self.relocate_input([x, x_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(n_spks)
        else:
            spk = None

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(z, y_mask, mu_y, n_timesteps, stoc, spk)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]
        # decoder_outputs = decoder_outputs.transpose(1, 2)
        # decoder_outputs = decoder_outputs[0]
        print("encoder_outputs:")
        print(encoder_outputs.shape)
        print("decoder_outputs:")
        print(decoder_outputs.shape)
        w = w[0]
        print("duration:", w.shape)

        return dict(
            encoder_outputs=encoder_outputs,
            feat_gen=decoder_outputs[0],
            attn=attn[:, :, :y_max_length],
            duration=w,
        )
