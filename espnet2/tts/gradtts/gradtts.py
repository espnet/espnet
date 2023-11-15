from espnet2.tts.abs_tts import AbsTTS
from espnet2.tts.gradtts.utils import sequence_mask, generate_path, duration_loss, fix_len_compatibility, convert_pad_shape
import torch
from espnet2.tts.gradtts.denoiser import Diffusion
import math
import random
from espnet2.tts.gradtts import monotonic_align
from typing import Dict, Optional
import torch.nn.functional as F
import argparse
from espnet2.tts.gradtts.text_encoder import TextEncoder
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import DurationPredictor
from espnet.nets.pytorch_backend.transformer.encoder import (
    Encoder as TransformerEncoder,
)
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet.nets.pytorch_backend.fastspeech.length_regulator import LengthRegulator
from espnet2.tts.gradtts.denoiser import Diffusion
from espnet2.torch_utils.device_funcs import force_gatherable

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
            #encoder parameters
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
            #reduction_factor: int = 1,
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
        #self.use_scaled_pos_enc = use_scaled_pos_enc
        #self.reduction_factor = reduction_factor

        # get positional encoding class
        # pos_enc_class = (
        #     ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding
        # )

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        self.encoder = TextEncoder(idim, odim, n_enc_channels,
                                    filter_channels, filter_channels_dp, n_heads,
                                    n_enc_layers, enc_kernel, enc_dropout, window_size)
        self.decoder = Diffusion(odim, dec_dim, n_spks, spk_emb_dim, beta_min, beta_max, pe_scale)

        # # define duration predictor
        # self.duration_predictor = DurationPredictor(
        #     idim=idim,
        #     n_layers=n_enc_layers,
        #     n_chans=filter_channels,
        #     kernel_size=enc_kernel,
        #     dropout_rate=enc_dropout,
        # )
        #
        # # define encoder
        # encoder_input_layer = torch.nn.Embedding(
        #     num_embeddings=idim, embedding_dim=adim, padding_idx=self.padding_idx
        # )
        # if encoder_type == "transformer":
        #     self.encoder = TransformerEncoder(
        #         idim=idim,
        #         attention_dim=adim,
        #         attention_heads=aheads,
        #         linear_units=eunits,
        #         num_blocks=elayers,
        #         input_layer=encoder_input_layer,
        #         dropout_rate=transformer_enc_dropout_rate,
        #         positional_dropout_rate=transformer_enc_positional_dropout_rate,
        #         attention_dropout_rate=transformer_enc_attn_dropout_rate,
        #         pos_enc_class=pos_enc_class,
        #         normalize_before=encoder_normalize_before,
        #         concat_after=encoder_concat_after,
        #         positionwise_layer_type=positionwise_layer_type,
        #         positionwise_conv_kernel_size=positionwise_conv_kernel_size,
        #     )

        # # define length regulator
        # self.length_regulator = LengthRegulator()
        #
        # # define decoder
        # if decoder_type == "diffusion":
        #     self.decoder = Diffusion(
        #         odim, dec_dim, n_spks, spk_emb_dim, beta_min, beta_max, pe_scale
        #     )
        # else:
        #     raise NotImplementedError(decoder_type)
        #
        # # define final projection
        # if decoder_type != "diffusion":
        #     self.feat_out = torch.nn.Linear(adim, odim * reduction_factor)
        # if reduction_factor > 1:
        #     raise NotImplementedError()

        # define postnet
        # self.postnet = (
        #     None
        #     if postnet_layers == 0
        #     else Postnet(
        #         idim=idim,
        #         odim=odim,
        #         n_layers=postnet_layers,
        #         n_chans=postnet_chans,
        #         n_filts=postnet_filts,
        #         use_batch_norm=use_batch_norm,
        #         dropout_rate=postnet_dropout_rate,
        #     )
        # )
        #
        # # initialize parameters
        # self._reset_parameters(
        #     init_type=init_type,
        #     init_enc_alpha=init_enc_alpha,
        #     init_dec_alpha=init_dec_alpha,
        # )

        # define criterions
        # self.criterion = ProDiffLoss(
        #     use_masking=use_masking, use_weighted_masking=use_weighted_masking
        # )

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
        **kwargs,
    ):
        batch_size = text.size(0)
        # text = text[:, : text_lengths.max()]  # for data-parallel
        # feats = feats[:, : feats_lengths.max()]  # for data-parallel
        #x = self.emb(text) * math.sqrt(self.n_channels)
        #x = torch.transpose(x, 1, -1)


        encoder_outputs, decoder_outputs, attn = self._forward(
            text,
            text_lengths,
            feats,
            feats_lengths,
            **kwargs,
        )

        # calculate loss
        dur_loss, prior_loss, diff_loss = self.compute_loss(
            x = text,
            x_lengths = text_lengths,
            y = feats,
            y_lengths = feats_lengths,
            out_size = fix_len_compatibility(2*22050//256),
            #x, x_lengths, y, y_lengths=self.relocate_input([x, x_lengths, y, y_lengths])
        )
        loss = dur_loss + prior_loss + diff_loss

        stats = dict(
            dur_loss=dur_loss.item(),
            prior_loss=prior_loss.item(),
            diff_loss=diff_loss.item(),
        )

        # report extra information
        # if self.encoder_type == "transformer" and self.use_scaled_pos_enc:
        #     stats.update(
        #         encoder_alpha=self.encoder.embed[-1].alpha.data.item(),
        #     )
        # if self.decoder_type == "transformer" and self.use_scaled_pos_enc:
        #     stats.update(
        #         decoder_alpha=self.decoder.embed[-1].alpha.data.item(),
        #     )

        if not joint_training:
            stats.update(loss=loss.item())
            loss, stats, weight = force_gatherable(
                (loss, stats, batch_size), loss.device
            )
            return loss, stats, weight
        # else:
        #     return loss, stats, after_outs if after_outs is not None else before_outs


    @torch.no_grad()
    def _forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        n_timesteps=100,
        temperature=1.0,
        stoc=False,
        spk=None,
        length_scale=1.0
    ):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        x = self.emb(text) * math.sqrt(self.n_channels)
        x = torch.transpose(x, 1, -1)
        x, x_lengths = self.relocate_input([x, text_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

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

        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]

    def compute_loss(self, x, x_lengths, y, y_lengths, spk=None, out_size=None):
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
        x, x_lengths, y, y_lengths = self.relocate_input([x, x_lengths, y, y_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad():
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
            attn = attn.detach()

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor([
                torch.tensor(random.choice(range(start, end)) if end > start else 0)
                for start, end in offset_ranges
            ]).to(y_lengths)

            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)
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

        # Compute loss of score-based decoder
        diff_loss, xt = self.decoder.compute_loss(y, y_mask, mu_y, spk)

        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)

        return dur_loss, prior_loss, diff_loss

    def inference(
            self,
    ) -> Dict[str, torch.Tensor]:
        generator = GradTTS(149, 1, 64,
                            192, 768,
                            256, 2, 6,
                            3, 0.1, 4,
                            80, 64, 0.05, 20.0, 1000)
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        #generator.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc))
        _ = generator.cuda().eval()


        return generator