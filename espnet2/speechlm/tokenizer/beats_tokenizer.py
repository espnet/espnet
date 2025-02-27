# --------------------------------------------------------
# BEATs: Audio Pre-Training with Acoustic Tokenizers (https://arxiv.org/abs/2212.09058)
# Github source: https://github.com/microsoft/unilm/tree/master/beats
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# Based on VQGAN code bases
# https://github.com/CompVis/taming-transformers
# This code is adapted from the original BEATs implementation and
#  can be used to tokenize audio using a BEATs model.
# --------------------------------------------------------

import logging
from typing import Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as ta_kaldi
from packaging.version import parse as V
from torch.nn import LayerNorm
import torch.distributed as dist

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


from espnet2.asr.encoder.beats_encoder import (
    BeatsConfig,
    BeatsEncoder,
    TransformerSentenceEncoderLayer,
    init_bert_params,
)
from espnet2.speechlm.tokenizer.beats_utils import (
    l2norm,
    kmeans,
    norm_ema_inplace,
    ema_inplace,
    forward_padding_mask_conv,
    freeze_conv_module,
    beats_frontend,
)
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.speechlm.tokenizer.random_tokenizer import RandomProjectionQuantizer


class BeatsTokenizerConfig(BeatsConfig):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        # quantizer
        self.quant_n: int = 1024  # number of vec in quantizer coebook
        self.quant_dim: int = 256  # codebook dimension in quantizer
        self.embed_loss_beta: float = 1.0  # e15  # beta for embedding loss

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)


class BeatsTokenizer(BeatsEncoder):
    def __init__(
        self,
        beats_tokenizer_ckpt_path: str = None,
        tokenizer_config: Optional[Dict] = None,
        max_layer: int = None,
        use_weighted_representation: bool = False,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
    ) -> None:
        if beats_tokenizer_ckpt_path is None:
            logging.info(
                "No checkpoint path provided for BEATs tokenizer encoder!"
                "We will encode audio using a randomly initialized model."
            )
        super().__init__(
            input_size=1,
            beats_ckpt_path=beats_tokenizer_ckpt_path,
            max_layer=max_layer,
            use_weighted_representation=use_weighted_representation,
            beats_config=tokenizer_config,
            fbank_mean=fbank_mean,
            fbank_std=fbank_std,
        )

        config = BeatsTokenizerConfig()  # default config
        if self.loaded_state_dict_ is not None:
            config = BeatsTokenizerConfig(self.loaded_state_dict_["cfg"])
        if tokenizer_config:
            config.update(tokenizer_config)

        self.quantize = NormEMAVectorQuantizer(
            n_embed=config.quant_n,
            embedding_dim=config.quant_dim,
            beta=config.embed_loss_beta,
            decay=0.99,
            kmeans_init=True,
        )
        self.quantize_layer = nn.Sequential(
            nn.Linear(
                self._output_size,
                self._output_size,
            ),
            nn.Tanh(),
            nn.Linear(self._output_size, config.quant_dim),
        )
        self.initialize_tokenizer_params()

    def initialize_tokenizer_params(self):
        logging.info("Beats Tokenizer initialization function called.")
        init_bert_params(self.quantize_layer)
        self.reload_pretrained_parameters()

    def reload_pretrained_parameters(self):
        super().reload_pretrained_parameters()
        logging.info("Beats Tokenizer reload pretrained params function called.")
        if self.loaded_state_dict_ is not None:
            load_info = self.load_state_dict(
                self.loaded_state_dict_["model"], strict=False
            )
            logging.info(
                f"Loaded Beats tokenizer pretrained model."
                f"Following keys were missing"
                f" in your custom model: {load_info.missing_keys}. "
                f"Follwing keys could not be loaded from the pretrained"
                f"checkpoint: {load_info.unexpected_keys}."
            )

    def encode(
        self,
        xs_pad: torch.Tensor,
        ilens: Optional[torch.Tensor] = None,
        waveform_input: bool = True,
    ):
        """
        Encodes input audio xs_pad to quantized features.
        Args:
            xs_pad (torch.Tensor): Input tensor (B, T, D) or (B,T,1).
            ilens (torch.Tensor): Input length tensor (B,).
            waveform_input (bool): If True, input is raw waveform.
        Returns:
            embed_ind (torch.Tensor): Embedding indices (B, T).
            embed_loss (torch.Tensor): Embedding loss.
            quantize_feature (torch.Tensor): Quantized features.
        """
        x, x_len, _ = self(xs_pad, ilens, waveform_input=waveform_input)
        quantize_input = self.quantize_layer(x)
        quantize_feature, embed_loss, embed_ind = self.quantize(quantize_input)
        return {
            "codes": embed_ind,
            "code_lengths": x_len,
            "embed_loss": embed_loss,
            "quantize_feature": quantize_feature,
        }


class NormEMAVectorQuantizer(nn.Module):
    def __init__(
        self,
        n_embed,
        embedding_dim,
        beta,
        decay=0.99,
        kmeans_init=False,
        eps=1e-5,
        statistic_code_usage=True,
    ):
        super().__init__()
        self.num_tokens = n_embed
        self.codebook_dim = embedding_dim
        self.beta = beta
        self.decay = decay
        self.embedding = EmbeddingEMA(
            self.num_tokens, self.codebook_dim, decay, eps, kmeans_init
        )
        self.statistic_code_usage = statistic_code_usage
        if statistic_code_usage:
            self.register_buffer("cluster_size", torch.zeros(n_embed))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            logging.info(
                "ddp is enabled, so use ddp_reduce to sync the cluster_size for each gpu!"
            )
            self.all_reduce_fn = torch.distributed.all_reduce
        else:
            self.all_reduce_fn = nn.Identity()

    def reset_cluster_size(self, device):
        if self.statistic_code_usage:
            self.register_buffer("cluster_size", torch.zeros(self.num_tokens))
            self.cluster_size = self.cluster_size.to(device)

    def forward(self, z):
        """Encode the input with the vector quantizer.
        Args
            z: (B, T, D) input tensor
        Returns
            z_q: (B, T, D) quantized tensor
            loss: scalar quantization loss
            encoding_indices: (B, T) indices of the quantized embeddings
        """
        z = l2norm(z)
        z_flattened = z.reshape(-1, self.codebook_dim)  # B*T,D

        if self.training:
            self.embedding.init_embed_(z_flattened)

        d = (
            z_flattened.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * torch.einsum("bd,nd->bn", z_flattened, self.embedding.weight)
        )
        encoding_indices = torch.argmin(d, dim=1)  # B*T
        z_q = self.embedding(encoding_indices).view(z.shape)  # B,T,D
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(
            z.dtype
        )  # B*T,num_tokens
        encoding_indices = encoding_indices.view(z.shape[:-1])  # B,T

        if not self.training:
            # just update the codebook statistics
            # logging.info("Updating codebook statistics...")
            with torch.no_grad():
                cluster_size = encodings.sum(0)
                self.all_reduce_fn(cluster_size)
                ema_inplace(self.cluster_size, cluster_size, self.decay)

        if self.training:
            # EMA cluster size
            bins = encodings.sum(0)  # num_tokens
            self.all_reduce_fn(bins)
            ema_inplace(self.cluster_size, bins, self.decay)

            zero_mask = bins == 0
            bins = bins.masked_fill(zero_mask, 1.0)
            # [D, B*T] @ [B*T,num_tokens] --> [D, num_tokens]
            embed_sum = z_flattened.t() @ encodings
            self.all_reduce_fn(embed_sum)
            # num_tokens, D
            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = l2norm(embed_normalized)
            embed_normalized = torch.where(
                zero_mask[..., None], self.embedding.weight, embed_normalized
            )
            norm_ema_inplace(self.embedding.weight, embed_normalized, self.decay)
            # logging.info("Codebook statistics updated in training mode!")

        # compute loss for embedding
        loss = self.beta * F.mse_loss(z_q.detach(), z)
        # copy gradient from z_q to z
        z_q = z + (z_q - z).detach()
        return z_q, loss, encoding_indices


class EmbeddingEMA(nn.Module):
    # Not learnt via backpropagation.
    def __init__(
        self,
        num_tokens,
        codebook_dim,
        decay=0.99,
        eps=1e-5,
        kmeans_init=True,
    ):
        # NOTE: removed option to load weights from a supplied codebook.
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.eps = eps

        assert kmeans_init, "Only kmeans init is supported for now."
        # if not kmeans_init:
        #     weight = torch.randn(num_tokens, codebook_dim)
        #     weight = l2norm(weight)
        #     self.register_buffer("initted", torch.Tensor([True]))
        # else:
        weight = torch.zeros(num_tokens, codebook_dim)
        self.register_buffer("initted", torch.Tensor([False]))
        # NOTE: Very important that weight is always normalized.
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)

    # @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return
        logging.info("Performing Kmeans init for codebook")
        # embed[float]: (num_tokens, codebook_dim), cluster_size[int]: (num_tokens,)
        embed, cluster_size = kmeans(
            data, self.num_tokens, num_iters=10, use_cosine_sim=True
        )
        cluster_size = cluster_size.to(self.cluster_size.device)
        embed = embed.to(self.weight.device)
        if dist.is_initialized():
            # NOTE(shikhar): This might not work with deepspeed Zero.
            dist.broadcast(embed, src=0)
            dist.broadcast(cluster_size, src=0)
        self.weight.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]).to(self.initted.device))
        logging.info("Kmeans init done!")

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)


class BeatsTokenizerPretrainingPredictor(nn.Module):

    def __init__(self, tokenizer_config: Optional[Dict] = None):
        super().__init__()
        tokenizer_config = BeatsTokenizerConfig(tokenizer_config)
        self.connector_layer = nn.Linear(
            tokenizer_config.quant_dim, tokenizer_config.decoder_embed_dim
        )
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=tokenizer_config.decoder_embed_dim,
                    ffn_embedding_dim=tokenizer_config.encoder_ffn_embed_dim,
                    num_attention_heads=tokenizer_config.encoder_attention_heads,
                    dropout=tokenizer_config.dropout,
                    attention_dropout=tokenizer_config.attention_dropout,
                    activation_dropout=tokenizer_config.activation_dropout,
                    activation_fn=tokenizer_config.activation_fn,
                    layer_norm_first=tokenizer_config.layer_norm_first,
                    deep_norm=tokenizer_config.deep_norm,
                    has_relative_attention_bias=tokenizer_config.relative_position_embedding,
                    num_buckets=tokenizer_config.num_buckets,
                    max_distance=tokenizer_config.max_distance,
                    gru_rel_pos=tokenizer_config.gru_rel_pos,
                    encoder_layers=tokenizer_config.decoder_layers,
                    use_flash_attn=tokenizer_config.use_flash_attn,
                )
                for i in range(tokenizer_config.decoder_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(tokenizer_config.decoder_embed_dim)
        if tokenizer_config.relative_position_embedding:
            for i in range(1, tokenizer_config.decoder_layers):
                del self.decoder_blocks[i].self_attn.relative_attention_bias
                self.decoder_blocks[i].self_attn.relative_attention_bias = (
                    self.decoder_blocks[0].self_attn.relative_attention_bias
                )
        self.initialize(tokenizer_config)

    def initialize(self, config):
        self.apply(init_bert_params)
        if config.deep_norm:
            logging.info("Deep Norm is applied to pretraining predictor.")
            for i in range(config.decoder_layers):
                deep_norm_beta = math.pow(8 * config.decoder_layers, -1 / 4)
                nn.init.xavier_normal_(
                    self.decoder_blocks[i].self_attn.k_proj.weight, gain=1
                )
                nn.init.xavier_normal_(
                    self.decoder_blocks[i].self_attn.v_proj.weight, gain=deep_norm_beta
                )
                nn.init.xavier_normal_(
                    self.decoder_blocks[i].self_attn.q_proj.weight, gain=1
                )
                nn.init.xavier_normal_(
                    self.decoder_blocks[i].self_attn.out_proj.weight,
                    gain=deep_norm_beta,
                )
                nn.init.xavier_normal_(
                    self.decoder_blocks[i].fc1.weight, gain=deep_norm_beta
                )
                nn.init.xavier_normal_(
                    self.decoder_blocks[i].fc2.weight, gain=deep_norm_beta
                )

    def forward(self, quantize_feature, quantize_feats_len):
        quantize_feature = quantize_feature[:, : quantize_feats_len.max(), :]
        padding_mask = make_pad_mask(
            lengths=quantize_feats_len,
            traceable=False,
        ).to(quantize_feature.device)
        x = self.connector_layer(quantize_feature)

        pos_bias = None
        # ===========================#
        # B x T x D -> T x B x D
        x = x.transpose(0, 1)
        for blk in self.decoder_blocks:
            x, _, pos_bias = blk(
                x,
                self_attn_padding_mask=padding_mask,
                need_weights=False,
                pos_bias=pos_bias,
            )
        x = x.transpose(0, 1)
        # T x B x D -> B x T x D
        # ===========================#
        pred = self.decoder_norm(x)
        return pred


class BeatsRandomTokenizer(nn.Module):
    # Note: Note tested with espnet trainer. Only use with dump_audio_tokens.py
    def __init__(
        self,
        tokenizer_config: Optional[Dict] = None,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
    ) -> None:
        super().__init__()
        self.fbank_mean = fbank_mean
        self.fbank_std = fbank_std

        config = BeatsTokenizerConfig(tokenizer_config)
        self.config = config
        self.layer_norm = LayerNorm(config.embed_dim, elementwise_affine=False)
        self.patch_embedding = nn.Conv2d(
            1,
            config.embed_dim,
            kernel_size=config.input_patch_size,
            stride=config.input_patch_size,
            bias=config.conv_bias,
        )
        self.patch_embedding_pad = nn.Conv2d(
            1,
            1,
            kernel_size=config.input_patch_size,
            stride=config.input_patch_size,
            bias=False,
        )
        self.raw2fbank_pad = nn.Conv1d(
            1,
            1,
            kernel_size=400,
            stride=160,
            bias=False,
        )
        seed = config.seed
        self.random_projection_quantizer = RandomProjectionQuantizer(
            config.embed_dim,
            codebook_size=config.quant_n,
            codebook_dim=config.quant_dim,
            seed=seed,
        )
        self._initialize(seed)

    def _initialize(self, seed):
        logging.info(
            f"Beats Random Tokenizer initialization function called with seed {seed}."
        )
        original_rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
        torch.nn.init.xavier_normal_(self.patch_embedding.weight)
        torch.set_rng_state(original_rng_state)
        self.patch_embedding.weight.requires_grad = False
        if self.patch_embedding.bias is not None:
            torch.nn.init.constant_(self.patch_embedding.bias, 0)
            self.patch_embedding.bias.requires_grad = False
        freeze_conv_module(self.patch_embedding_pad)
        freeze_conv_module(self.raw2fbank_pad)

    @torch.no_grad()
    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward padding mask. Featuires: BTC, padding_mask: BT."""
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.any(-1)  # remove totally empty sequences
        return padding_mask

    @torch.no_grad()
    def forward(
        self, xs_pad: torch.Tensor, ilens: torch.Tensor, waveform_input: bool = True
    ):
        """
        Args:
            xs_pad (torch.Tensor): Input tensor (B, T) or (B,T,D).
                (B,T) for raw waveform and (B,T,D) for features.
            ilens (torch.Tensor): Input length tensor (B,).
            waveform_input (bool): If True, input is raw waveform.
        """
        xs_pad = xs_pad[:, : ilens.max()]
        if waveform_input:
            assert xs_pad.dim() == 2
        padding_mask = make_pad_mask(lengths=ilens, traceable=False).to(xs_pad.device)
        if waveform_input:
            padding_mask = forward_padding_mask_conv(
                padding_mask=padding_mask, n_dim=0, conv_module=self.raw2fbank_pad
            )
        with autocast(False):
            if waveform_input:
                fbank = beats_frontend(
                    xs_pad, fbank_mean=self.fbank_mean, fbank_std=self.fbank_std
                )
            else:
                fbank = (xs_pad - self.fbank_mean) / (2 * self.fbank_std)
        n_mels = fbank.size(2)
        fbank = fbank.unsqueeze(1).float()
        features = self.patch_embedding(fbank)  # B, C, t, d=8
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        padding_mask = forward_padding_mask_conv(
            padding_mask, n_dim=n_mels, conv_module=self.patch_embedding_pad
        )
        embed_ind = self.random_projection_quantizer(features)
        embed_len = (~padding_mask).sum(1)
        return embed_ind, embed_len

    @torch.no_grad()
    def encode(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        waveform_input: bool = True,
    ):
        embed_ind, embed_len = self(xs_pad, ilens, waveform_input=waveform_input)
        return {"codes": embed_ind, "code_lengths": embed_len}
