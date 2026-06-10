import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.mlm_decoder import MLMDecoder
from espnet2.legacy.nets.pytorch_backend.transformer.layer_norm import LayerNorm


def expand(
    padded_tensor: torch.Tensor,
    segment_ids: torch.Tensor,
    original_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    圧縮されたTensorを元のフレーム長に拡張（復元）します。

    Args:
        padded_tensor: (B, T_compressed, ...) 圧縮されたデータ（トークンや特徴量など）
        segment_ids: (B, T_original) aggregate時に返されたインデックス
        original_lengths: (B,) 元の有効なフレーム長

    Returns:
        expanded_tensor: 元のフレーム長に拡張されたTensor
    """
    B, T_orig = segment_ids.shape
    device = padded_tensor.device

    # 特徴量次元がある場合に対応 (B, T, D)
    if padded_tensor.dim() == 3:
        D = padded_tensor.shape[-1]
        # segment_idsを特徴量次元に合わせて拡張
        expanded_indices = segment_ids.unsqueeze(-1).expand(-1, -1, D)
        expanded_tensor = torch.gather(padded_tensor, 1, expanded_indices)
    else:
        # スカラーやインデックステンソルの場合 (B, T)
        expanded_tensor = torch.gather(padded_tensor, 1, segment_ids)

    # パディング領域を0などで埋めるためのマスク処理
    time_mask = torch.arange(T_orig, device=device)[None, :] < original_lengths[:, None]
    if padded_tensor.dim() == 3:
        time_mask = time_mask.unsqueeze(-1)

    expanded_tensor = expanded_tensor = expanded_tensor * time_mask.type_as(
        expanded_tensor
    )

    return expanded_tensor


def aggregate(ctc_logits):
    """
    連続する同じCTCアライメントを単一のセグメントに集約（圧縮）します。

    Args:
        ctc_logits (torch.Tensor): (B, T, V) フレームレベルのCTCアライメントトークン

    Returns:
        padded_seqs: (B, T_compressed, V) 圧縮されたctc_logits
        lens: (B,) 圧縮後の各シーケンスの有効フレーム長
        seg_ids: (B, T) 【追加】expand時に元のフレーム長に復元するためのインデックス
    """
    align_lps, align_toks = ctc_logits.max(dim=-1)
    bsz, align_len, V = ctc_logits.size()
    device = ctc_logits.device

    # Fastest implementation: vectorized implementation using bi-gram checking and scatter_reduce():
    seg_start_mask = torch.ones_like(align_toks, dtype=torch.bool)
    seg_start_mask[:, 1:] = align_toks[:, 1:] != align_toks[:, :-1]

    # 圧縮後の長さを計算
    lens = seg_start_mask.long().sum(dim=1)
    max_len: int = int(lens.max().item())

    # expand用のインデックス (B, T_orig) を生成（ここで生成したものを返す）
    seg_ids = seg_start_mask.long().cumsum(dim=-1) - 1
    # seg_ids を特徴量次元 V に合わせて拡張
    expanded_seg_ids = seg_ids.unsqueeze(-1).expand(-1, -1, V)

    # 連続するフレーム内で最大の確率を取るため、-infで初期化してscatter_reduce (amax)
    padded_seqs = ctc_logits.new_full((bsz, max_len, V), float("-inf"))
    padded_seqs.scatter_reduce_(
        dim=1, index=expanded_seg_ids, src=ctc_logits, reduce="amax", include_self=False
    )

    # パディング領域を0で埋める
    valid_mask = torch.arange(max_len, device=device)[None, :] < lens[:, None]
    # valid_maskがFalse（パディング部分）の場所を0.0にする
    padded_seqs[~valid_mask] = 0.0

    return padded_seqs, lens, seg_ids


class MaskCTC(CTC):
    """MaskCTC module based on ESPnet CTC."""

    @typechecked
    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        ctc_type: str = "builtin",
        reduce: bool = True,
        ignore_nan_grad: Optional[bool] = None,
        zero_infinity: bool = True,
        brctc_risk_strategy: str = "exp",
        brctc_group_strategy: str = "end",
        brctc_risk_factor: float = 0.0,
        compression_type: Optional[str] = None,
        mask_threshold: float = 0.5,
        blank_id: int = 0,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        mlm_dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = False,
        normalize_before: bool = True,
        concat_after: bool = False,
        use_flash_attn: bool = True,
        confidence_estimation: str = "uniform",  # uniform, gaussian, gumbel, softmax, predict, none
        discrete: bool = False,
        residual: bool = False,
        gate_type: str = "none",  # soft, hard, sum
    ):
        super().__init__(
            odim,
            encoder_output_size,
            dropout_rate,
            ctc_type,
            reduce,
            ignore_nan_grad,
            zero_infinity,
            brctc_risk_strategy,
            brctc_group_strategy,
            brctc_risk_factor,
        )

        # setting for intermediate mask CTC
        self.compression_type = compression_type
        self.black_id = blank_id
        self.num_iter = 1
        self.eos_token = (
            F.one_hot(torch.tensor(odim - 1), num_classes=odim)
            .to(float)
            .scatter_(-1, torch.tensor(odim - 1), float("inf"))
        )

        self.residual = residual
        self.confidence_estimation = confidence_estimation
        if self.confidence_estimation in {"uniform", "gaussian", "predict"}:
            self.mask_threshold = 0.5
        else:
            self.mask_threshold = mask_threshold

        self.discrete = discrete

        if self.residual:
            self.after_norm = LayerNorm(encoder_output_size)

        if self.discrete:
            self.gate_type = "hard"
        else:
            self.gate_type = gate_type

        if self.discrete:
            odim += 1
            self.mask_token = odim - 1
        else:
            # Mask embedding for MLM
            self.mask_token = torch.nn.Parameter(torch.rand(encoder_output_size))

        self.ctc_mlm = MLMDecoder(
            odim,
            encoder_output_size,
            attention_heads=attention_heads,
            linear_units=linear_units,
            num_blocks=num_blocks,
            dropout_rate=mlm_dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            self_attention_dropout_rate=self_attention_dropout_rate,
            src_attention_dropout_rate=src_attention_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            normalize_before=normalize_before,
            concat_after=concat_after,
            use_flash_attn=use_flash_attn,
        )

        if self.confidence_estimation == "predict":
            # masking position is base on linear prediction
            self.confidence_predictor = torch.nn.Linear(encoder_output_size, 1)
        else:
            self.confidence_predictor = None

    def forward(
        self,
        hs_pad,
        hlens=None,
        ys_pad=None,  # Only for mask ctc style padding
        ys_pad_lens=None,  # Only for mask ctc style padding
    ):
        """Calculate CTC loss.
        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
            ys_mask: batch of padded masked character id sequence tensor (B, Lmax)
            ys_mask_lens: batch of lengths of masked character sequence (B)
        """

        if self.residual and len(self.intermediate_outs) > 0:
            # Ensure mlm_hidden exists in the last intermediate_outs dictionary
            last_mlm = self.intermediate_outs[-1].get("mlm_hidden", 0)
            hs_pad = self.after_norm(hs_pad + last_mlm)

        # Compress ctc outputs
        ctc_logits = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        intermediate_out = {"ctc": ctc_logits, "ctc_lens": hlens}

        mask_ctc, mask_ctc_lens, seg_ids = aggregate(ctc_logits)

        for iter in range(self.num_iter):
            # Generate CTC sequence and its confidence
            mask_ctc, confidence = self.estimate_ctc_confidence(mask_ctc)

            # Masking - assuming self.masking is defined elsewhere in the parent class or needs implementation
            # mask_ctc, mask_ctc_lens = self.masking(mask_ctc, mask_ctc_lens)

            # masking
            mask_ctc = self.gating(mask_ctc, confidence, iter)

            # decode
            mask_ctc, _ = self.ctc_mlm(hs_pad, hlens, mask_ctc, mask_ctc_lens)

            # expanding the output to the original length for condition

        mask_ctc = expand(mask_ctc, seg_ids, hlens)
        assert hs_pad.size(1) == mask_ctc.size(1)

        # temporal
        mask_ctc = self.ctc_lo(F.dropout(mask_ctc, p=self.dropout_rate))
        intermediate_out["mlm"] = mask_ctc
        intermediate_out["mlm_lens"] = hlens
        self.intermediate_outs.append(intermediate_out)

        return mask_ctc

    def estimate_ctc_confidence(self, ctc_logits):
        B, L, _ = ctc_logits.size()
        device = ctc_logits.device

        if self.confidence_estimation == "gumbel" and self.training:
            tau = 1.0
            ctc_hat = F.gumbel_softmax(ctc_logits, tau=tau, hard=False, dim=-1)
            confidence = ctc_hat.max(-1)[0].unsqueeze(-1)
        else:
            ctc_hat = torch.softmax(ctc_logits, dim=-1)
            confidence = ctc_hat.max(-1)[0].unsqueeze(-1)

        if self.confidence_estimation == "uniform":
            confidence = torch.rand(1, L, 1, device=device).expand(B, L, 1)
        elif self.confidence_estimation == "gaussian":
            mean, std = 0.5, 0.15
            confidence = torch.normal(
                mean=mean, std=std, size=(1, L, 1), device=device
            ).expand(B, L, 1)
        elif self.confidence_estimation == "predict":
            confidence = torch.sigmoid(self.confidence_predictor(hs_pad))

        return ctc_hat, confidence

    # def gating(self, sequence, confidence, iter):
    #     """Applies masking gate to the sequence based on confidence."""
    #     mask_threshold = self.mask_threshold
    #     if not self.training and self.confidence_estimation =="predict":
    #         mask_threshold = 0.5

    #     if self.discrete:
    #         sequence = sequence.argmax(-1)

    #     if self.gate_type == "hard":
    #         # Straight-Through Estimator (STE) for continuous vectors
    #         # Forward pass: 0 or 1. Backward pass: gradients flow through confidence.
    #         hard_gate = (confidence > mask_threshold).float()
    #         gate = hard_gate.detach() - confidence.detach() + confidence

    #         sequence = (gate.squeeze() * sequence + (1 - gate.squeeze()) * self.mask_token).to(sequence.dtype)

    #     elif self.gate_type == "soft":
    #         sequence = confidence * sequence + (1 - confidence) * self.mask_token

    #     elif self.gate_type == "sum":
    #         sequence = sequence + (1 - confidence) * self.mask_token

    #     if not self.discrete:
    #         weights = self.ctc_mlm.embed[0].weight.T
    #         sequence = self.ctc_mlm.embed(F.linear(sequence, weights))

    #     return sequence

    def gating(self, sequence, confidence, iter):
        """Applies masking gate to the sequence based on confidence."""

        # 1. iterに基づく減衰係数の計算 (例: 指数関数的減衰)
        # self.decay_rate が未定義の場合はデフォルトで 0.9 とします
        # decay_rate = getattr(self, "decay_rate", 0.9)
        decay_factor = 0.8**iter

        # 2. 基本となる閾値の決定
        base_threshold = self.mask_threshold
        if not self.training and self.confidence_estimation == "predict":
            base_threshold = 0.5

        # 動的閾値：iterが増加すると閾値が下がり、マスクされにくくなる
        mask_threshold = base_threshold * decay_factor

        if self.discrete:
            sequence = sequence.argmax(-1)

        if self.gate_type == "hard":
            # Straight-Through Estimator (STE) for continuous vectors
            # Forward pass: 0 or 1. Backward pass: gradients flow through confidence.
            hard_gate = (confidence > mask_threshold).float()
            gate = hard_gate.detach() - confidence.detach() + confidence

            sequence = (
                gate.squeeze() * sequence + (1 - gate.squeeze()) * self.mask_token
            ).to(sequence.dtype)

        elif self.gate_type == "soft":
            # iterが増えるほど、実質的なconfidenceが1.0（マスクなし）に近づくよう補正
            eff_confidence = 1.0 - ((1.0 - confidence) * decay_factor)
            sequence = (
                eff_confidence * sequence + (1 - eff_confidence) * self.mask_token
            )

        elif self.gate_type == "sum":
            # iterが増えるほど、mask_tokenの加算される重みを減らす
            sequence = sequence + (1 - confidence) * decay_factor * self.mask_token

        # if not self.discrete:
        #     weights = self.ctc_mlm.embed[0].weight.T
        #     sequence = self.ctc_mlm.embed(F.linear(sequence, weights))

        return sequence
