#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from typing import Dict, Mapping, Tuple

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM
from espnet2.speechlm.module.transformer import ResidualAttentionBlock
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.speechlm.net_utils import length_mask, pad_and_concat


class ESPnetSpeechLMRLModel(AbsESPnetModel):

    @typechecked
    def __init__(
        self,
        corelm: AbsCoreLM,
        reflm: AbsCoreLM,
        algo: str = "dpo",
        beta: float = 0.1,
        ce_loss_weight: float = 0.0,
        rl_loss_weight: float = 1.0,
        length_norm: bool = False,
        reward_margin: float = 0.0,
        extract_feats_in_collect_stats: bool = False,
    ):
        super().__init__()

        self.corelm = corelm
        self.algo = algo
        self.beta = beta
        self.ce_loss_weight = ce_loss_weight
        self.rl_loss_weight = rl_loss_weight
        self.length_norm = length_norm
        self.reward_margin = reward_margin
        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

        if algo in ["simpo"]:
            del reflm
            self.reflm = None
        else:
            self.reflm = reflm

        if algo in ["simpo"]:
            assert self.length_norm, f"Algo {algo} requires length_normalize"

    def forward(
        self,
        dec_seq: torch.Tensor,
        dec_seq_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        if kwargs.get("enc_seq", None) is not None:
            raise ValueError("Encoder-Decoder speechlm is not supported yet")
        prefix_len = kwargs.get("prefix_len").squeeze(1)

        # (1) Get and concat examples
        # NOTE(Jinchuan): by default, dec_seq are positive examples
        # while rej_seq are negative examples.
        rej_seq = kwargs.get("sampled_seq", None)
        rej_seq_lengths = kwargs.get("sampled_seq_lengths", None)
        if rej_seq is None or rej_seq_lengths is None:
            raise ValueError(f"The negative examples are not available")

        n_positive, n_negative = len(dec_seq), len(rej_seq)
        assert n_negative % n_positive == 0, (n_negative, n_positive)

        all_seq = pad_and_concat([dec_seq, rej_seq])  # [B_pos + B_neg, T, nq]
        all_seq_lengths = torch.cat([dec_seq_lengths, rej_seq_lengths], dim=0)
        all_prefix_lengths = torch.cat(
            [prefix_len, torch.repeat_interleave(prefix_len, n_negative // n_positive)],
            dim=0,
        )  # prefix lengths is shared among positive and negative examples

        # (2) LM forward
        _, policy_logits, stats, _ = self.corelm(
            all_seq,
            all_seq_lengths,
            None,
            None,
            all_prefix_lengths,
            compute_loss=False,
        )
        stats["loss_ce"] = stats.pop("loss")

        if self.reflm is not None:
            with torch.no_grad():
                _, ref_logits, _, _ = self.reflm(
                    all_seq,
                    all_seq_lengths,
                    None,
                    None,
                    all_prefix_lengths,
                    compute_loss=False,
                )
        else:
            ref_logits = None

        # (3) RL loss computing. Shift by 1 since <sos> has been removed in lm forward
        loss_rl, stats_rl = self.loss_rl(
            all_seq[:, 1:],
            all_seq_lengths - 1,
            all_prefix_lengths - 1,
            policy_logits,
            ref_logits,
            n_positive,
        )
        loss = loss_rl

        stats.update(stats_rl)
        stats.update({"loss": loss.detach()})

        loss, stats, weight = force_gatherable((loss, stats, len(dec_seq)), loss.device)

        return loss, stats, weight

    def loss_rl(
        self,
        all_seq: torch.Tensor,
        all_seq_lengths: torch.Tensor,
        all_prefix_lengths: torch.Tensor,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        n_positive: int,
    ):
        """
        Compute Reinforcement Learning loss

        all_seq (torch.Tensor): the concat tokens of positive and negative sequences.
            (B_pos + B_neg, T, nq)
        all_seq_lengths (torch.Tensor): the concat length tensor of positive and negative
            sequences. (B_pos + B_neg)
        all_prefix_lengths (torch.Tensor): prefix length of each positive sequence.
            (B_pos + B_neg)
        policy_logits (torch.Tensor): the concat policy logits of positive and negative
            sequences. (B_pos + B_neg, T, nq, V)
        ref_logits (torch.Tensor): the concat reference logits of positive and negative
            sequences. (B_pos + B_neg, T, nq, V)
        n_positive (int): number of positive examples, a.k.a., B_pos

        """
        # (1) mask for target region
        mask = length_mask(all_seq_lengths)
        prefix_mask = length_mask(all_prefix_lengths, maxlen=all_seq_lengths.max())
        mask = (mask * torch.abs(prefix_mask - 1)).unsqueeze(2)  # [B_pos + B_neg, T, 1]

        # (2) logp, summed to utterance level
        policy_logp = torch.gather(
            policy_logits.log_softmax(-1), dim=3, index=all_seq.unsqueeze(3)
        ).squeeze(3)
        policy_logp = (policy_logp * mask).sum(dim=(1, 2))

        if ref_logits is not None:
            ref_logp = torch.gather(
                ref_logits.log_softmax(-1), dim=3, index=all_seq.unsqueeze(3)
            ).squeeze(
                3
            )  # [B_pos + B_neg, T, nq]
            ref_logp = (ref_logp * mask).sum(dim=(1, 2))  # [B_pos + B_neg]
        else:
            ref_logp = torch.zeros_like(policy_logp)

        # (3) length normalize
        if self.length_norm:
            nq = all_seq.size(-1)
            policy_logp = policy_logp / (all_seq_lengths - all_prefix_lengths) / nq
            ref_logp = ref_logp / (all_seq_lengths - all_prefix_lengths) / nq

        # (4) the exact loss computing
        pos_repeat = (len(policy_logp) - n_positive) // n_positive
        if pos_repeat == 1:
            loss_rl, stats = self.compute_loss(
                pos_policy_logp=policy_logp[:n_positive].tile(pos_repeat),
                neg_policy_logp=policy_logp[n_positive:],
                pos_ref_logp=ref_logp[:n_positive].tile(pos_repeat),
                neg_ref_logp=ref_logp[n_positive:],
            )
        else:
            # TODO(Jinchuan): Update with Plackett-Luce ranking model.
            raise NotImplementedError

        return loss_rl.mean(), stats

    def compute_loss(
        self,
        pos_policy_logp: torch.Tensor,
        neg_policy_logp: torch.Tensor,
        pos_ref_logp: torch.Tensor,
        neg_ref_logp: torch.Tensor,
    ):
        """Compute exactly the DPO-series loss"""
        logits = (pos_policy_logp - neg_policy_logp) - (pos_ref_logp - neg_ref_logp)

        if self.algo == "dpo":
            loss = -F.logsigmoid(logits * self.beta)

        elif self.algo == "simpo":
            loss = -F.logsigmoid(logits * self.beta - self.reward_margin)

        else:
            raise NotImplementedError(f"{self.algo} is not supported yet")

        pos_reward = pos_policy_logp - pos_ref_logp
        neg_reward = neg_policy_logp - neg_ref_logp
        acc = (pos_reward > neg_reward).float()

        stats = {
            "loss_rl": loss,
            "pos_reward": pos_reward,
            "neg_reward": neg_reward,
            "reward_gap": pos_reward - neg_reward,
            "reward_acc": acc,
            "pos_policy_logp": pos_policy_logp,
            "neg_policy_logp": neg_policy_logp,
            "pos_ref_logp": pos_ref_logp,
            "neg_ref_logp": neg_ref_logp,
        }
        stats = {k: v.detach().mean() for k, v in stats.items()}

        return loss, stats

    def collect_feats(self, **kwargs):
        raise NotImplementedError

    @property
    def layer_cls(self):
        """All layer class that can be warpped by FSDP"""
        return [
            ResidualAttentionBlock,  # Espnet built-in transformer layer.
        ]


def printf(*string):
    if torch.distributed.get_rank() == 0:
        print(*string, flush=True)
