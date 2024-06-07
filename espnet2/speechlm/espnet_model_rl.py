#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from copy import deepcopy
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
        algo: str = "dpo",
        beta: float = 0.1,
        ce_loss_weight: float = 0.0,
        rl_loss_weight: float = 1.0,
        length_normalize: bool = True,
        label_smoothing: float = 0.0,
        extract_feats_in_collect_stats: bool = False,
    ):
        super().__init__()

        self.corelm = corelm
        self.algo = algo
        self.beta = beta
        self.ce_loss_weight = ce_loss_weight
        self.rl_loss_weight = rl_loss_weight
        self.length_normalize = length_normalize
        self.label_smoothing = label_smoothing
        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

        # clone the model and freeze all parameters.
        self.reflm = deepcopy(corelm)
        self.reflm.requires_grad_(False)

    def forward(
        self,
        dec_seq: torch.Tensor,
        dec_seq_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        if kwargs.get("enc_seq", None) is not None:
            raise ValueError("Encoder-Decoder speechlm is not supported yet")
        prefix_len = kwargs.get("prefix_len")
        
        # (1) Get and concat examples
        # NOTE(Jinchuan): by default, dec_seq are positive examples
        # while rej_seq are negative examples.
        rej_seq = kwargs.get("rej_seq", None)
        rej_seq_lengths = kwargs.get("rej_seq_lengths", None)
        if rej_seq is None or rej_seq_lengths is None:
            raise ValueError(f"The negative examples are not available")

        n_positive, n_negative = len(dec_seq), len(rej_seq)
        assert n_negative % n_positive == 0, (n_negative, n_positive)

        all_seq = pad_and_concat([dec_seq, rej_seq]) # [B_pos + B_neg, T, nq]
        all_seq_lengths = torch.cat([dec_seq_lengths, rej_seq_lengths], dim=0)
        all_prefix_lengths = torch.cat(
            [
                prefix_len, 
                torch.repeat_interleave(prefix_len, n_negative // n_positive)
            ], dim=0
        ) # prefix lengths is shared among positive and negative examples

        # (2) LM forward
        loss_ce, policy_logits, stats, weight = self.corelm(
            all_seq,
            all_seq_lengths,
            None,
            None,
            all_prefix_lengths,
        )

        _, ref_logits, _, _ = self.reflm(
            all_seq,
            all_seq_lengths,
            None,
            None,
            all_prefix_lengths,
        )

        # (3) RL loss computing. Shift by 1 since <sos> has been removed in lm forward
        loss_rl, stats_rl = self.loss_rl(
            all_seq[:, 1:],
            all_seq_lengths - 1,
            all_prefix_lengths - 1,
            policy_logits,
            ref_logits,
            n_positive,
        )

        loss = loss_ce * self.ce_loss_weight + loss_rl * self.rl_loss_weight
        stats.update(stats_rl)

        loss, stats, weight = force_gatherable((loss, stats, weight), loss.device)
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

        # (1) logp, summed to utterance level
        policy_logp = torch.gather(
            policy_logits.log_softmax(-1), dim=3, index=all_seq.unsqueeze(3)
        ).squeeze(3)
        ref_logp = torch.gather(
            ref_logits.log_softmax(-1), dim=3, index=all_seq.unsqueeze(3)
        ).squeeze(3) # [B_pos + B_neg, T, nq]
        nq = ref_logp.size(-1)

        # (2) mask for target region
        mask = length_mask(all_seq_lengths)
        prefix_mask = length_mask(all_prefix_lengths, maxlen=all_seq_lengths.max())
        mask = (mask * torch.abs(prefix_mask - 1)).unsqueeze(2) # [B_pos + B_neg, T, 1]

        # (3) apply mask and sum up
        policy_logp = (policy_logp * mask).sum(dim=(1, 2))
        ref_logp = (ref_logp * mask).sum(dim=(1, 2)) # [B_pos + B_neg]
        if self.length_normalize:
            policy_logp = policy_logp / (all_prefix_lengths - all_prefix_lengths) / nq
            ref_logp = ref_logp / (all_prefix_lengths - all_prefix_lengths) / nq

        # (4) the exact loss computing
        pos_repeat = (len(policy_logp) - n_positive) // n_positive
        if self.algo in ["dpo"]:
            loss_rl, pos_reward, neg_reward, reward_acc = self.loss_dpo(
                pos_policy_logp=policy_logp[:n_positive].tile(pos_repeat),
                neg_policy_logp=policy_logp[n_positive:],
                pos_ref_logp=ref_logp[:n_positive].tile(pos_repeat),
                neg_ref_logp=ref_logp[n_positive:],
            )
        else:
            raise NotImplementedError

        stats = {
            "pos_reward": pos_reward.detach().mean(),
            "neg_reward": neg_reward.detach().mean(),
            "reward_acc": reward_acc.detach().mean(),
        }
        
        return loss_rl.mean(), stats
    
    def loss_dpo(
        self,
        pos_policy_logp: torch.Tensor,
        neg_policy_logp: torch.Tensor,
        pos_ref_logp: torch.Tensor,
        neg_ref_logp: torch.Tensor,
    ):
        """ Compute exactly the DPO-series loss """
        logits = (pos_policy_logp - neg_policy_logp) - (pos_ref_logp - neg_ref_logp)

        if self.algo == "dpo":
            loss = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                -F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        
        else:
            raise NotImplementedError
        
        pos_reward = pos_policy_logp - pos_ref_logp
        neg_reward = neg_policy_logp - neg_ref_logp
        acc = (pos_reward > neg_reward).float()

        return loss, pos_reward, neg_reward, acc

    def state_dict(self, **kwargs):
        """ Will only reserve the corelm states """
        return {"corelm": self.state_dict(**kwargs)}

    def collect_feats(self, **kwargs):
        raise NotImplementedError

    @property
    def layer_cls(self):
        """All layer class that can be warpped by FSDP"""
        return [
            ResidualAttentionBlock,  # Espnet built-in transformer layer.
        ]


# test code
if __name__ == "__main__":
    from espnet2.speechlm.core_lm.ar_multiscale import MultiScaleLM
    model = ESPnetSpeechLMRLModel(
        corelm=MultiScaleLM(vocab_size=10, nq=2, g_layer=1, l_layer=1)
    ).cuda()

    dec_seq = torch.Tensor([
        [1, 1, 2, 4, 5],
        [7, 7, 7, 8, 9],
    ]).unsqueeze(2).repeat(1, 1, 2).long().cuda()
    dec_seq_lengths = torch.Tensor([4, 5]).long().cuda()
    print(f"dec_seq: ", dec_seq, dec_seq_lengths)

    rej_seq = torch.Tensor([
        [1, 1, 5, 6, 7, 8],
        [7, 7, 7, 3, 4, 9],
    ]).unsqueeze(2).repeat(1, 1, 2).long().cuda()
    rej_seq_lengths = torch.Tensor([5, 6]).long().cuda()
    print(f"rej_seq: ", rej_seq, rej_seq_lengths)
    
    prefix_lengths = torch.Tensor([2, 3]).long().cuda()
    print(f"prefix length: ", prefix_lengths)

    _ = model(
        dec_seq=dec_seq,
        dec_seq_lengths=dec_seq_lengths,
        rej_seq_lengths=rej_seq_lengths,
        rej_seq=rej_seq,
        prefix_len=prefix_lengths,
    )
    print("done")
    

