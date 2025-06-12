#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from copy import deepcopy
from typing import Dict, Mapping, Tuple

import torch
from typeguard import typechecked

from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


class ESPnetSpeechLMDPOModel(AbsESPnetModel):

    @typechecked
    def __init__(
        self,
        corelm: AbsCoreLM,
        criterion,
        beta: float = 0.1,
        extract_feats_in_collect_stats: bool = False,
    ):
        super().__init__()

        self.corelm = corelm
        self.criterion = criterion
        self.beta = beta
        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

        # Currently, always use itself as the reference LM. Freeze it
        self.reflm = deepcopy(corelm)
        for p in self.reflm.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        dec_seq: torch.Tensor,
        loss_mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        # [B, T, nq, 2] -> [2 * B, T, nq]
        dec_seq = torch.cat([dec_seq[..., 0], dec_seq[..., 1]], dim=0)
        loss_mask = torch.cat([loss_mask[..., 0], loss_mask[..., 1]], dim=0)

        # (1) LM forward
        policy_ce_loss, policy_elem_logp, _, _ = self.criterion(
            *self.corelm(dec_seq, loss_mask)
        )

        with torch.no_grad():
            ref_ce_loss, ref_elem_logp, _, _ = self.criterion(
                *self.reflm(dec_seq, loss_mask)
            )

        # (2) DPO
        # neg-log-likelihood -> log-likelihood
        policy_logp = -1 * policy_elem_logp.sum(dim=(1, 2))
        ref_logp = -1 * ref_elem_logp.sum(dim=(1, 2))

        num = len(policy_logp) // 2
        pos_policy_logp = policy_logp[:num]
        neg_policy_logp = policy_logp[num:]
        pos_ref_logp = ref_logp[:num]
        neg_ref_logp = ref_logp[num:]

        loss, stats = self.dpo(
            pos_policy_logp=pos_policy_logp,
            neg_policy_logp=neg_policy_logp,
            pos_ref_logp=pos_ref_logp,
            neg_ref_logp=neg_ref_logp,
        )

        loss, stats, num = force_gatherable((loss, stats, num), loss.device)

        return loss, stats, num

    def dpo(
        self,
        pos_policy_logp: torch.Tensor,
        neg_policy_logp: torch.Tensor,
        pos_ref_logp: torch.Tensor,
        neg_ref_logp: torch.Tensor,
    ):
        logits = (pos_policy_logp - neg_policy_logp) - (pos_ref_logp - neg_ref_logp)
        loss = -torch.nn.functional.logsigmoid(logits * self.beta)

        pos_reward = pos_policy_logp - pos_ref_logp
        neg_reward = neg_policy_logp - neg_ref_logp
        reward_gap = pos_reward - neg_reward
        win_rate = (reward_gap > 0).float().sum() / len(reward_gap)
        loss_rate = (reward_gap < 0).float().sum() / len(reward_gap)
        equal_rate = (reward_gap == 0).float().sum() / len(reward_gap)

        stats = {
            "loss_dpo": loss,
            "pos_reward": pos_reward,
            "neg_reward": neg_reward,
            "reward_gap": reward_gap,
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "equal_rate": equal_rate,
        }
        stats = {k: v.clone().detach().mean() for k, v in stats.items()}

        return loss.mean(), stats

    def collect_feats(self, **kwargs):
        raise NotImplementedError

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Make the reflm the same as corelm"""
        keys = [key for key in state_dict if key.startswith("corelm")]
        for key in keys:
            ref_key = key.replace("corelm.", "reflm.")
            state_dict[ref_key] = state_dict[key]
        super().load_state_dict(state_dict, strict, assign)

    def state_dict(self, **kwargs):
        """Only save the corelm parameters, not reflm"""
        state_dict = super().state_dict(**kwargs)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("reflm.")}
        return state_dict
