# The implementation is based on:
# https://github.com/sp-uhh/sgmse
# Licensed under MIT


from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Optional, Tuple

from espnet2.enh.diffusion.abs_diffusion import AbsDiffusion
from espnet2.enh.diffusion.sdes import SDE, OUVESDE, OUVPSDE
from espnet2.enh.layers.dcunet import DCUNet

import torch

from espnet2.train.class_choices import ClassChoices

score_choices = ClassChoices(
    name="score_model",
    classes=dict(dcunet=DCUNet),
    type_check=torch.nn.Module,
    default=None,
)

sde_choices = ClassChoices(
    name="sde",
    classes=dict(
        ouve=OUVESDE,
        ouvp=OUVPSDE,
    ),
    type_check=SDE,
    default="ouve"
)


class ScoreModel(AbsDiffusion):

    def __init__(self, **kwargs):
        super().__init__()

        self.dnn = score_choices.get_class(kwargs['score_model'])(**kwargs['score_model_conf'])
        self.sde = sde_choices.get_class(kwargs['sde'])(**kwargs['sde_conf'])
        self.loss_type = getattr(kwargs, 'loss_type', 'mse')
        self.t_eps = getattr(kwargs, 't_eps', 3e-2)


    def _loss(self, err):
        if self.loss_type == 'mse':
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def forward(
        self,
        feature_ref,
        feature_mix,
    ):
        # feature_ref: B, T, F
        # feature_mix: B, T, F
        x = feature_mix.permute(0, 2, 1).unsqueeze(1)
        y = feature_ref.permute(0, 2, 1).unsqueeze(1)
        
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(x, t, y)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        sigmas = std[:, None, None, None]
        perturbed_data = mean + sigmas * z

        # Concatenate y as an extra channel
        dnn_input = torch.cat([perturbed_data, y], dim=1)

        # the minus is most likely unimportant here - taken from Song's repo
        score = - self.dnn(dnn_input, t)
        err = score * sigmas + z
        loss = self._loss(err)

        return loss

    def enhance(self, input: torch.Tensor):
        raise NotImplementedError
