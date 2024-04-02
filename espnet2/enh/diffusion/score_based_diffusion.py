# The implementation is based on:
# https://github.com/sp-uhh/sgmse
# Licensed under MIT


import math

import torch

import espnet2.enh.diffusion.sampling as sampling
from espnet2.enh.diffusion.abs_diffusion import AbsDiffusion
from espnet2.enh.diffusion.sdes import OUVESDE, OUVPSDE, SDE
from espnet2.enh.layers.dcunet import DCUNet
from espnet2.enh.layers.ncsnpp import NCSNpp
from espnet2.train.class_choices import ClassChoices

score_choices = ClassChoices(
    name="score_model",
    classes=dict(dcunet=DCUNet, ncsnpp=NCSNpp),
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
    default="ouve",
)


class ScoreModel(AbsDiffusion):
    def __init__(self, **kwargs):
        super().__init__()

        score_model = kwargs["score_model"]  # noqa
        score_model_class = score_choices.get_class(kwargs["score_model"])
        self.dnn = score_model_class(**kwargs["score_model_conf"])
        self.sde = sde_choices.get_class(kwargs["sde"])(**kwargs["sde_conf"])
        self.loss_type = getattr(kwargs, "loss_type", "mse")
        self.t_eps = getattr(kwargs, "t_eps", 3e-2)

    def _loss(self, err):
        if self.loss_type == "mse":
            losses = torch.square(err.abs())
        elif self.loss_type == "mae":
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position
        # and mean over batch dim presumably only important for absolute
        # loss number, not for gradients
        loss = torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def get_pc_sampler(
        self, predictor_name, corrector_name, y, N=None, minibatch=None, **kwargs
    ):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(
                predictor_name,
                corrector_name,
                sde=sde,
                score_fn=self.score_fn,
                y=y,
                **kwargs
            )
        else:
            M = y.shape[0]

            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(math.ceil(M / minibatch))):
                    y_mini = y[i * minibatch : (i + 1) * minibatch]
                    sampler = sampling.get_pc_sampler(
                        predictor_name,
                        corrector_name,
                        sde=sde,
                        score_fn=self.score_fn,
                        y=y_mini,
                        **kwargs
                    )
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns

            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(
                sde, self.score_fn, y=y, device=y.device, **kwargs
            )
        else:
            M = y.shape[0]

            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(math.ceil(M / minibatch))):
                    y_mini = y[i * minibatch : (i + 1) * minibatch]
                    sampler = sampling.get_ode_sampler(
                        sde, self.score_fn, y=y_mini, device=y_mini.device, **kwargs
                    )
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return sample, ns

            return batched_sampling_fn

    def score_fn(self, x, t, y):
        # Concatenate y as an extra channel
        dnn_input = torch.cat([x, y], dim=1)

        # the minus is most likely unimportant here - taken from Song's repo
        score = -self.dnn(dnn_input, t)
        return score

    def forward(
        self,
        feature_ref,
        feature_mix,
    ):
        # feature_ref: B, T, F
        # feature_mix: B, T, F
        x = feature_ref.permute(0, 2, 1).unsqueeze(1)
        y = feature_mix.permute(0, 2, 1).unsqueeze(1)

        t = (
            torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps)
            + self.t_eps
        )
        mean, std = self.sde.marginal_prob(x, t, y)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        sigmas = std[:, None, None, None]
        perturbed_data = mean + sigmas * z

        score = self.score_fn(perturbed_data, t, y)
        assert score.shape == x.shape, "Check the output shape of the score_fn."
        err = score * sigmas + z
        loss = self._loss(err)

        return loss

    def enhance(
        self,
        noisy_specturm,
        sampler_type="pc",
        predictor="reverse_diffusion",
        corrector="ald",
        N=30,
        corrector_steps=1,
        snr=0.5,
        **kwargs
    ):
        """Enhance function.

        Args:
            noisy_specturm (torch.Tensor): noisy feature in [Batch, T, F]
            sampler_type (str): sampler, 'pc' for Predictor-Corrector and 'ode' for ODE
                                sampler.
            predictor (str): the name of Predictor. 'reverse_diffusion',
                            'euler_maruyama', or 'none'
            corrector (str): the name of Corrector. 'langevin', 'ald' or 'none'
            N (int): The number of reverse sampling steps.
            corrector_steps (int) : number of steps in the Corrector.
            snr (float): The SNR to use for the corrector.
        Returns:
            X_Hat (torch.Tensor): enhanced feature in [Batch, T, F]
        """
        Y = noisy_specturm.permute(0, 2, 1).unsqueeze(1)
        if sampler_type == "pc":
            sampler = self.get_pc_sampler(
                predictor,
                corrector,
                Y,
                N=N,
                corrector_steps=corrector_steps,
                snr=snr,
                intermediate=False,
                **kwargs
            )
        elif sampler_type == "ode":
            sampler = self.get_ode_sampler(Y, N=N, **kwargs)
        else:
            print("{} is not a valid sampler type!".format(sampler_type))

        X_Hat, nfe = sampler()

        X_Hat = X_Hat.squeeze(1).permute(0, 2, 1)

        return X_Hat
