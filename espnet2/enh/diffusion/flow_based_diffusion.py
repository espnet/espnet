# Copyright 2024 ...
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""Flow-based (OT-FM) speech enhancement.

Based on score_based_diffusion.py (SGMSE+), replacing:
  - SDE perturbation -> OT-FM linear interpolant
  - PC sampler       -> Euler ODE solver
  - score target     -> velocity field target

YAML:
    diffusion_model: fgmse
    diffusion_model_conf:
        score_model: ncsnpp
        score_model_conf: {}
        loss_type: mse
        sigma: 1.0e-4
        N: 30
        t_eps: 0.03
"""

import torch

from espnet2.enh.diffusion.abs_diffusion import AbsDiffusion


def _build_backbone(score_model: str, score_model_conf: dict):
    if score_model == "ncsnpp":
        from espnet2.enh.layers.ncsnpp import NCSNpp

        return NCSNpp(**score_model_conf)
    raise ValueError(f"Unknown score_model '{score_model}'. Supported: 'ncsnpp'.")


class FlowModel(AbsDiffusion):
    """OT-Flow Matching speech enhancement (FGMSE).

    Forward process (OT interpolant):
        x_t = t * x + (1 - (1-sigma) * t) * z,  z ~ N(0,I)
        u   = x - (1-sigma) * z                  (target velocity)

    Reverse process (Euler ODE):
        dx/dt = v_theta(x_t, t, y),  t: t_eps -> 1

    Args:
        score_model: backbone name ('ncsnpp').
        score_model_conf: kwargs for the backbone.
        loss_type: 'mse' or 'mae'.
        sigma: OT interpolant noise width (default 1e-4).
        N: Euler ODE steps at inference.
        t_eps: minimum training time step (avoids t=0 NaN in NCSNpp).
    """

    def __init__(
        self,
        score_model: str = "ncsnpp",
        score_model_conf: dict = None,
        loss_type: str = "mse",
        sigma: float = 1e-4,
        N: int = 30,
        t_eps: float = 0.03,
        **kwargs,
    ):
        super().__init__()
        if score_model_conf is None:
            score_model_conf = {}
        self.dnn = _build_backbone(score_model, score_model_conf)
        self.loss_type = loss_type
        self.sigma = sigma
        self.N = N
        self.t_eps = t_eps

    def _loss(self, err: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            losses = torch.square(err.abs())
        elif self.loss_type == "mae":
            losses = err.abs()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        return torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

    def velocity_fn(self, x_t, t, y):
        """v_theta(x_t, t | y): concatenate x_t and y as ScoreModel.score_fn does."""
        return self.dnn(torch.cat([x_t, y], dim=1), t)

    def forward(self, feature_ref, feature_mix):
        """OT-FM loss. Inputs are [B, T, F] from STFTEncoder."""
        # [B,T,F] -> [B,1,F,T]
        x = feature_ref.permute(0, 2, 1).unsqueeze(1)
        y = feature_mix.permute(0, 2, 1).unsqueeze(1)

        # t ~ U[t_eps, 1], shape [B]
        t = torch.rand(x.shape[0], device=x.device) * (1.0 - self.t_eps) + self.t_eps

        z = torch.randn_like(x)
        t_4d = t[:, None, None, None]

        x_t = x * t_4d + (1.0 - (1.0 - self.sigma) * t_4d) * z
        u = x - (1.0 - self.sigma) * z

        return self._loss(u - self.velocity_fn(x_t, t, y))

    def enhance(self, noisy_spectrum, N=None, **kwargs):
        """Euler ODE inference. Input/output: [B, T, F]."""
        N = N if N is not None else self.N
        # [B,T,F] -> [B,1,F,T]
        Y = noisy_spectrum.permute(0, 2, 1).unsqueeze(1)

        with torch.no_grad():
            x_t = torch.randn_like(Y)
            # start from t_eps
            t_span = torch.linspace(self.t_eps, 1.0, N + 1, device=Y.device)
            x_hat = self._ode_solver(x_t, t_span, Y)

        return x_hat.squeeze(1).permute(0, 2, 1)

    def _ode_solver(self, x_t, t_span, Y):
        """First-order Euler integration of dx/dt = v_theta(x_t, t, Y)."""
        t = t_span[0]
        dt = t_span[1] - t_span[0]
        for step in range(1, len(t_span)):
            t_vec = t.unsqueeze(0).expand(x_t.shape[0])
            x_t = x_t + dt * self.velocity_fn(x_t, t_vec, Y)
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return x_t
