"""Self-contained fixed-step ODE solvers for F5-TTS sampling.

F5's CFM sampler integrates the flow with a FIXED-step method (``euler`` by
default, optionally ``midpoint``) over a caller-supplied time grid ``t``. Those
two methods are a few lines each and reproduce ``torchdiffeq.odeint`` exactly, so
F5 needs no ODE dependency at all.

Those two are also the only methods supported. Adaptive solvers such as
``dopri5``/``rk4`` are rejected with ``ValueError`` rather than delegated: they
would need ``torchdiffeq``, which espnet does not ship, and they reinterpret the
supplied grid instead of stepping along it, so the EPSS / sway-sampling schedule
F5 relies on would not survive the round trip.

API mirrors ``torchdiffeq.odeint(func, y0, t, method=...)``:
  * ``func(t_scalar, y) -> dy/dt``
  * ``t`` is the 1-D grid; the step from ``t[i]`` to ``t[i+1]`` uses
    ``dt = t[i+1] - t[i]`` (handles the non-uniform grids F5 produces via EPSS /
    sway sampling).
  * returns the trajectory stacked as ``[len(t), *y0.shape]`` with
    ``solution[0] == y0`` (same as torchdiffeq), so ``trajectory[-1]`` is final.

Matches torchdiffeq's fixed solvers:
  euler:    y1 = y0 + dt * f(t0, y0)
  midpoint: y1 = y0 + dt * f(t0 + dt/2,  y0 + (dt/2) * f(t0, y0))
"""

from __future__ import annotations

from typing import Callable

import torch

_FIXED_STEP = {"euler", "midpoint"}


def odeint(
    func: Callable, y0: torch.Tensor, t: torch.Tensor, method: str = "euler", **kwargs
):
    """Fixed-step ODE integration over a caller-supplied time grid.

    Args:
        func: Derivative ``func(t_scalar, y) -> dy/dt``.
        y0: Initial state.
        t: 1-D time grid. Steps use ``dt = t[i + 1] - t[i]``, so non-uniform
            grids (EPSS, sway sampling) work unchanged.
        method: ``"euler"`` or ``"midpoint"``. No other method is supported.
        **kwargs: Accepted and ignored, so the ``torchdiffeq.odeint`` call
            signature stays a drop-in.

    Returns:
        Trajectory stacked as ``[len(t), *y0.shape]`` with ``solution[0] == y0``,
        matching ``torchdiffeq.odeint``, so ``solution[-1]`` is the final state.

    Raises:
        ValueError: If ``method`` is not ``"euler"`` or ``"midpoint"``.

    Example:
        .. code-block:: python

            >>> sol = odeint(lambda t, y: y, torch.tensor([1.0]),
            ...              torch.linspace(0, 1, 3))
            >>> sol.shape
            torch.Size([3, 1])

    Note:
        Both methods reproduce ``torchdiffeq``'s fixed-step solvers exactly, so
        dropping that dependency changes no output. Returning the whole
        trajectory rather than just the endpoint is deliberate: it keeps the
        ``torchdiffeq`` signature, so this stays a drop-in replacement.
    """
    if method not in _FIXED_STEP:
        raise ValueError(
            f"Unsupported odeint_method {method!r}; F5-TTS supports only "
            f"{sorted(_FIXED_STEP)}."
        )

    solution = [y0]
    y = y0
    for i in range(t.shape[0] - 1):
        t0 = t[i]
        dt = t[i + 1] - t0
        f0 = func(t0, y)
        if method == "euler":
            y = y + dt * f0
        else:  # midpoint
            half_dt = 0.5 * dt
            y = y + dt * func(t0 + half_dt, y + half_dt * f0)
        solution.append(y)

    return torch.stack(solution)
