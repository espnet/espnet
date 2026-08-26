"""Self-contained fixed-step ODE solvers for F5-TTS sampling.

F5's CFM sampler integrates the flow with a FIXED-step method (``euler`` by
default, optionally ``midpoint``) over a caller-supplied time grid ``t``. Those
two methods are a few lines each and reproduce ``torchdiffeq.odeint`` exactly, so
F5 no longer needs ``torchdiffeq`` for its default configuration.

Only the fixed-step methods are implemented here. Any other method (adaptive
solvers such as ``dopri5``/``rk4``) falls back to ``torchdiffeq`` via a lazy
import, so exotic configs still work if the package is installed, but the
common path has no dependency.

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
    """Fixed-step ODE integration; torchdiffeq fallback for other methods.

    Args:
        func: Derivative ``func(t_scalar, y) -> dy/dt``.
        y0: Initial state.
        t: 1-D time grid. Steps use ``dt = t[i + 1] - t[i]``, so non-uniform
            grids (EPSS, sway sampling) work unchanged.
        method: ``"euler"`` or ``"midpoint"`` built in; anything else is
            delegated to ``torchdiffeq``.
        **kwargs: Forwarded to ``torchdiffeq`` on the fallback path only.

    Returns:
        Trajectory stacked as ``[len(t), *y0.shape]`` with ``solution[0] == y0``,
        matching ``torchdiffeq.odeint``, so ``solution[-1]`` is the final state.

    Raises:
        ImportError: If a non-fixed-step method is requested and ``torchdiffeq``
            is not installed.

    Example:
        .. code-block:: python

            >>> sol = odeint(lambda t, y: y, torch.tensor([1.0]),
            ...              torch.linspace(0, 1, 3))
            >>> sol.shape
            torch.Size([3, 1])

    Note:
        The two built-in methods reproduce ``torchdiffeq``'s fixed-step solvers
        exactly, so the default F5 configuration needs no extra dependency.
        Returning the whole trajectory rather than just the endpoint is
        deliberate: it keeps the ``torchdiffeq`` signature, so the fallback is a
        drop-in.
    """
    if method not in _FIXED_STEP:
        # Adaptive / higher-order solvers: defer to torchdiffeq if available.
        try:
            from torchdiffeq import odeint as _tdq_odeint
        except ImportError as e:
            raise ImportError(
                f"ODE method {method!r} needs torchdiffeq (only "
                f"{sorted(_FIXED_STEP)} are built in). Install torchdiffeq or use "
                "odeint_method: euler / midpoint."
            ) from e
        return _tdq_odeint(func, y0, t, method=method, **kwargs)

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
