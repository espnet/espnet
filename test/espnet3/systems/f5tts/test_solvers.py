"""Fixed-step ODE solvers.

``solvers.odeint`` replaces ``torchdiffeq.odeint`` for the two fixed-step
methods F5's sampler uses, so the tests pin it against the closed-form
solutions rather than against torchdiffeq, which espnet does not ship. The
drop-in contract matters as much as the arithmetic: ``CFM.sample`` forwards
``odeint_kwargs`` wholesale and reads ``trajectory[-1]``, so the accepted
signature and the returned shape are covered here too.
"""

import pytest
import torch

from espnet3.systems.f5tts.solvers import odeint


def test_euler_matches_the_closed_form_solution():
    """dy/dt = y from y0 = 1 steps to (1 + dt) ** n under euler."""
    t = torch.linspace(0.0, 1.0, 5)
    sol = odeint(lambda _t, y: y, torch.tensor([1.0]), t, method="euler")

    assert sol.shape == (5, 1)
    torch.testing.assert_close(sol[0], torch.tensor([1.0]))
    torch.testing.assert_close(sol[-1], torch.tensor([1.25**4]))


def test_midpoint_matches_the_closed_form_step():
    """One midpoint step on dy/dt = y is y0 * (1 + dt + dt ** 2 / 2)."""
    dt = 0.5
    sol = odeint(
        lambda _t, y: y,
        torch.tensor([1.0]),
        torch.tensor([0.0, dt]),
        method="midpoint",
    )

    torch.testing.assert_close(sol[-1], torch.tensor([1.0 + dt + dt**2 / 2]))


def test_midpoint_is_second_order_so_beats_euler_on_the_same_grid():
    """Both approximate exp(1); midpoint must land closer."""

    def f(_t, y):
        return y

    t = torch.linspace(0.0, 1.0, 5)
    y0 = torch.tensor([1.0])
    exact = torch.e

    euler_err = abs(odeint(f, y0, t, method="euler")[-1].item() - exact)
    midpoint_err = abs(odeint(f, y0, t, method="midpoint")[-1].item() - exact)

    assert midpoint_err < euler_err


def test_step_size_follows_a_non_uniform_grid():
    """F5's sway sampling produces uneven grids, so dt is read per step."""
    t = torch.tensor([0.0, 0.1, 1.0])
    sol = odeint(lambda _t, y: torch.ones_like(y), torch.tensor([0.0]), t)

    # dy/dt = 1, so the exact solution is y = t regardless of the spacing.
    torch.testing.assert_close(sol.reshape(-1), t)


def test_derivative_receives_the_grid_time_not_the_step_index():
    seen = []
    t = torch.tensor([0.0, 0.25, 0.75])
    odeint(
        lambda ti, y: seen.append(float(ti)) or torch.zeros_like(y),
        torch.tensor([0.0]),
        t,
    )
    assert seen == [0.0, 0.25]


def test_midpoint_evaluates_the_derivative_at_the_half_step():
    """The second evaluation sits at t0 + dt / 2, not at the grid point."""
    seen = []
    odeint(
        lambda ti, y: seen.append(float(ti)) or torch.zeros_like(y),
        torch.tensor([0.0]),
        torch.tensor([0.0, 0.4]),
        method="midpoint",
    )

    assert seen == pytest.approx([0.0, 0.2])


def test_the_default_method_is_euler():
    """CFM leaves the method unset for its default sampler."""
    f, y0, t = (lambda _t, y: y), torch.tensor([1.0]), torch.linspace(0.0, 1.0, 4)

    torch.testing.assert_close(odeint(f, y0, t), odeint(f, y0, t, method="euler"))


def test_the_trajectory_keeps_the_state_shape():
    """F5 integrates a [batch, frames, mel] state, not a scalar."""
    y0 = torch.randn(2, 7, 100)

    sol = odeint(lambda _t, y: y, y0, torch.linspace(0.0, 1.0, 4))

    assert sol.shape == (4, 2, 7, 100)
    torch.testing.assert_close(sol[0], y0)


def test_a_single_point_grid_returns_the_initial_state_untouched():
    """No interval means no step, so the derivative is never evaluated."""
    calls = []
    y0 = torch.randn(2, 3)

    sol = odeint(lambda _t, y: calls.append(_t) or y, y0, torch.tensor([0.3]))

    assert sol.shape == (1, 2, 3)
    torch.testing.assert_close(sol[0], y0)
    assert calls == []


def test_extra_keyword_arguments_are_accepted_and_ignored():
    """CFM forwards odeint_kwargs wholesale, so tolerances must not raise."""
    f, y0, t = (lambda _t, y: y), torch.tensor([1.0]), torch.linspace(0.0, 1.0, 4)

    sol = odeint(f, y0, t, method="euler", atol=1e-5, rtol=1e-5, options={"foo": 1})

    torch.testing.assert_close(sol, odeint(f, y0, t, method="euler"))


@pytest.mark.parametrize("method", ["euler", "midpoint"])
def test_the_initial_state_is_not_modified_in_place(method):
    y0 = torch.randn(3)
    before = y0.clone()

    odeint(lambda _t, y: y, y0, torch.linspace(0.0, 1.0, 3), method=method)

    torch.testing.assert_close(y0, before)


@pytest.mark.parametrize("method", ["euler", "midpoint"])
def test_the_state_dtype_survives_a_float32_time_grid(method):
    """Sampling runs under autocast, so a half-precision state must stay half."""
    y0 = torch.ones(2, dtype=torch.bfloat16)

    sol = odeint(lambda _t, y: y, y0, torch.linspace(0.0, 1.0, 3), method=method)

    assert sol.dtype == torch.bfloat16


def test_unsupported_method_is_rejected_rather_than_delegated():
    """Adaptive solvers would need torchdiffeq, which espnet does not ship."""
    with pytest.raises(ValueError, match="dopri5"):
        odeint(
            lambda _t, y: y,
            torch.tensor([1.0]),
            torch.linspace(0, 1, 3),
            method="dopri5",
        )


def test_the_error_names_the_supported_methods():
    with pytest.raises(ValueError) as excinfo:
        odeint(lambda _t, y: y, torch.tensor([1.0]), torch.linspace(0, 1, 3), "rk4")
    message = str(excinfo.value)
    assert "euler" in message and "midpoint" in message


def test_the_method_is_checked_before_any_step_is_taken():
    """A bad method must not leave a half-integrated state behind."""
    calls = []

    with pytest.raises(ValueError):
        odeint(
            lambda _t, y: calls.append(_t) or y,
            torch.tensor([1.0]),
            torch.linspace(0, 1, 3),
            method="rk4",
        )

    assert calls == []
