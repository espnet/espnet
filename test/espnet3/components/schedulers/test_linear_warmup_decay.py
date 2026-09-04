import pytest
import torch

from espnet3.components.schedulers.linear_warmup_decay import LinearWarmupDecayLR

BASE_LR = 1e-4
WARMUP = 20
TOTAL = 100


def _make(warmup=WARMUP, total=TOTAL, **kwargs):
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=BASE_LR)
    scheduler = LinearWarmupDecayLR(
        optimizer, warmup_steps=warmup, total_steps=total, **kwargs
    )
    return optimizer, scheduler


def _lr_curve(steps):
    """Return the lr seen at each of `steps` optimizer updates."""
    optimizer, scheduler = _make()
    curve = []
    for _ in range(steps):
        curve.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return curve


def test_starts_at_the_warmup_floor():
    optimizer, scheduler = _make(start_factor=1e-3)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(BASE_LR * 1e-3)


def test_peak_is_exactly_the_base_lr_at_the_handover():
    """The warmup/decay handover must land on base_lr with no drift."""
    curve = _lr_curve(TOTAL + 1)
    assert curve[WARMUP] == pytest.approx(BASE_LR, rel=1e-12)


def test_warmup_is_monotonically_increasing():
    curve = _lr_curve(WARMUP + 1)
    assert all(b > a for a, b in zip(curve, curve[1:]))


def test_decay_is_monotonically_decreasing():
    curve = _lr_curve(TOTAL + 1)[WARMUP:]
    assert all(b < a for a, b in zip(curve, curve[1:]))


def test_peak_is_the_maximum_of_the_whole_schedule():
    curve = _lr_curve(TOTAL + 1)
    assert max(curve) == pytest.approx(curve[WARMUP], rel=1e-12)


def test_decays_to_end_factor_at_total_steps():
    optimizer, scheduler = _make(end_factor=1e-2)
    for _ in range(TOTAL):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(BASE_LR * 1e-2, rel=1e-6)


def test_lr_is_clamped_past_total_steps():
    """Training beyond total_steps must hold the floor, never go negative."""
    optimizer, scheduler = _make()
    for _ in range(TOTAL):
        optimizer.step()
        scheduler.step()
    at_total = optimizer.param_groups[0]["lr"]
    for _ in range(50):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(at_total)
    assert optimizer.param_groups[0]["lr"] > 0.0


def test_lr_stays_positive_across_the_whole_schedule():
    assert all(lr > 0.0 for lr in _lr_curve(TOTAL + 1))


def test_is_steppable_per_batch():
    """`scheduler_interval: step` requires the AbsBatchStepScheduler contract."""
    from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler

    _, scheduler = _make()
    assert isinstance(scheduler, AbsBatchStepScheduler)


def test_repr_reports_the_configured_horizon():
    _, scheduler = _make()
    text = repr(scheduler)
    assert "warmup_steps=20" in text and "total_steps=100" in text


def test_zero_warmup_starts_at_the_base_lr():
    """With no warmup there is no floor to ramp from."""
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    scheduler = LinearWarmupDecayLR(optimizer, warmup_steps=0, total_steps=10)

    lrs = []
    for _ in range(11):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert lrs[0] == pytest.approx(1e-3)
    assert max(lrs) == pytest.approx(1e-3)
    assert lrs[-1] < lrs[0]


@pytest.mark.parametrize("warmup_steps, total_steps", [(100, 10), (100, 100), (10, 0)])
def test_a_horizon_shorter_than_the_warmup_is_rejected(warmup_steps, total_steps):
    """The warmup peak would land at or after the end of training."""
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    with pytest.raises(ValueError, match="total_steps"):
        LinearWarmupDecayLR(
            optimizer, warmup_steps=warmup_steps, total_steps=total_steps
        )
