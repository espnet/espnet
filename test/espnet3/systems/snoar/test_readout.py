import pytest
import torch

from espnet3.systems.snoar.src.readout import (
    check_readout,
    read_speech,
    read_student,
    read_teacher,
)


def test_raw_teacher_is_mixture_consistent():
    x = torch.randn(2, 300)
    out = torch.randn(2, 1, 300)
    sp, nd = read_teacher(out, x, "raw")
    assert torch.allclose(sp + nd, x, atol=1e-5)


def test_dnf_teacher_projects_and_purifies():
    s, n = torch.randn(1, 2000), torch.randn(1, 2000)
    out = torch.stack([s + 0.5 * n, n], dim=1)
    sp, nd = read_teacher(out, s + n, "dnf", tau_rel=0.0)
    assert torch.allclose(sp, align_scale_ref(s, n, s + n), atol=1e-3)
    assert abs(float((nd * sp).sum())) < 1e-2


def align_scale_ref(s, n, x):
    from espnet3.systems.snoar.src.ops import align_scale, project

    return align_scale(project(s + 0.5 * n, n, tau_rel=0.0), x)


def test_student_and_speech_reads():
    torch.manual_seed(0)
    x = torch.randn(1, 100)
    o3 = torch.randn(1, 3, 100)
    sh, nh = read_student(o3, x)
    assert torch.allclose(nh, o3[:, 1] + o3[:, 2])
    sh1, nh1 = read_student(o3[:, :1], x)
    assert torch.allclose(sh1 + nh1, x, atol=1e-6)
    assert read_speech(o3, x, "mixit").shape == (1, 100)


def test_readout_slot_mismatch_raises():
    check_readout("raw", 2)
    with pytest.raises(ValueError, match="needs 3 output slot"):
        check_readout("mixit", 2)
    with pytest.raises(ValueError, match="not in"):
        check_readout("bogus", 1)
