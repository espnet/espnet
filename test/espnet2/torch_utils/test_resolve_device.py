"""device="auto", and named devices checked against the machine."""

import pytest
import torch

from espnet2.torch_utils.resolve_device import resolve_device


class _FakeAccelerator:
    """A torch.accelerator stand-in that reports one accelerator type."""

    def __init__(self, device_type):
        self.device_type = device_type

    def is_available(self):
        return self.device_type is not None

    def current_accelerator(self):
        return _FakeDevice(self.device_type) if self.device_type else None


class _FakeDevice:
    """Stands in for the torch.device of the accelerator torch reports."""

    def __init__(self, device_type):
        self.type = device_type


def test_auto_is_cuda_when_torch_sees_it_else_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "accelerator", _FakeAccelerator("cuda"), raising=False)
    assert resolve_device("auto") == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "accelerator", _FakeAccelerator(None), raising=False)
    assert resolve_device() == "cpu"


def test_auto_follows_a_non_cuda_accelerator(monkeypatch):
    """auto is whichever accelerator torch reports, not CUDA by assumption."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "accelerator", _FakeAccelerator("npu"), raising=False)
    assert resolve_device("auto") == "npu"


def test_auto_never_picks_mps_on_its_own(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "accelerator", _FakeAccelerator(None), raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("mps") == "mps"


def test_a_torch_without_torch_accelerator_still_answers_auto(monkeypatch):
    """torch.accelerator is 2.5+; older builds keep the CUDA-or-CPU answer."""
    monkeypatch.delattr(torch, "accelerator", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device() == "cpu"


def test_a_named_device_the_machine_lacks_is_refused(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="cuda:0 asked for, but torch sees no CUDA"):
        resolve_device("cuda:0")
    with pytest.raises(ValueError, match="mps asked for, but torch has no MPS"):
        resolve_device("mps")
    assert resolve_device("cpu") == "cpu"


def test_a_cuda_index_is_checked_against_the_count(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("cuda:1") == "cuda:1"
    with pytest.raises(ValueError, match="cuda:2 asked for, but torch sees 2"):
        resolve_device("cuda:2")


def test_a_non_cuda_accelerator_is_named_and_its_index_checked(monkeypatch):
    """An NPU is asked for the same way a GPU is, and checked the same way."""
    npu = type("npu", (), {"device_count": staticmethod(lambda: 4)})
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    assert resolve_device("npu") == "npu"
    assert resolve_device("npu:3") == "npu:3"
    with pytest.raises(ValueError, match="npu:4 asked for, but torch sees 4"):
        resolve_device("npu:4")


def test_an_accelerator_torch_does_not_have_is_refused(monkeypatch):
    monkeypatch.delattr(torch, "npu", raising=False)
    with pytest.raises(ValueError, match="npu asked for, but torch sees no npu"):
        resolve_device("npu")


def test_a_name_that_is_no_device_is_refused():
    with pytest.raises(ValueError, match="unknown device 'gpu'"):
        resolve_device("gpu")
    with pytest.raises(ValueError, match="unknown device 'cuda:one'"):
        resolve_device("cuda:one")
