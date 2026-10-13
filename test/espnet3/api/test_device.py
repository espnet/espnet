"""device="auto", and named devices checked against the machine."""

import pytest
import torch

from espnet3.api.inference import resolve_device


def test_auto_is_cuda_when_torch_sees_it_else_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device() == "cpu"


def test_auto_never_picks_mps_on_its_own(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("mps") == "mps"


def test_a_named_device_the_machine_lacks_is_refused(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
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


def test_a_name_that_is_no_device_is_refused():
    with pytest.raises(ValueError, match="unknown device 'gpu'"):
        resolve_device("gpu")
