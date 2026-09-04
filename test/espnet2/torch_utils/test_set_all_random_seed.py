import os

import pytest
import torch

from espnet2.torch_utils.set_all_random_seed import (
    set_all_random_seed,
    set_deterministic,
)


def test_set_all_random_seed():
    set_all_random_seed(0)


def test_set_all_random_seed_reproducible():
    set_all_random_seed(0)
    a = torch.rand(10)
    set_all_random_seed(0)
    b = torch.rand(10)
    assert torch.equal(a, b)


@pytest.mark.parametrize("warn_only", [True, False])
def test_set_deterministic(warn_only):
    prev_mode = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    prev_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        set_deterministic(warn_only=warn_only)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled() is warn_only
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] in (":4096:8", ":16:8")
    finally:
        torch.use_deterministic_algorithms(prev_mode, warn_only=prev_warn_only)
        if prev_config is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = prev_config


def test_set_deterministic_keeps_existing_cublas_config():
    prev_mode = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    prev_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    try:
        set_deterministic()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    finally:
        torch.use_deterministic_algorithms(prev_mode, warn_only=prev_warn_only)
        if prev_config is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = prev_config
