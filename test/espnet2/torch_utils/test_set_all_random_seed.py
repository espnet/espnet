import os
import random

import numpy as np
import pytest
import torch

from espnet2.torch_utils.set_all_random_seed import (
    DETERMINISTIC_CUBLAS_CONFIGS,
    set_all_random_seed,
    set_deterministic,
)


def _draw():
    return random.random(), np.random.rand(), torch.rand(())


def test_set_all_random_seed_is_reproducible():
    set_all_random_seed(0)
    py_a, np_a, torch_a = _draw()
    set_all_random_seed(0)
    py_b, np_b, torch_b = _draw()
    assert py_a == py_b
    assert np_a == np_b
    assert torch.equal(torch_a, torch_b)


def test_set_all_random_seed_differs_between_seeds():
    set_all_random_seed(0)
    py_a, np_a, torch_a = _draw()
    set_all_random_seed(1)
    py_b, np_b, torch_b = _draw()
    assert py_a != py_b
    assert np_a != np_b
    assert not torch.equal(torch_a, torch_b)


@pytest.fixture
def restore_deterministic_state():
    prev_mode = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    prev_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    # The value inherited from the test process must not decide what
    # set_deterministic() does, so start every case from "not set".
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    yield
    torch.use_deterministic_algorithms(prev_mode, warn_only=prev_warn_only)
    if prev_config is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = prev_config


@pytest.mark.parametrize("warn_only", [True, False])
def test_set_deterministic(restore_deterministic_state, warn_only):
    set_deterministic(warn_only=warn_only)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled() is warn_only
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == DETERMINISTIC_CUBLAS_CONFIGS[0]


@pytest.mark.parametrize("config", DETERMINISTIC_CUBLAS_CONFIGS)
def test_set_deterministic_keeps_supported_cublas_config(
    restore_deterministic_state, config
):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = config
    set_deterministic()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == config


def test_set_deterministic_overwrites_unsupported_cublas_config(
    restore_deterministic_state,
):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"
    set_deterministic()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == DETERMINISTIC_CUBLAS_CONFIGS[0]
