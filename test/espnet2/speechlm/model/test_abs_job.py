"""Tests for espnet2/speechlm/model/abs_job.py — AbsJobTemplate."""

import pytest

from espnet2.speechlm.model.abs_job import AbsJobTemplate


class _ConcreteJob(AbsJobTemplate):
    """Minimal concrete subclass for testing."""

    def build_preprocessor(self):
        return lambda x: x

    def build_model(self):
        import torch.nn as nn

        return nn.Linear(1, 1)


class _PartialJob(AbsJobTemplate):
    """Subclass that only implements build_preprocessor (still abstract)."""

    def build_preprocessor(self):
        return lambda x: x


def test_init_stores_config_and_is_train():
    config = {"key": "value"}
    job = _ConcreteJob(config, is_train=False)
    assert job.config is config
    assert job.is_train is False


def test_init_default_is_train():
    job = _ConcreteJob({})
    assert job.is_train is True


def test_build_preprocessor_abstract():
    """Direct instantiation of AbsJobTemplate should fail."""
    with pytest.raises(TypeError):
        AbsJobTemplate({})


def test_build_model_abstract():
    """Subclass missing build_model should fail to instantiate."""
    with pytest.raises(TypeError):
        _PartialJob({})


def test_concrete_subclass():
    job = _ConcreteJob({"a": 1})
    preprocessor = job.build_preprocessor()
    assert callable(preprocessor)
    model = job.build_model()
    assert model is not None
