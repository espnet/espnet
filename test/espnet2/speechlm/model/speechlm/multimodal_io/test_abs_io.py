"""Tests for AbsIO abstract base class."""

import pytest
import torch

from espnet2.speechlm.model.speechlm.multimodal_io.abs_io import AbsIO


class TestAbsIO:
    """Tests for the AbsIO abstract interface."""

    def _make_concrete(self, modality="text", is_discrete=True):
        """Create a minimal concrete subclass for testing."""

        class _ConcreteIO(AbsIO):
            pass

        obj = object.__new__(_ConcreteIO)
        AbsIO.__init__(obj, modality=modality, is_discrete=is_discrete)
        return obj

    def test_init_sets_attributes(self):
        io = self._make_concrete(modality="audio", is_discrete=False)
        assert io.modality == "audio"
        assert io.is_discrete is False

    def test_is_nn_module(self):
        io = self._make_concrete()
        assert isinstance(io, torch.nn.Module)

    def test_unimplemented_methods_raise(self):
        io = self._make_concrete()
        methods_no_args = [
            "copy_for_worker",
            "feature_dim",
            "num_stream",
            "get_vocabulary",
            "get_stream_interval",
            "get_stream_weight",
        ]
        for name in methods_no_args:
            with pytest.raises(NotImplementedError):
                getattr(io, name)()

        with pytest.raises(NotImplementedError):
            io.preprocess("data")
        with pytest.raises(NotImplementedError):
            io.encode_batch([])
        with pytest.raises(NotImplementedError):
            io.decode_batch({})
        with pytest.raises(NotImplementedError):
            io.find_length("data")
        with pytest.raises(NotImplementedError):
            io.dummy_forward(torch.zeros(1))

    def test_concrete_subclass_can_override(self):
        class MyIO(AbsIO):
            def num_stream(self):
                return 4

        obj = object.__new__(MyIO)
        AbsIO.__init__(obj, modality="audio", is_discrete=True)
        assert obj.num_stream() == 4
        # Other methods still raise
        with pytest.raises(NotImplementedError):
            obj.feature_dim()
