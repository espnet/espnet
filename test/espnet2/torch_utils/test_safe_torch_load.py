"""Tests for safe_torch_load."""

import os
from unittest import mock

import pytest
import torch

from espnet2.torch_utils.safe_torch_load import (
    _ENV_VAR,
    UnsafeLoadRefusedError,
    safe_torch_load,
)


def _save_simple_tensor(path):
    """Save a simple tensor checkpoint (weights_only=True compatible)."""
    torch.save({"x": torch.tensor([1.0, 2.0])}, path)


def _save_incompatible_object(path):
    """Save a checkpoint containing a non-tensor Python object."""
    # Use torch.save so the file is a valid torch archive; weights_only=True
    # will still reject it (non-tensor object), but weights_only=False can
    # load it successfully.
    torch.save({"obj": object()}, path)


class TestSafeTorchLoad:
    def test_loads_simple_checkpoint(self, tmp_path):
        p = tmp_path / "ckpt.pt"
        _save_simple_tensor(p)
        result = safe_torch_load(p)
        assert "x" in result

    def test_raises_on_incompatible_without_opt_in(self, tmp_path):
        p = tmp_path / "bad.pkl"
        _save_incompatible_object(p)
        with pytest.raises(UnsafeLoadRefusedError, match="Refusing to fall back"):
            safe_torch_load(p)

    def test_unsafe_load_refused_is_runtime_error(self, tmp_path):
        """UnsafeLoadRefusedError must be a sub-class of RuntimeError."""
        p = tmp_path / "bad.pkl"
        _save_incompatible_object(p)
        with pytest.raises(RuntimeError):
            safe_torch_load(p)

    def test_env_var_opt_in(self, tmp_path):
        p = tmp_path / "bad.pkl"
        _save_incompatible_object(p)
        with mock.patch.dict(os.environ, {_ENV_VAR: "1"}):
            with pytest.warns(UserWarning, match="unsafe fallback"):
                result = safe_torch_load(p)
        assert result is not None

    def test_error_message_lists_supported_opt_in_paths_only(self, tmp_path):
        p = tmp_path / "bad.pkl"
        _save_incompatible_object(p)

        with pytest.raises(UnsafeLoadRefusedError) as excinfo:
            safe_torch_load(p)

        message = str(excinfo.value)
        assert _ENV_VAR in message
        assert "interactive terminal" in message
        assert "allow_unsafe_fallback" not in message

    def test_env_var_not_set_raises(self, tmp_path):
        p = tmp_path / "bad.pkl"
        _save_incompatible_object(p)
        env = {k: v for k, v in os.environ.items() if k != _ENV_VAR}
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(UnsafeLoadRefusedError):
                safe_torch_load(p)

    def test_no_auto_fallback_without_opt_in(self, tmp_path):
        """Ensure weights_only=False is never called without opt-in."""
        p = tmp_path / "bad.pkl"
        _save_incompatible_object(p)
        with mock.patch("torch.load", wraps=torch.load) as mock_load:
            with pytest.raises(UnsafeLoadRefusedError):
                safe_torch_load(p)
            # Verify weights_only=False was never used
            for call in mock_load.call_args_list:
                kwargs = call.kwargs if call.kwargs else {}
                assert kwargs.get("weights_only") != False  # noqa: E712

    def test_interactive_tty_confirm(self, tmp_path):
        p = tmp_path / "bad.pkl"
        _save_incompatible_object(p)
        with mock.patch(
            "espnet2.torch_utils.safe_torch_load._confirm_unsafe_interactively",
            return_value=True,
        ):
            with pytest.warns(UserWarning, match="unsafe fallback"):
                result = safe_torch_load(p)
        assert result is not None

    def test_interactive_tty_deny(self, tmp_path):
        p = tmp_path / "bad.pkl"
        _save_incompatible_object(p)
        with mock.patch(
            "espnet2.torch_utils.safe_torch_load._confirm_unsafe_interactively",
            return_value=False,
        ):
            with pytest.raises(UnsafeLoadRefusedError, match="Refusing to fall back"):
                safe_torch_load(p)

    def test_non_tty_confirm_returns_false(self):
        """_confirm_unsafe_interactively must return False for non-TTY stdin."""
        from espnet2.torch_utils.safe_torch_load import _confirm_unsafe_interactively

        with mock.patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            assert _confirm_unsafe_interactively("/some/path") is False

    def test_weights_only_kwarg_stripped(self, tmp_path):
        """Caller-supplied weights_only= must be stripped and not conflict."""
        p = tmp_path / "ckpt.pt"
        _save_simple_tensor(p)
        # Should not raise TypeError about duplicate keyword argument
        result = safe_torch_load(p, weights_only=False)
        assert "x" in result
