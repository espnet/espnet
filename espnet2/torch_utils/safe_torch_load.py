"""Safe wrapper for torch.load that defaults to weights_only=True."""

import logging
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Union

import torch
from packaging.version import parse as V

is_torch_2_6_plus = V(torch.__version__) >= V("2.6.0")

_ENV_VAR = "ESPNET_ALLOW_UNSAFE_TORCH_LOAD"
_CONFIRM_PHRASE = "I_UNDERSTAND_THE_RISK"


class UnsafeLoadRefusedError(RuntimeError):
    """Raised when safe_torch_load refuses an unsafe fallback.

    This is a sub-class of :class:`RuntimeError` for backward compatibility,
    but it can be caught specifically to distinguish security refusals from
    ordinary :class:`RuntimeError` exceptions raised by, e.g.,
    ``model.load_state_dict``.
    """


def _confirm_unsafe_interactively(path: Union[str, Path]) -> bool:
    """Prompt the user on an interactive TTY to confirm unsafe loading.

    Returns ``True`` only when the user types the exact confirmation phrase.
    Returns ``False`` immediately in non-interactive environments (no TTY).
    """
    if not sys.stdin.isatty():
        return False
    print(
        f"\n[SECURITY WARNING] Loading '{path}' with weights_only=False can "
        "execute arbitrary code embedded in the checkpoint.\n"
        f"Type exactly '{_CONFIRM_PHRASE}' to proceed, or press Enter to abort:\n> ",
        end="",
        flush=True,
    )
    try:
        answer = input().strip()
    except EOFError:
        answer = ""
    return answer == _CONFIRM_PHRASE


def safe_torch_load(
    path: Union[str, Path],
    map_location=None,
    **kwargs,
):
    """Load a PyTorch checkpoint safely, defaulting to ``weights_only=True``.

    In PyTorch >= 2.6 the safe default is ``weights_only=True``, which prevents
    arbitrary code execution via pickle gadgets (CWE-502).  This wrapper always
    tries ``weights_only=True`` first.

    If that attempt fails (e.g. the checkpoint contains non-tensor objects),
    **no automatic fallback is performed**.  Instead, a
    :class:`UnsafeLoadRefusedError` (a sub-class of :class:`RuntimeError`) is
    raised with instructions for explicit opt-in.

    Unsafe fallback (``weights_only=False``) is only performed when *at least
    one* of the following explicit opt-in mechanisms is active:

    * the environment variable ``ESPNET_ALLOW_UNSAFE_TORCH_LOAD=1`` is set, **or**
    * the process is running on an interactive TTY and the user types the
      confirmation phrase ``I_UNDERSTAND_THE_RISK`` when prompted.

    Never pass ``weights_only`` via ``**kwargs``; callers should rely on this
    wrapper's policy.

    Args:
        path: Path to the checkpoint file.
        map_location: Passed directly to ``torch.load``.
        **kwargs: Additional keyword arguments forwarded to ``torch.load``
            (excluding ``weights_only``).

    Returns:
        The deserialized checkpoint object.

    Raises:
        UnsafeLoadRefusedError: If ``weights_only=True`` fails and no explicit
            opt-in is provided, with an actionable message describing the
            opt-in options.
        OSError: If the file cannot be opened (propagated without fallback).
    """
    # Remove any caller-supplied weights_only to enforce our policy.
    kwargs.pop("weights_only", None)

    if not is_torch_2_6_plus:
        raise RuntimeError(
            "safe_torch_load requires PyTorch >= 2.6 for weights_only support. "
            f"Found torch.__version__={torch.__version__}, "
            "which is no longer supported by ESPnet."
        )

    try:
        return torch.load(path, map_location=map_location, weights_only=True, **kwargs)

    except (pickle.UnpicklingError, RuntimeError, TypeError, AttributeError) as e:
        # OSError/FileNotFoundError/PermissionError are intentionally NOT caught
        # here and will propagate to the caller unchanged.

        env_opt_in = os.environ.get(_ENV_VAR, "0") == "1"

        if not (env_opt_in or _confirm_unsafe_interactively(path)):
            raise UnsafeLoadRefusedError(
                f"torch.load with weights_only=True failed for '{path}' "
                f"({type(e).__name__}: {e}).\n"
                "Refusing to fall back to unsafe loading automatically.\n"
                "To load this checkpoint you must explicitly opt in using one of:\n"
                f"  1. Set the environment variable {_ENV_VAR}=1.\n"
                "  2. Run in an interactive terminal and confirm the prompt.\n"
                "Only do this if you fully trust the source of the checkpoint file."
            ) from e

        warnings.warn(
            f"Loading '{path}' with weights_only=False (unsafe fallback). "
            "This can execute arbitrary code from the checkpoint. "
            "Only do this for checkpoints you fully trust and control.",
            UserWarning,
            stacklevel=2,
        )
        logging.warning(
            "Loading '%s' with weights_only=False (unsafe fallback). "
            "Ensure the checkpoint comes from a trusted source.",
            path,
        )
        return torch.load(path, map_location=map_location, weights_only=False, **kwargs)
