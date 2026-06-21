"""Safe wrapper for torch.load that defaults to weights_only=True."""

import logging
import pickle
import warnings
from pathlib import Path
from typing import Union

import torch
from packaging.version import parse as V

is_torch_2_6_plus = V(torch.__version__) >= V("2.6.0")


def safe_torch_load(
    path: Union[str, Path],
    map_location=None,
    **kwargs,
):
    """Load a PyTorch checkpoint with weights_only=True by default.

    In PyTorch >= 2.6 the safe default is ``weights_only=True``, which prevents
    arbitrary code execution via pickle gadgets (CWE-502).  This wrapper tries
    ``weights_only=True`` first; if the checkpoint contains non-tensor objects
    that cannot be deserialized that way it falls back to ``weights_only=False``
    with a loud warning so that users of *self-produced* checkpoints are not
    broken while the attack surface for *untrusted* checkpoints is minimised.

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
        OSError: If the file cannot be opened (propagated without fallback).
        Exception: Exceptions not caught by the weights-only fallback handler are
            propagated to the caller unchanged.
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
        # These exceptions are raised when weights_only=True rejects non-tensor
        # objects in the checkpoint.  OSError/FileNotFoundError/PermissionError
        # are intentionally NOT caught here and will propagate to the caller.
        warnings.warn(
            f"torch.load with weights_only=True failed for '{path}' "
            f"({type(e).__name__}: {e}). "
            "Falling back to weights_only=False.  This is potentially unsafe "
            "if the checkpoint file comes from an untrusted source.  "
            "Consider re-saving the checkpoint with a version of ESPnet that "
            "writes weights-only-compatible files.",
            UserWarning,
            stacklevel=2,
        )
        logging.warning(
            "Loading '%s' with weights_only=False (unsafe fallback). "
            "See the UserWarning above for details.",
            path,
        )
        return torch.load(path, map_location=map_location, weights_only=False, **kwargs)
