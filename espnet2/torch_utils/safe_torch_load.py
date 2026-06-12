"""Safe wrapper for torch.load that defaults to weights_only=True."""

import logging
import warnings
from pathlib import Path
from typing import Union

import torch


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
    """
    # Remove any caller-supplied weights_only to enforce our policy.
    kwargs.pop("weights_only", None)

    try:
        return torch.load(path, map_location=map_location, weights_only=True, **kwargs)
    except Exception as e:
        warnings.warn(
            f"torch.load with weights_only=True failed for '{path}' ({type(e).__name__}: {e}). "
            "Falling back to weights_only=False.  This is potentially unsafe if the "
            "checkpoint file comes from an untrusted source.  Consider re-saving the "
            "checkpoint with a version of ESPnet that writes weights-only-compatible files.",
            UserWarning,
            stacklevel=2,
        )
        logging.warning(
            "Loading '%s' with weights_only=False (unsafe fallback). "
            "See the UserWarning above for details.",
            path,
        )
        return torch.load(
            path, map_location=map_location, weights_only=False, **kwargs
        )
