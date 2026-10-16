"""Generator checkpoint loading, including conversion of official kNN-VC weights.

The kNN-VC authors released HiFi-GAN generators trained on (prematched) WavLM
features (``prematch_g_02500000.pt`` / ``g_02500000.pt``) in the layout of the
original HiFi-GAN ``Generator`` class (Kong et al., 2020,
https://github.com/jik876/hifi-gan). ESPnet3 uses
``espnet2.gan_tts.hifigan.HiFiGANGenerator`` instead, which is the same
network with different module names, so those weights can be loaded after a
key rename. This module performs that rename and also understands the
checkpoints written by an ESPnet3 training run.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping

import torch

# Official kNN-VC ``Generator`` -> :class:`KNNVCGenerator` parameter names.
# ``lin_pre`` is the WavLM projection; everything else is original HiFi-GAN,
# which lives under ``hifigan.`` as an espnet2 ``HiFiGANGenerator``. Its
# ``resblocks`` are flattened identically in both (upsample-major order), and
# every conv sits at index 1 of an ``(activation, conv)`` Sequential on the
# espnet side, hence the inserted ``.1.``.
_KNN_VC_KEY_RULES = (
    (re.compile(r"^lin_pre\.(.+)$"), r"input_projection.\1"),
    (re.compile(r"^conv_pre\.(.+)$"), r"hifigan.input_conv.\1"),
    (re.compile(r"^ups\.(\d+)\.(.+)$"), r"hifigan.upsamples.\1.1.\2"),
    (
        re.compile(r"^resblocks\.(\d+)\.(convs[12])\.(\d+)\.(.+)$"),
        r"hifigan.blocks.\1.\2.\3.1.\4",
    ),
    (re.compile(r"^conv_post\.(.+)$"), r"hifigan.output_conv.1.\1"),
)

_GENERATOR_PREFIX = "generator."


def load_state_dict_from_path_or_url(
    checkpoint: str | Path, map_location: str | torch.device = "cpu"
) -> Dict[str, Any]:
    """Load a checkpoint from a local path or an ``http(s)://`` URL.

    URLs are fetched with :func:`torch.hub.load_state_dict_from_url`, which
    caches the file under ``$TORCH_HOME/hub/checkpoints`` so repeated runs
    (and Dask workers on the same machine) do not download it again.

    Args:
        checkpoint: Local file path or URL of a ``torch.save``-d object.
        map_location: Device passed to ``torch.load``.

    Returns:
        The deserialized object, typically a dict.

    Raises:
        FileNotFoundError: If ``checkpoint`` is a local path that does not exist.
    """
    checkpoint = str(checkpoint)
    if checkpoint.startswith(("http://", "https://")):
        return torch.hub.load_state_dict_from_url(
            checkpoint, map_location=map_location, progress=True
        )
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def convert_knn_vc_generator_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Rename official kNN-VC generator keys to :class:`KNNVCGenerator` names.

    Args:
        state_dict: The ``"generator"`` entry of an official ``g_*.pt`` file
            (keys such as ``lin_pre.weight``, ``conv_pre.weight_g``,
            ``ups.0.weight_v``, ``resblocks.3.convs1.2.bias``,
            ``conv_post.weight_g``).

    Returns:
        A new state dict loadable into a ``KNNVCGenerator`` built with
        :data:`espnet3.systems.knnvc.vocoder.DEFAULT_GENERATOR_PARAMS`.

    Raises:
        KeyError: If a key does not match any known HiFi-GAN generator parameter,
            which usually means the file is not a HiFi-GAN generator.

    Examples:
        >>> official = torch.load("prematch_g_02500000.pt", map_location="cpu")
        >>> converted = convert_knn_vc_generator_state_dict(official["generator"])
        >>> generator.load_state_dict(converted)
    """
    converted: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        for pattern, replacement in _KNN_VC_KEY_RULES:
            if pattern.match(key):
                converted[pattern.sub(replacement, key)] = value
                break
        else:
            raise KeyError(
                f"Unexpected key '{key}' in kNN-VC generator state dict; "
                "expected lin_pre/conv_pre/ups/resblocks/conv_post parameters."
            )
    return converted


def is_knn_vc_generator_state_dict(state_dict: Mapping[str, Any]) -> bool:
    """Return whether ``state_dict`` uses the official kNN-VC generator layout."""
    return any(key.startswith(("lin_pre.", "conv_pre.")) for key in state_dict)


def load_generator_state_dict(
    checkpoint: str | Path, map_location: str | torch.device = "cpu"
) -> Dict[str, torch.Tensor]:
    """Load HiFi-GAN generator weights from any supported checkpoint layout.

    Supported inputs, detected from the file contents:

    1. Official kNN-VC release files (``prematch_g_02500000.pt``,
       ``g_02500000.pt``): ``{"generator": <original HiFi-GAN state dict>}``. Keys are
       converted with :func:`convert_knn_vc_generator_state_dict`.
    2. Lightning checkpoints written by the ESPnet3 ``train`` stage
       (``exp/<tag>/*.ckpt``): ``{"state_dict": {"generator.*": ...,
       "discriminator.*": ...}, ...}``. Only ``generator.*`` is kept and the
       prefix is stripped.
    3. Averaged models written by ``AverageCheckpointsCallback``
       (``exp/<tag>/*.ave_<K>best.pth``) or any plain state dict, with keys
       either prefixed ``generator.`` or already in ``KNNVCGenerator`` form.

    Args:
        checkpoint: Local path or URL of the checkpoint.
        map_location: Device passed to ``torch.load``.

    Returns:
        State dict with ``KNNVCGenerator`` key names (no ``generator.`` prefix).

    Raises:
        ValueError: If no generator parameters can be found in the file.
    """
    obj = load_state_dict_from_path_or_url(checkpoint, map_location=map_location)
    if not isinstance(obj, Mapping):
        raise ValueError(f"Checkpoint {checkpoint} does not contain a dict.")

    if "generator" in obj and isinstance(obj["generator"], Mapping):
        inner = obj["generator"]
        if is_knn_vc_generator_state_dict(inner):
            return convert_knn_vc_generator_state_dict(inner)
        return dict(inner)

    state_dict = obj["state_dict"] if "state_dict" in obj else obj
    if is_knn_vc_generator_state_dict(state_dict):
        return convert_knn_vc_generator_state_dict(state_dict)

    prefixed = {
        key[len(_GENERATOR_PREFIX) :]: value
        for key, value in state_dict.items()
        if key.startswith(_GENERATOR_PREFIX)
    }
    if prefixed:
        return prefixed

    if any(key.startswith(("input_projection.", "hifigan.")) for key in state_dict):
        return {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}

    raise ValueError(
        f"No HiFi-GAN generator parameters found in {checkpoint}. Expected an "
        "official kNN-VC g_*.pt file, an ESPnet3 .ckpt, or a generator state dict."
    )
