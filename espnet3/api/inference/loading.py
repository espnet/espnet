"""Finding a published model and the system that serves it."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml

from espnet3.api.inference.base import BaseInference

# A system renamed after bundles were published under its old name: old name
# to current directory. ``load`` looks the name in ``meta.yaml`` up here, so a
# rename is one row and every bundle already on the Hub keeps loading.
SYSTEM_ALIASES: dict[str, str] = {}


def locate_pack(tag_or_dir: str | Path) -> Path:
    """Return the directory of a ``pack_model`` bundle, downloading a Hub tag.

    Args:
        tag_or_dir: An existing directory, returned resolved; or a tag
            ``espnet_model_zoo`` downloads and unpacks.

    Returns:
        The directory holding ``meta.yaml``.

    Raises:
        RuntimeError: If the download holds no ``inference_config``, so is
            not a ``pack_model`` bundle.

    Examples:
        >>> locate_pack("exp/train/model_pack")
        PosixPath('/.../exp/train/model_pack')
        >>> locate_pack("espnet/some_pack")   # downloaded into the cache
        PosixPath('/.../.cache/espnet/.../model_pack')
    """
    path = Path(tag_or_dir)
    if path.is_dir():
        return path.resolve()
    from espnet_model_zoo.downloader import ModelDownloader

    artifacts = ModelDownloader().download_and_unpack(str(tag_or_dir))
    if "inference_config" not in artifacts:
        raise RuntimeError(
            f"{tag_or_dir} is not a pack_model bundle: it has no inference_config"
        )
    return Path(artifacts["inference_config"]).resolve().parent.parent


def load(
    tag_or_dir: str | Path,
    *,
    device: str = "cpu",
    system: str | None = None,
    **kwargs: Any,
) -> BaseInference:
    """Load a published model behind its system's :class:`BaseInference`.

    The bundle's ``meta.yaml`` names the system that trained it
    (``system: esp2_asr``, written by ``pack_model``); this imports
    ``espnet3.systems.<system>.inference`` and calls its
    ``Inference.from_pretrained``. The caller needs to know nothing about
    the system.

    Args:
        tag_or_dir: A ``pack_model`` directory or a Hub tag.
        device: Where to build the model.
        system: Overrides the name in ``meta.yaml`` - for a bundle packed
            before the name was recorded, or a model that is not an ESPnet3
            bundle at all, in which case ``tag_or_dir`` is passed to the
            system as it is. A name a system has since given up is followed
            through :data:`SYSTEM_ALIASES`.
        **kwargs: Forwarded to ``from_pretrained``.

    Returns:
        The system's ``Inference`` instance.

    Raises:
        ValueError: If ``meta.yaml`` names no system and none is given.
        ImportError: If the system has no ``inference.Inference``.

    Examples:
        >>> model = load("espnet/some_pack")
        >>> model = load("exp/train/model_pack", device="cuda:0")
        >>> model = load("exp/old_pack", system="esp2_asr")   # older meta.yaml
    """
    if system is None:
        tag_or_dir = locate_pack(tag_or_dir)
        meta = yaml.safe_load((tag_or_dir / "meta.yaml").read_text("utf-8")) or {}
        system = meta.get("system")
        if not system:
            raise ValueError(
                f"{tag_or_dir}/meta.yaml does not name its system; it was packed "
                "before pack_model recorded one. Pass system=<name>."
            )
    system = SYSTEM_ALIASES.get(system, system)
    name = f"espnet3.systems.{system}.inference"
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as e:
        # the system's own module missing is one thing; a dependency it
        # imports missing is another, and stays the error it was
        if e.name is None or not (e.name == name or name.startswith(e.name + ".")):
            raise
        raise ImportError(
            f"no {name}: system {system!r} has no Inference yet, or meta.yaml "
            "names the wrong system. Pass system=<name>."
        ) from e
    cls = getattr(module, "Inference", None)
    if not isinstance(cls, type) or not issubclass(cls, BaseInference):
        raise ImportError(
            f"espnet3.systems.{system}.inference defines no Inference(BaseInference)"
        )
    return cls.from_pretrained(tag_or_dir, device=device, **kwargs)
