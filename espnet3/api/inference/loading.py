"""Finding a published model, reading its bundle, and the system that serves it.

A ``pack_model`` bundle is a directory with ``meta.yaml`` (the system that
trained it, the schema version, where its files are) and
``conf/inference.yaml`` (how to build its model). This module is the one
place that reads one::

    >>> model = load("espnet/some_pack")                # by Hub tag
    >>> model = load("exp/train/model_pack", device="cuda:0")   # by directory
    >>> model("utt.wav")["text"]

:func:`load` resolves the tag (:func:`locate_pack`), reads ``meta.yaml`` for
the system (:data:`SYSTEM_ALIASES` follows a renamed one) and hands the
directory to that system's ``Inference.from_pretrained``. A system that
wraps one backend builds it with :func:`load_model`, which reads the bundle
(:func:`read_bundle`) and instantiates its ``model`` (:func:`build_model`)
without importing the recipe's own code. The ``infer`` stage's provider
builds through the same :func:`build_model`, so a model is built one way
everywhere.

Calls go one way: the ``infer`` stage's provider and the systems call this
module, and nothing here calls a provider back. A bundle's ``provider``,
``runner`` and ``output_fn`` are the stage's business; loading reads only
its ``model``. A system whose model needs building beyond its constructor
does it in its ``Inference``.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from espnet2.utils.pretrained import ModelTagError
from espnet3.api.inference.base import InferenceAPI
from espnet3.publication.schema import PACK_SCHEMA_VERSION
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)

# A system renamed after bundles were published under its old name: old name
# to current directory. ``load`` looks the name in ``meta.yaml`` up here, so a
# rename is one row and every bundle already on the Hub keeps loading.
SYSTEM_ALIASES: dict[str, str] = {
    "asr": "esp2_asr",  # renamed in #6795; bundles packed before say "asr"
}


# What the infer stage runs a model with, not part of the model: loading
# drops them before the bundled-code check, so a bundle whose only own code
# is one of these loads without trusting anything.
_STAGE_KEYS = ("output_fn", "runner", "provider")

_BUILD_LOCK = threading.Lock()


def locate_pack(tag_or_dir: str | Path) -> Path:
    """Return the directory of a ``pack_model`` bundle, downloading a Hub tag.

    Args:
        tag_or_dir: An existing directory, returned resolved; or a tag
            ``espnet_model_zoo`` downloads and unpacks.

    Returns:
        The directory holding ``meta.yaml``.

    Raises:
        ModelTagError: If the download holds no ``inference_config``, so is
            not a ``pack_model`` bundle (a ``RuntimeError``, the type
            ESPnet2's loaders raise for a tag they cannot serve).

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
        raise ModelTagError(
            f"{tag_or_dir} is not a pack_model bundle: it has no inference_config"
        )
    # Resolve the directory, not the file: in a Hub cache the config is a
    # symlink into `blobs/`, and resolving it first would walk out of the
    # snapshot directory that holds `meta.yaml`.
    return Path(artifacts["inference_config"]).parent.parent.resolve()


def read_meta(pack_dir: str | Path) -> dict[str, Any]:
    """Return a bundle's ``meta.yaml`` as a dict, checking the schema version.

    Args:
        pack_dir: The output directory of ``pack_model()``.

    Returns:
        The parsed ``meta.yaml``.

    Raises:
        FileNotFoundError: If ``pack_dir`` is not a directory or has no
            ``meta.yaml``.
        ValueError: If the bundle was written by a newer ``pack_model`` than
            this installation reads (its ``schema_version`` is higher). An
            older version loads with a warning.

    Examples:
        >>> read_meta("exp/train/model_pack")["system"]
        'esp2_asr'
    """
    bundle_root = Path(pack_dir).resolve()
    if not bundle_root.is_dir():
        raise FileNotFoundError(
            "pack_dir must point to the output directory created by "
            f"pack_model(), but got: {bundle_root}"
        )
    meta_path = bundle_root / "meta.yaml"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"pack_dir must contain meta.yaml from pack_model(), "
            f"but none was found under: {bundle_root}"
        )
    meta = yaml.safe_load(meta_path.read_text("utf-8")) or {}
    schema = int(meta.get("schema_version", 0))
    if schema == 0:
        logger.warning(
            "Bundle at %s has no schema_version (legacy format). "
            "Some features may not be available.",
            bundle_root,
        )
    elif schema > PACK_SCHEMA_VERSION:
        raise ValueError(
            f"Bundle was produced by a newer pack_model "
            f"(schema_version={schema}) than this installation supports. "
            f"Upgrade espnet3."
        )
    elif schema < PACK_SCHEMA_VERSION:
        logger.warning(
            "Bundle at %s has schema_version %d; this installation writes %d. "
            "It loads, but newer features may be missing.",
            bundle_root,
            schema,
            PACK_SCHEMA_VERSION,
        )
    return meta


def read_bundle(
    pack_dir: str | Path,
    *,
    trust_user_code: bool = False,
    drop: tuple[str, ...] = (),
) -> tuple[DictConfig, Path]:
    """Read a bundle's inference config, bound to the bundle directory.

    ``meta.yaml`` says where the inference config is; the config is loaded
    with ``recipe_dir`` set to the bundle root, so the relative paths
    ``pack_model`` wrote resolve. A config that imports code shipped inside
    the bundle (a module or package at its root) is refused unless the
    caller trusts it, in which case the bundle root goes on ``sys.path``.

    Args:
        pack_dir: The output directory of ``pack_model()``.
        trust_user_code: Allow the config to import the bundle's own code.
        drop: Top-level keys removed before the check - what the caller will
            not build, such as the recipe's ``output_fn``, so a reference to
            bundled code there does not count.

    Returns:
        The resolved inference config and the bundle root.

    Raises:
        FileNotFoundError: If the bundle, its ``meta.yaml`` or the inference
            config it names is missing.
        ValueError: If the schema is newer than this installation reads, or
            the config needs bundled code and ``trust_user_code`` is false.

    Examples:
        >>> config, root = read_bundle("exp/train/model_pack")
        >>> config.model._target_
        'espnet3.systems.esp2_asr.inference.Inference'
    """
    meta = read_meta(pack_dir)
    bundle_root = Path(pack_dir).resolve()
    inference_config_rel = (meta.get("yaml_files") or {}).get("inference_config")
    if not inference_config_rel:
        raise FileNotFoundError(
            "meta.yaml must contain yaml_files.inference_config, "
            f"but it was missing in: {bundle_root / 'meta.yaml'}"
        )
    config_path = bundle_root / inference_config_rel
    if not config_path.is_file():
        raise FileNotFoundError(
            f"inference config listed in meta.yaml not found: {config_path}"
        )
    config = _load_inference_config(config_path, bundle_root)
    with open_dict(config):
        for key in drop:
            config.pop(key, None)
    if _uses_bundled_code(config, _bundled_module_names(bundle_root)):
        if not trust_user_code:
            system = meta.get("system")
            if system:
                raise ValueError(
                    "This inference config references bundled user code, but "
                    f"meta.yaml names system {system!r}, whose Inference builds "
                    "the model and never runs a bundle's own code. Re-pack the "
                    "bundle without it."
                )
            raise ValueError(
                "This inference config references bundled user code. "
                "Set trust_user_code=True to allow imports from the "
                "published bundle."
            )
        if str(bundle_root) not in sys.path:
            sys.path.insert(0, str(bundle_root))
        config = _load_inference_config(config_path, bundle_root)
        with open_dict(config):
            for key in drop:
                config.pop(key, None)
    return config, bundle_root


def build_model(config: DictConfig, *, device: Optional[str] = None) -> Any:
    """Instantiate a config's ``model`` on ``device``.

    The one way a model is built from an inference config, for a bundle
    (:func:`load_model`) and for the ``infer`` stage (whose provider calls
    this). ``model._target_`` is instantiated with ``device`` added to its
    arguments; while it builds, the working directory is ``recipe_dir`` so
    the bare relative paths ESPnet2 training configs carry resolve from the
    bundle or recipe root. Only ``model`` is read: ``config.provider`` is
    the ``infer`` stage's, and this never calls one, so a provider calling
    this cannot be called back.

    Args:
        config: An inference config with ``model`` (and perhaps
            ``recipe_dir``, ``device``).
        device: Where to build, such as ``"cpu"`` or ``"cuda:0"``; the
            config's own ``device`` when omitted, else ``"cpu"``.

    Returns:
        The instantiated model: an ``Inference`` when ``model._target_``
        names one, or whatever backend the config names.

    Raises:
        ValueError: If the config has no ``model``.

    Examples:
        >>> config, _ = read_bundle("exp/train/model_pack")
        >>> model = build_model(config, device="cuda:0")
    """
    if isinstance(config, Mapping) and not isinstance(config, DictConfig):
        config = OmegaConf.create(dict(config))
    if device is None:
        device = config.get("device", None) or "cpu"
    if config.get("model", None) is None:
        raise ValueError("inference config has no `model` to build")
    logger.info(
        "Instantiating model %s on %s", getattr(config.model, "_target_", None), device
    )
    recipe_dir = config.get("recipe_dir", None)
    if not recipe_dir:
        return instantiate(config.model, device=device)
    # the working directory is process-wide: one build at a time while it
    # is moved, so a threaded worker or a server loading two models is safe
    with _BUILD_LOCK:
        cwd = os.getcwd()
        os.chdir(str(recipe_dir))
        try:
            model = instantiate(config.model, device=device)
            _absolutise_paths(model)  # while the paths still resolve from here
        finally:
            os.chdir(cwd)
    return model


def load_model(
    pack_dir: str | Path,
    *,
    device: Optional[str] = None,
    trust_user_code: bool = False,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Build a bundle's model, without the ``infer`` stage's trimmings.

    Reads the bundle (:func:`read_bundle`) and builds its ``model``
    (:func:`build_model`). The ``infer`` stage's ``output_fn``, ``runner``
    and ``provider`` are dropped first: none is needed to build, and an
    ``Inference`` fixes its own output, so a bundle whose only bundled code
    is one of them loads without trusting anything.

    Args:
        pack_dir: The output directory of ``pack_model()``.
        device: Where to build the model; the config's own ``device`` when
            omitted, else ``"cpu"``.
        trust_user_code: Allow the bundle's model to be its own bundled
            code.
        overrides: Arguments of the packed ``model``'s constructor that
            replace the packed values, any number and any of them, such as
            ``{"beam_size": 5, "ctc_weight": 0.3}`` - what ESPnet2's
            ``from_pretrained(tag, **kwargs)`` does. A key starting with
            ``_`` (``_target_``) is refused: an override changes how the
            model is built, not which model it is. One the model does not
            take is a ``TypeError`` naming it.

    Returns:
        The bundle's model: an ``Inference`` when the recipe names one as
        its model, else the backend (a ``Speech2Text``, say) for a system's
        ``Inference`` to wrap.

    Raises:
        FileNotFoundError, ValueError: As :func:`read_bundle`.

    Examples:
        >>> load_model("exp/train/model_pack")                 # an Inference
        >>> load_model("exp/old_pack", device="cuda:0")        # a Speech2Text
    """
    config, _ = read_bundle(pack_dir, trust_user_code=trust_user_code, drop=_STAGE_KEYS)
    return _build_bundle_model(config, device, overrides)


def _build_bundle_model(
    config: DictConfig,
    device: Optional[str],
    overrides: Optional[Mapping[str, Any]],
) -> Any:
    """Build a read bundle's ``model`` with the caller's overrides applied.

    An override the model does not take is a ``TypeError`` naming it, not
    hydra's wrapping of the constructor's.
    """
    apply_overrides(config, overrides)
    if device is not None:
        with open_dict(config):
            config.device = device
    try:
        return build_model(config, device=device)
    except Exception as e:
        name = _unexpected_argument(e)
        if not overrides or name not in overrides:
            raise
        # hydra wraps the constructor's TypeError; the caller named this
        # argument, so the error is theirs and says so, as in ESPnet2
        raise TypeError(
            f"{name}={overrides[name]!r}: the bundle's model "
            f"({config.model.get('_target_')}) takes no argument {name!r}"
        ) from e


def apply_overrides(config: DictConfig, overrides: Optional[Mapping[str, Any]]) -> None:
    """Merge constructor arguments into a bundle's ``model``, in place.

    Args:
        config: An inference config, as :func:`read_bundle` returns it.
        overrides: Keyword arguments for the model's constructor; nothing
            to do when empty.

    Raises:
        ValueError: If a key starts with ``_``, or the config has no
            ``model`` to pass them to.

    Examples:
        >>> config, _ = read_bundle("exp/train/model_pack")
        >>> apply_overrides(config, {"beam_size": 5, "ctc_weight": 0.3})
        >>> config.model.beam_size, config.model.ctc_weight
        (5, 0.3)
    """
    if not overrides:
        return
    hidden = sorted(k for k in overrides if str(k).startswith("_"))
    if hidden:
        raise ValueError(
            f"{hidden} cannot be overridden: an override is a constructor "
            "argument of the packed model, not a change of model"
        )
    if config.get("model", None) is None:
        raise ValueError(
            f"inference config has no `model` to pass {sorted(overrides)} to"
        )
    with open_dict(config):
        config.model = OmegaConf.merge(config.model, dict(overrides))


def load(
    tag_or_dir: str | Path,
    *,
    device: str = "cpu",
    system: str | None = None,
    trust_user_code: bool = False,
    **kwargs: Any,
) -> InferenceAPI:
    """Load a published model behind its system's :class:`InferenceAPI`.

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
        trust_user_code: For a bundle that names no system and whose
            ``model`` is an ``Inference`` of its own, shipped as bundled
            code: allow importing it. A bundle served by an installed
            system never needs this.
        **kwargs: Any arguments of the packed model's constructor, replacing
            the packed values - for ESPnet2 ASR, any ``Speech2Text``
            argument, such as ``beam_size=5, ctc_weight=0.3, nbest=3``.
            Forwarded to ``from_pretrained``, or applied to the bundle's
            ``model`` when it is an ``Inference`` itself
            (:func:`apply_overrides`).

    Returns:
        The system's ``Inference`` instance.

    Raises:
        ModelTagError: If ``meta.yaml`` names no system, none is given, and
            the bundle's model is not an ``Inference`` itself; or it names a
            system with no ``inference`` module. Both say the tag cannot be
            served as asked, as ESPnet2's loaders do.
        ImportError: If the system's ``inference`` module defines no
            ``Inference``.

    Examples:
        >>> model = load("espnet/some_pack")
        >>> model = load("exp/train/model_pack", device="cuda:0")
        >>> model = load("exp/old_pack", system="esp2_asr")   # older meta.yaml
    """
    if system is None:
        tag_or_dir = locate_pack(tag_or_dir)
        system = read_meta(tag_or_dir).get("system")
        if not system:
            # no system to serve it: the bundle must build an Inference itself
            config, _ = read_bundle(
                tag_or_dir,
                trust_user_code=trust_user_code,
                drop=_STAGE_KEYS,
            )
            model = (
                _build_bundle_model(config, device, kwargs)
                if config.get("model", None) is not None
                else None
            )
            if isinstance(model, InferenceAPI):
                return model
            what = "no model" if model is None else f"a {type(model).__name__}"
            raise ModelTagError(
                f"{tag_or_dir}/meta.yaml does not name its system, and the bundle "
                f"builds {what}, not an Inference. Pass system=<name>."
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
        raise ModelTagError(
            f"no {name}: system {system!r} has no Inference yet, or meta.yaml "
            "names the wrong system. Pass system=<name>."
        ) from e
    cls = getattr(module, "Inference", None)
    if not isinstance(cls, type) or not issubclass(cls, InferenceAPI):
        raise ImportError(
            f"espnet3.systems.{system}.inference defines no Inference(InferenceAPI)"
        )
    return cls.from_pretrained(tag_or_dir, device=device, **kwargs)


def _unexpected_argument(error: BaseException) -> Optional[str]:
    """Return the keyword a constructor refused, through hydra's wrapping."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        found = re.search(r"unexpected keyword argument '([^']+)'", str(error))
        if isinstance(error, TypeError) and found:
            return found.group(1)
        error = error.__cause__ or error.__context__
    return None


def _load_inference_config(config_path: Path, bundle_root: Path) -> DictConfig:
    """Load a packed config with ``recipe_dir`` rebound to the bundle root."""
    config = load_config_with_defaults(str(config_path), resolve=False)
    config.recipe_dir = str(bundle_root)
    OmegaConf.resolve(config)
    return config


def _bundled_module_names(bundle_root: Path) -> set[str]:
    """Top-level module and package names shipped at the bundle root.

    A directory counts when it holds any Python file, not only when it has
    an ``__init__.py``: Python imports a directory of modules as a namespace
    package, so ``src/code.py`` without ``src/__init__.py`` is still code
    that ``src.code.Local`` would run.
    """
    names = set()
    for child in bundle_root.iterdir():
        if child.is_dir() and (
            (child / "__init__.py").exists() or any(child.rglob("*.py"))
        ):
            names.add(child.name)
        elif child.is_file() and child.suffix == ".py":
            names.add(child.stem)
    return names


def _uses_bundled_code(config: DictConfig, bundled: set[str]) -> bool:
    """Whether any string in the config names a bundled module or something in it."""
    if not bundled:
        return False
    stack = [OmegaConf.to_container(config, resolve=False)]
    while stack:
        value = stack.pop()
        if isinstance(value, str):
            if any(value == m or value.startswith(f"{m}.") for m in bundled):
                return True
        elif isinstance(value, Mapping):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
    return False


def _absolutise_paths(obj: Any, seen: Optional[set] = None) -> None:
    """Turn the relative file paths a built model kept into absolute ones.

    ESPnet2 training configs carry bare relative paths (a token list, a BPE
    model); the model was built with ``recipe_dir`` as the working
    directory, and a lazily initialised part of it (a tokenizer, say) keeps
    working after the caller's directory changes only if those paths are
    made absolute now, while the directory is still the root. Only string
    attributes naming an existing file are rewritten.
    """
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return
    seen.add(id(obj))
    attrs = getattr(obj, "__dict__", None)
    if not attrs:
        return
    for name, value in list(attrs.items()):
        if isinstance(value, str):
            if not os.path.isabs(value) and os.path.isfile(os.path.abspath(value)):
                try:
                    setattr(obj, name, os.path.abspath(value))
                except (AttributeError, TypeError):
                    pass
        elif isinstance(value, dict):
            for v in value.values():
                _absolutise_paths(v, seen)
        elif isinstance(value, (list, tuple)):
            for v in value:
                _absolutise_paths(v, seen)
        elif not isinstance(value, (int, float, bool, bytes, type(None))):
            _absolutise_paths(value, seen)
