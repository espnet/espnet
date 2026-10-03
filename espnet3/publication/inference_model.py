"""Publication-side inference API for packed ESPnet models.

This module is the runtime entry point used after ``pack_model()`` has created
an unpacked publication bundle. The public API is :class:`InferenceModel`,
which loads ``conf/inference.yaml`` from that bundle, rebuilds the configured
backend through :class:`espnet3.systems.base.inference_provider.InferenceProvider`,
and exposes a small direct-inference interface for single samples and batches.

Typical call flow:

- ``espnet3.publication.InferenceModel.from_packed(...)``
- read ``meta.yaml``
- locate and resolve ``conf/inference.yaml``
- optionally allow bundled recipe code when ``trust_user_code=True``
- instantiate the backend model and optional ``output_fn``
- run ``forward()`` or ``forward_batch()``
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from espnet_model_zoo.downloader import ModelDownloader
from hydra.utils import get_class, get_object
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from espnet3.publication.schema import PACK_SCHEMA_VERSION
from espnet3.systems.base.inference_provider import InferenceProvider
from espnet3.systems.base.inference_runner import InferenceRunner, _load_output_fn
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.logging_utils import configure_logging

logger = configure_logging()


def _load_inference_config(
    config_path: Path,
    bundle_root: Path,
) -> DictConfig:
    """Load a packed inference config and bind it to the bundle root.

    Called by :meth:`InferenceModel.from_packed` after the packed bundle has
    been located. Packed configs are written with ``recipe_dir: .``,
    so this helper rewrites ``recipe_dir`` to the unpacked bundle root before
    resolving OmegaConf interpolations.
    """
    config = load_config_with_defaults(str(config_path), resolve=False)
    config.recipe_dir = str(bundle_root)
    OmegaConf.resolve(config)
    return config


def _get_bundled_module_names(bundle_root: Path) -> set[str]:
    """Return importable top-level module names shipped in the bundle.

    Called by :meth:`InferenceModel.from_packed` before deciding whether the
    packed config references recipe-local Python code. Only top-level package
    directories and ``.py`` files are considered because those are the names
    that can appear in import paths inside the packed config.
    """
    bundled_modules = set()
    for child in bundle_root.iterdir():
        if child.is_dir() and (child / "__init__.py").exists():
            bundled_modules.add(child.name)
        elif child.is_file() and child.suffix == ".py":
            bundled_modules.add(child.stem)
    return bundled_modules


def _uses_bundled_code(
    config: DictConfig,
    bundled_modules: set[str],
) -> bool:
    """Return whether a config references importable modules in the bundle.

    Called by :meth:`InferenceModel.from_packed` to decide whether loading the
    packed model would execute bundled recipe code. The check walks the config
    tree and looks for string values that either equal a bundled module name or
    start with ``<module>.``.
    """
    if not bundled_modules:
        return False

    stack = [OmegaConf.to_container(config, resolve=False)]
    while stack:
        value = stack.pop()
        if isinstance(value, str):
            if any(
                value == module or value.startswith(f"{module}.")
                for module in bundled_modules
            ):
                return True
            continue
        if isinstance(value, Mapping):
            stack.extend(value.values())
            continue
        if isinstance(value, (list, tuple)):
            stack.extend(value)
    return False


_ALLOWED_TARGET_PREFIXES = (
    "espnet2.",
    "espnet3.",
    "lightning.",
    "torch.optim.",
)

# Refused by name and by resolved identity. The resolved-origin check below
# already rejects everything outside the allowed namespaces, so this list is
# defence in depth: it keeps a loader out even if a prefix is widened later or
# one of these is ever re-exported from an allowed module.
_DENIED_TARGETS = frozenset(
    {
        "torch.load",
        "torch.jit.load",
        "torch.serialization.load",
        "builtins.eval",
        "builtins.exec",
        "builtins.__import__",
        "os.system",
        "subprocess.run",
        "subprocess.Popen",
        "subprocess.call",
        "subprocess.check_output",
    }
)


def _iter_target_strings(config: DictConfig) -> list[str]:
    """Return every ``_target_`` string in the config tree.

    Called by :func:`_disallowed_targets`. Hydra instantiates recursively, so a
    ``_target_`` nested inside an argument runs just like the top-level one and
    has to be collected here too.
    """
    targets = []
    stack = [OmegaConf.to_container(config, resolve=False)]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            target = value.get("_target_")
            if isinstance(target, str):
                targets.append(target)
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
    return targets


def _resolve_target(target: str):
    """Return ``(object, "<module>.<qualname>")`` for a target, or None.

    Called by :func:`_disallowed_target`. A dotted path says where a name was
    written, not what it resolves to: hydra imports the longest importable
    prefix and then walks attributes, so an allowed module that does
    ``import os`` turns ``<allowed module>.os.system`` into :func:`os.system`.
    Resolving tells us both where the callable really comes from and what kind
    of object it is.
    """
    try:
        obj = get_object(target)
    except Exception:
        return None
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    if not module or not qualname:
        return None
    return obj, f"{module}.{qualname}"


def _disallowed_target(target: str, *, require_class: bool) -> str | None:
    """Return a rejection reason for a path, or None when it is allowed.

    Both kinds of path share the namespace and resolved-origin checks. Check
    the written name before resolving it, so an untrusted module is never
    imported. Only the kind check differs: Hydra targets must be classes,
    while ``output_fn`` must be a callable that is not a class.
    """
    if target in _DENIED_TARGETS or not target.startswith(_ALLOWED_TARGET_PREFIXES):
        return target
    resolved = _resolve_target(target)
    if resolved is None:
        return f"{target} (does not resolve)"
    obj, origin = resolved
    if origin in _DENIED_TARGETS or not origin.startswith(_ALLOWED_TARGET_PREFIXES):
        return f"{target} (resolves to {origin})"
    if require_class:
        if not inspect.isclass(obj):
            return f"{target} (resolves to {origin}, which is not a class)"
    elif not callable(obj) or inspect.isclass(obj):
        return (
            f"{target} (resolves to {origin}, "
            "expected a callable that is not a class)"
        )
    return None


def _disallowed_targets(config: DictConfig) -> list[str]:
    """Return the ``_target_`` and ``output_fn`` paths this loader refuses to load.

    Called by :meth:`InferenceModel.from_packed` and :func:`load_backend`.
    Each path must pass the same namespace and resolved-origin checks, but
    ``_target_`` must name a class and ``output_fn`` a non-class callable.

    The class requirement is not a claim that every class is safe to
    instantiate. It is narrowing by what published bundles actually do: every
    ``_target_`` in espnet3, egs3 and the tests is a class, so requiring one
    costs nothing in use, while it removes module-level functions as a
    category -- and some of those are not model components at all.
    ``espnet2.bin.launch.main`` clears both namespace checks and runs
    ``subprocess.Popen`` with the arguments it is handed.

    None of this makes an arbitrary config safe: the arguments passed to an
    allowed class, and OmegaConf resolvers, are a separate surface.
    """
    targets = [(target, True) for target in _iter_target_strings(config)]
    output_fn = config.get("output_fn")
    if isinstance(output_fn, str) and output_fn:
        targets.append((output_fn, False))

    disallowed = set()
    for target, require_class in targets:
        reason = _disallowed_target(target, require_class=require_class)
        if reason is not None:
            disallowed.add(reason)
    return sorted(disallowed)


def _resolve_packed_config(pack_dir: str | Path) -> tuple[Path, Path]:
    """Return the packed inference config's path and the bundle root.

    The checks every loader of a ``pack_model`` bundle makes: the directory
    exists, ``meta.yaml`` is there and of a schema this installation reads,
    and it names an inference config that exists.
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
    with meta_path.open("r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}

    schema = int(meta.get("schema_version", 0))
    if schema == 0:
        logger.warning(
            "Bundle at %s has no schema_version (legacy format). "
            "Some features may not be available.",
            bundle_root,
        )
    elif schema == PACK_SCHEMA_VERSION:
        pass
    else:
        raise ValueError(
            f"Bundle was produced by a newer pack_model "
            f"(schema_version={schema}) than this installation supports. "
            f"Upgrade espnet3."
        )

    inference_config_rel = (meta.get("yaml_files") or {}).get("inference_config")
    if not inference_config_rel:
        raise FileNotFoundError(
            "meta.yaml must contain yaml_files.inference_config, "
            f"but it was missing in: {meta_path}"
        )
    inference_config_path = bundle_root / inference_config_rel
    if not inference_config_path.is_file():
        raise FileNotFoundError(
            "inference config listed in meta.yaml not found: "
            f"{inference_config_path}"
        )
    return inference_config_path, bundle_root


def _provider_class(config: DictConfig):
    """Return the provider class the packed config names, or the default.

    The default is for backward compatibility: bundles packed before
    ``provider`` was recorded in ``conf/inference.yaml`` name none, and
    were built with :class:`InferenceProvider`. A bundle packed today
    always names its provider.
    """
    target = getattr(getattr(config, "provider", None), "_target_", None)
    return get_class(target) if target else InferenceProvider


def load_backend(pack_dir: str | Path, *, device: str | None = None):
    """Build a bundle's model alone, for a caller that owns the output.

    An ``espnet3.api`` system fixes what it returns, so the recipe's
    ``output_fn`` is never needed and never imported; nor is the runner,
    which belongs to the ``infer`` stage. Both are dropped from the config
    before the bundled-code check, which then covers what is built here:
    the model and its provider. A bundle whose model or provider needs
    bundled code is refused, because this caller trusts none.

    Args:
        pack_dir: The output directory of ``pack_model()``.
        device: Where to build the model; the provider picks when omitted.

    Returns:
        The instantiated backend, such as a ``Speech2Text``.

    Raises:
        FileNotFoundError: If ``pack_dir`` is not a bundle, or its
            ``meta.yaml`` names no inference config.
        ValueError: If the model or provider needs the bundle's own code.

    Examples:
        >>> speech2text = load_backend("exp/train/model_pack")
        >>> speech2text = load_backend("exp/train/model_pack", device="cuda:0")
    """
    inference_config_path, bundle_root = _resolve_packed_config(pack_dir)
    config = _load_inference_config(inference_config_path, bundle_root=bundle_root)
    with open_dict(config):
        # neither is built here: the runner is the infer stage's, and the
        # recipe's output_fn is what an API system replaces
        config.pop("output_fn", None)
        config.pop("runner", None)
        if device is not None:
            config.device = device
    if _uses_bundled_code(config, _get_bundled_module_names(bundle_root)):
        raise ValueError(
            f"The model in {bundle_root} needs the bundle's own code to build, "
            "which an espnet3.api system does not import. Load it with "
            "InferenceModel.from_packed(..., trust_user_code=True) instead."
        )
    # The check above only sees code shipped in the bundle. A `_target_` naming
    # something already installed here passes it untouched and is then built,
    # so constrain the targets too. This caller trusts none, so there is
    # nothing to opt into: a disallowed target is simply refused.
    disallowed = _disallowed_targets(config)
    if disallowed:
        raise ValueError(
            f"The model in {bundle_root} builds targets outside the namespaces "
            "a published bundle may build from: " + ", ".join(disallowed) + ". "
            "Load it with InferenceModel.from_packed(..., trust_user_code=True) "
            "if you trust the publisher of this bundle."
        )
    return _provider_class(config).build_model(config)


class InferenceModel:
    """User-facing inference wrapper for packaged ESPnet models.

    This class is the public runtime API for a bundle produced by
    ``espnet3.utils.publication_utils.pack_model()``. It sits on the publication side of
    the pipeline: stage runners produce the packed directory, then external
    callers use :class:`InferenceModel` to reopen that directory and execute
    the bundled inference configuration without going back through
    ``run.py``.

    Internally the wrapper rebuilds the backend declared in
    ``conf/inference.yaml`` through :class:`InferenceProvider`, normalizes
    sample inputs to match ``input_key``, and optionally applies the recipe's
    ``output_fn`` so the published model returns the same payload shape used by
    recipe inference.

    The inference model can be built from:

    - a packaged model tag via :meth:`from_pretrained`
    - a packed model directory via :meth:`from_packed`

    When bundled user code is enabled, the packed bundle root is added to
    ``sys.path`` before backend construction. This is meant for explicitly
    trusted recipe code bundled with the published model.

    Args:
        The constructor is usually reached through :meth:`from_packed` or
        :meth:`from_pretrained`, not called directly. Those classmethods handle
        bundle lookup, config loading, and bundled-code trust checks before
        passing the resolved inference config here.

    Notes:
        This wrapper does not require dataset objects. Single-sample inference
        accepts either a raw value for single-input models or a mapping that
        contains the configured ``input_key`` fields.

    Examples:
        >>> model = InferenceModel.from_pretrained(
        ...     "espnet/some_model",
        ...     trust_user_code=True,
        ... )
        >>> result = model(audio_array)
        >>> batch = model.forward_batch([audio_a, audio_b])
    """

    def __init__(self, inference_config: DictConfig) -> None:
        """Initialize the inference model from a resolved inference config.

        Called by :meth:`from_packed` and :meth:`from_pretrained` after bundle
        discovery and trust checks are complete. This constructor instantiates
        the backend model, normalizes ``input_key`` into either a string or a
        list of strings, and loads the optional recipe ``output_fn``.

        Args:
            inference_config: Resolved inference config loaded from the packed
                bundle.
        """
        provider_cls = _provider_class(inference_config)
        runner_target = getattr(
            getattr(inference_config, "runner", None), "_target_", None
        )
        self.runner_cls = get_class(runner_target) if runner_target else InferenceRunner

        self.model = provider_cls.build_model(inference_config)
        input_key = getattr(inference_config, "input_key", "speech")
        self.input_key = (
            list(input_key)
            if isinstance(input_key, (list, tuple, ListConfig))
            else input_key
        )
        output_fn_path = getattr(inference_config, "output_fn", None)
        self.output_fn = _load_output_fn(output_fn_path) if output_fn_path else None

    @classmethod
    def from_packed(
        cls,
        pack_dir: str | Path,
        trust_user_code: bool = False,
        device: str | None = None,
    ) -> "InferenceModel":
        """Build an inference model from a packed model directory.

        This is the main entry point for local publication bundles. It is
        called by external users, CI checks, and any runtime that already has
        an unpacked ``pack_model()`` output directory. The method validates the
        bundle layout, loads ``meta.yaml``, finds ``yaml_files.inference_config``,
        resolves the inference config against the bundle root, and then
        instantiates :class:`InferenceModel`.

        If the config references bundled code, or ``_target_`` or ``output_fn``
        paths outside the allowed namespaces, the load is blocked unless
        ``trust_user_code=True``. For trusted bundled code, the bundle root is
        inserted into ``sys.path`` and the config is reloaded so import-based
        objects resolve against the newly trusted code.

        Args:
            pack_dir: Path to the output directory created by
                ``espnet3.utils.publication_utils.pack_model()``. This directory must
                contain ``conf/inference.yaml`` and any files referenced by
                that config.
            trust_user_code: Set to ``True`` to allow bundled recipe code and
                ``_target_`` or ``output_fn`` paths outside the allowed namespaces.
                Enable this only if you trust the bundle's publisher.
            device: Where to build the model, such as ``"cpu"`` or
                ``"cuda:0"``. Takes the place of the packed config's own
                ``device``; when omitted the provider picks, as the ``infer``
                stage does.

        Returns:
            InferenceModel: Inference model loaded from ``pack_model`` output.

        Raises:
            FileNotFoundError: If the bundle directory, ``meta.yaml``, or the
                referenced inference config is missing.
            ValueError: If the config requires bundled code or disallowed
                ``_target_`` or ``output_fn`` paths, but ``trust_user_code`` is
                ``False``.

        Notes:
            ``meta.yaml`` is treated as the source of truth for locating the
            packed inference config. The method does not assume that the config
            lives at a fixed path other than the metadata contract written by
            ``pack_model()``.

        Examples:
            >>> model = InferenceModel.from_packed("/path/to/packed_model")
            >>> result = model(audio_array)

            >>> model = InferenceModel.from_packed(
            ...     "/path/to/packed_model",
            ...     trust_user_code=True,
            ... )
        """
        inference_config_path, bundle_root = _resolve_packed_config(pack_dir)
        inference_config = _load_inference_config(
            inference_config_path,
            bundle_root=bundle_root,
        )
        bundled_modules = _get_bundled_module_names(bundle_root)

        if _uses_bundled_code(inference_config, bundled_modules):
            if not trust_user_code:
                raise ValueError(
                    "This inference config references bundled user code. "
                    "Set trust_user_code=True to allow imports from the "
                    "published bundle."
                )
            bundle_root_str = str(bundle_root)
            if bundle_root_str not in sys.path:
                sys.path.insert(0, bundle_root_str)
            inference_config = _load_inference_config(
                inference_config_path,
                bundle_root=bundle_root,
            )

        # The bundled-code check above only sees modules shipped in the bundle.
        # A ``_target_`` or ``output_fn`` naming something already installed in
        # this environment passes it untouched, so constrain those paths as well.
        if not trust_user_code:
            disallowed = _disallowed_targets(inference_config)
            if disallowed:
                raise ValueError(
                    "This inference config references targets outside the "
                    "namespaces a published bundle may build from: "
                    + ", ".join(disallowed)
                    + ". Loading `_target_` or `output_fn` can execute code. "
                    "Set trust_user_code=True only if "
                    "you trust the publisher of this bundle."
                )

        if device is not None:
            with open_dict(inference_config):
                inference_config.device = device

        return cls(inference_config)

    @classmethod
    def from_pretrained(
        cls,
        model_tag: str,
        trust_user_code: bool = False,
        device: str | None = None,
    ) -> "InferenceModel":
        """Download a packaged model and build an inference model from it.

        This is the remote-loading companion to :meth:`from_packed`. It is
        called when the caller has an ``espnet_model_zoo`` tag rather than a
        local packed directory. The downloader fetches and unpacks the model
        assets first, then this method locates the unpacked bundle root and
        delegates to :meth:`from_packed` for the actual config loading and
        backend construction.

        Args:
            model_tag: Pretrained model identifier understood by
                ``espnet_model_zoo``.
            trust_user_code: Forwarded to :meth:`from_packed`.
            device: Forwarded to :meth:`from_packed`.

        Returns:
            InferenceModel: Downloaded inference model.

        Raises:
            RuntimeError: If the downloaded artifacts do not include an
                ``inference_config`` entry.

        Notes:
            The downloader returns individual artifact paths. This method uses
            the downloaded ``inference_config`` path to recover the enclosing
            pack directory expected by :meth:`from_packed`.

        Examples:
            >>> model = InferenceModel.from_pretrained("espnet/some_model")
            >>> text = model(audio_array)

            >>> model = InferenceModel.from_pretrained(
            ...     "espnet/some_model",
            ...     trust_user_code=True,
            ... )
        """
        artifacts = ModelDownloader().download_and_unpack(model_tag)
        if "inference_config" not in artifacts:
            raise RuntimeError(
                "downloaded model artifacts must include inference_config so "
                "InferenceModel can locate the pack_model() output directory."
            )
        inference_config_path = Path(artifacts["inference_config"])
        pack_dir = inference_config_path.parent.parent
        return cls.from_packed(pack_dir, trust_user_code=trust_user_code, device=device)

    @property
    def primary_input_key(self) -> str:
        """Return the single configured input key."""
        if isinstance(self.input_key, list):
            if len(self.input_key) != 1:
                raise RuntimeError(
                    "A scalar sample requires exactly one configured input_key."
                )
            return self.input_key[0]
        return self.input_key

    def _build_single_inputs(self, sample: Any) -> tuple[dict[str, Any], Any]:
        """Normalize one sample into backend keyword arguments.

        Called by :meth:`forward` before invoking the backend model. Mapping
        inputs are filtered down to the configured ``input_key`` fields, while
        scalar inputs are wrapped under :attr:`primary_input_key`.
        """
        if isinstance(sample, Mapping):
            keys = (
                self.input_key if isinstance(self.input_key, list) else [self.input_key]
            )
            inputs = {}
            for key in keys:
                if key not in sample:
                    raise KeyError(f"Input key '{key}' not found in sample.")
                inputs[key] = sample[key]
            return inputs, sample

        key = self.primary_input_key
        return {key: sample}, {key: sample}

    def _normalize_sample_for_runner(self, sample: Any) -> dict[str, Any]:
        """Normalize a publication sample to the mapping form expected by the runner."""
        _, data = self._build_single_inputs(sample)
        return data

    def forward(self, sample: Any, idx: Any = 0, **extra_kwargs: Any) -> Any:
        """Run inference for a single sample.

        This is the main execution method used by :meth:`__call__` and by
        :meth:`forward_batch`. It normalizes the sample to match the configured
        model input signature, calls the instantiated backend, and then applies
        the optional recipe ``output_fn``.

        Args:
            sample: Either a raw input value for single-input models or a
                mapping containing the configured input key(s).
            idx: Optional sample identifier forwarded to ``output_fn``.
            **extra_kwargs: Additional keyword arguments forwarded to the
                underlying model callable (e.g. ``beam_size`` for demo
                overrides).

        Returns:
            Any: Backend output, or the transformed output from ``output_fn``.

        Raises:
            KeyError: If a required input field is missing.
            RuntimeError: If a scalar sample is used with multiple input keys.

        Examples:
            >>> model = InferenceModel.from_packed("/path/to/packed_model")
            >>> result = model.forward(audio_array)

            >>> result = model.forward(
            ...     {"speech": audio_array, "text": "prompt"},
            ...     idx="utt-0001",
            ... )
        """
        data = self._normalize_sample_for_runner(sample)
        return self.runner_cls.forward(
            idx,
            dataset={idx: data},
            model=self.model,
            input_key=self.input_key,
            output_fn=self.output_fn,
            model_kwargs=extra_kwargs,
        )

    def __call__(self, sample: Any, idx: Any = 0, **extra_kwargs: Any) -> Any:
        """Alias for :meth:`forward`.

        This keeps the publication API convenient for interactive use, so
        callers can write ``model(sample)`` instead of ``model.forward(sample)``.
        """
        return self.forward(sample, idx=idx, **extra_kwargs)

    def forward_batch(
        self,
        samples: Sequence[Any],
        indices: Sequence[Any] | None = None,
    ) -> list[Any]:
        """Run inference for a batch of samples.

        This helper first tries the same batched execution path used by
        :class:`InferenceRunner`, so published models can benefit from recipe
        backends that already support batched inputs. If that batched call
        fails, or if the returned value does not preserve the one-result-per-
        sample contract of :class:`InferenceModel`, it falls back to
        per-sample :meth:`forward` calls.

        Args:
            samples: Sequence of raw inputs or sample mappings.
            indices: Optional per-sample identifiers forwarded to
                ``output_fn``. Defaults to ``range(len(samples))``.

        Returns:
            list[Any]: One output per sample.

        Raises:
            ValueError: If ``indices`` length does not match ``samples``.

        Notes:
            The output list preserves input order. An empty ``samples``
            sequence returns an empty list.

        Examples:
            >>> model = InferenceModel.from_packed("/path/to/packed_model")
            >>> results = model.forward_batch([audio_a, audio_b])
        """
        sample_list = list(samples)
        if indices is None:
            index_list = list(range(len(sample_list)))
        else:
            index_list = list(indices)
            if len(index_list) != len(sample_list):
                raise ValueError("indices must have the same length as samples.")
        if not sample_list:
            return []

        normalized_samples = [
            self._normalize_sample_for_runner(sample) for sample in sample_list
        ]

        batch_result = None
        if len(set(index_list)) == len(index_list):
            dataset = {
                idx: sample for idx, sample in zip(index_list, normalized_samples)
            }
            try:
                batch_result = self.runner_cls.forward(
                    index_list,
                    dataset=dataset,
                    model=self.model,
                    input_key=self.input_key,
                    output_fn=self.output_fn,
                )
            except (TypeError, NotImplementedError):
                logger.debug(
                    "Runner does not support batched inference;"
                    " falling back to per-sample.",
                    exc_info=True,
                )
                batch_result = None
            except RuntimeError as e:
                logger.warning(
                    "Batched inference raised RuntimeError (%s);"
                    " falling back to per-sample."
                    " This may indicate CUDA OOM or a shape mismatch.",
                    e,
                )
                batch_result = None

        if isinstance(batch_result, list) and len(batch_result) == len(sample_list):
            return batch_result
        if isinstance(batch_result, tuple) and len(batch_result) == len(sample_list):
            return list(batch_result)

        return [
            self.runner_cls.forward(
                idx,
                dataset={idx: data},
                model=self.model,
                input_key=self.input_key,
                output_fn=self.output_fn,
            )
            for data, idx in zip(normalized_samples, index_list)
        ]
