"""Inference session helpers for packaged ESPnet models."""

from __future__ import annotations

import logging
import os
import sys
from importlib import import_module, invalidate_caches
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

from espnet3.systems.base.inference_provider import InferenceProvider
from espnet3.systems.base.inference_runner import _load_output_fn
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)

_DEFAULT_USER_CODE_PATHS = ("src",)
_DEFAULT_DOWNLOADER_CLASS = "espnet_model_zoo.downloader.ModelDownloader"
_DEFAULT_INFERENCE_CONFIG_KEYS = ("inference_config",)


def _import_object(path: str) -> Any:
    """Import an object from a dotted path."""
    module_path, _, name = path.rpartition(".")
    if not module_path:
        raise ValueError(f"Expected dotted import path, got: {path!r}")
    return getattr(import_module(module_path), name)


def _to_dict_config(config: Mapping[str, Any] | DictConfig) -> DictConfig:
    """Return a detached ``DictConfig`` for the given config-like object."""
    if isinstance(config, DictConfig):
        return OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    return OmegaConf.create(dict(config))


def _iter_artifact_paths(value: Any):
    """Yield path-like artifact values from metadata mappings."""
    if isinstance(value, (str, Path)):
        yield Path(value)
        return
    if isinstance(value, (list, tuple, ListConfig)):
        for item in value:
            yield from _iter_artifact_paths(item)


def _find_bundle_root(artifacts: Mapping[str, Any]) -> Path | None:
    """Infer the unpacked model bundle root from artifact paths."""
    candidates: list[Path] = []
    for value in artifacts.values():
        for path in _iter_artifact_paths(value):
            resolved = path.expanduser().absolute()
            for parent in (resolved.parent, *resolved.parents):
                if (parent / "meta.yaml").is_file():
                    return parent
            candidates.append(resolved.parent)

    if not candidates:
        return None

    common = Path(candidates[0])
    for candidate in candidates[1:]:
        common = Path(
            os.path.commonpath([common.as_posix(), candidate.as_posix()])
        ).absolute()
    return common


def _load_bundle_metadata(bundle_root: Path | None) -> dict[str, Any]:
    """Load ``meta.yaml`` from the unpacked model bundle when present."""
    if bundle_root is None:
        return {}
    meta_path = bundle_root / "meta.yaml"
    if not meta_path.is_file():
        return {}
    with meta_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Expected mapping metadata in {meta_path}")
    return loaded


def _resolve_inference_config_path(
    artifacts: Mapping[str, Any],
    bundle_root: Path | None,
    metadata: Mapping[str, Any],
) -> Path | None:
    """Resolve the primary inference-config path for a bundle."""
    config_key = metadata.get("inference_config_key")
    if isinstance(config_key, str) and config_key in artifacts:
        return Path(artifacts[config_key]).resolve()

    for key in _DEFAULT_INFERENCE_CONFIG_KEYS:
        if key in artifacts:
            return Path(artifacts[key]).resolve()

    if bundle_root is not None:
        candidate = bundle_root / "conf" / "inference.yaml"
        if candidate.is_file():
            return candidate.resolve()

    return None


def _resolve_user_code_entries(
    bundle_root: Path,
    metadata: Mapping[str, Any],
    user_code_paths: Sequence[str] | None,
) -> list[Path]:
    """Resolve bundle import roots used for recipe-defined user code."""
    manifest_paths = metadata.get("user_code_paths")
    if isinstance(manifest_paths, str):
        entries = [manifest_paths]
    elif isinstance(manifest_paths, (list, tuple)):
        entries = [str(item) for item in manifest_paths]
    else:
        entries = list(user_code_paths or _DEFAULT_USER_CODE_PATHS)

    resolved = [bundle_root]
    for entry in entries:
        candidate = (bundle_root / entry).resolve()
        if candidate.exists():
            resolved.append(candidate)

    unique: list[Path] = []
    for path in resolved:
        if path not in unique:
            unique.append(path)
    return unique


def _enable_user_code_paths(import_paths: Sequence[Path]) -> None:
    """Prepend user-code import roots to ``sys.path`` when needed."""
    module_names: set[str] = set()
    for root in import_paths:
        if not root.exists() or not root.is_dir():
            continue
        module_names.add(root.name)
        for child in root.iterdir():
            if child.name == "__pycache__":
                continue
            if child.is_file() and child.suffix == ".py" and child.stem != "__init__":
                module_names.add(child.stem)
            elif child.is_dir() and (child / "__init__.py").is_file():
                module_names.add(child.name)

    for module_name in sorted(module_names, key=len, reverse=True):
        for loaded_name in list(sys.modules):
            if loaded_name == module_name or loaded_name.startswith(f"{module_name}."):
                sys.modules.pop(loaded_name, None)

    for path in reversed([str(p) for p in import_paths]):
        if path not in sys.path:
            sys.path.insert(0, path)
    invalidate_caches()


def _uses_user_code(value: Any, import_roots: Sequence[str]) -> bool:
    """Return whether a config value references trusted local user code."""
    if isinstance(value, str):
        return any(value.startswith(f"{root}.") for root in import_roots)
    if isinstance(value, Mapping):
        return any(_uses_user_code(v, import_roots) for v in value.values())
    if isinstance(value, (list, tuple, ListConfig)):
        return any(_uses_user_code(v, import_roots) for v in value)
    return False


def _load_inference_config(
    config_path: Path,
    *,
    bundle_root: Path | None,
) -> DictConfig:
    """Load an inference config and bind ``recipe_dir`` to the bundle root."""
    config = load_config_with_defaults(str(config_path), resolve=False)
    if bundle_root is not None:
        config.recipe_dir = str(bundle_root)
    OmegaConf.resolve(config)
    return config


def _should_enable_user_code(
    *,
    explicit_enable: bool | None,
    trust_user_code: bool,
    config: DictConfig | None,
    metadata: Mapping[str, Any],
    user_code_paths: Sequence[str] | None,
) -> bool:
    """Decide whether bundled user code should be activated."""
    entries = metadata.get("user_code_paths")
    if isinstance(entries, str):
        import_roots = [Path(entries).name]
    elif isinstance(entries, (list, tuple)):
        import_roots = [Path(str(entry)).name for entry in entries]
    else:
        import_roots = [
            Path(entry).name for entry in (user_code_paths or _DEFAULT_USER_CODE_PATHS)
        ]

    config_requires_user_code = False
    if config is not None:
        plain = OmegaConf.to_container(config, resolve=False)
        config_requires_user_code = _uses_user_code(plain, import_roots)

    if explicit_enable is True:
        if not trust_user_code:
            raise ValueError("enable_user_code=True requires trust_user_code=True.")
        return True

    if config_requires_user_code:
        if not trust_user_code:
            raise ValueError(
                "This inference config references bundled user code. "
                "Set trust_user_code=True to allow imports from the published bundle."
            )
        return True

    return False


class InferenceSession:
    """User-facing inference wrapper for packaged ESPnet models.

    This class exposes a small direct-inference API around packaged ESPnet
    backends such as ``espnet2.bin.asr_inference.Speech2Text``. It is intended
    for use outside the stage runner, for example from a ``pixi shell`` session
    or a standalone Python environment after installing the model dependencies.

    The session can be built from:

    - an inference config via :meth:`from_config`
    - a packaged model tag via :meth:`from_pretrained`
    - already resolved artifact paths via :meth:`from_artifacts`

    When ``enable_user_code=True``, the bundle root and recipe-managed import
    directories such as ``src/`` are added to ``sys.path`` before backend
    construction. This is meant for explicitly trusted recipe code bundled with
    the published model.

    Args:
        model: Instantiated inference backend.
        input_key: Input field name or names expected by the backend.
        output_fn_path: Optional dotted output-function path compatible with the
            recipe ``output_fn(data=..., model_output=..., idx=...)`` contract.
        output_fn: Optional already imported output function. Use this instead
            of ``output_fn_path`` when the caller already has the function.
        prefer_model_batch: Whether :meth:`forward_batch` should try a single
            batched backend call before falling back to per-sample execution.
        fallback_to_single_on_batch_error: Whether :meth:`forward_batch` should
            transparently fall back to per-sample execution after a batched call
            fails.
        bundle_root: Optional unpacked model bundle root.
        bundle_metadata: Optional metadata loaded from ``meta.yaml``.
        artifacts: Optional resolved artifact mapping used to build the backend.

    Raises:
        ValueError: If both ``output_fn_path`` and ``output_fn`` are provided.

    Notes:
        This wrapper does not require dataset objects. Single-sample inference
        accepts either a raw value for single-input models or a mapping that
        contains the configured ``input_key`` fields.

    Examples:
        >>> session = InferenceSession.from_pretrained(
        ...     "espnet/some_model",
        ...     trust_user_code=True,
        ... )
        >>> result = session(audio_array)
        >>> batch = session.forward_batch([audio_a, audio_b])
    """

    def __init__(
        self,
        model: Any,
        *,
        input_key: str | Sequence[str] = "speech",
        output_fn_path: str | None = None,
        output_fn=None,
        prefer_model_batch: bool = False,
        fallback_to_single_on_batch_error: bool = True,
        bundle_root: Path | None = None,
        bundle_metadata: Mapping[str, Any] | None = None,
        artifacts: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the inference session."""
        if output_fn_path is not None and output_fn is not None:
            raise ValueError("Specify only one of output_fn_path or output_fn.")

        self.model = model
        self.input_key = (
            list(input_key)
            if isinstance(input_key, (list, tuple, ListConfig))
            else input_key
        )
        self.output_fn_path = output_fn_path
        self.output_fn = output_fn or (
            _load_output_fn(output_fn_path) if output_fn_path else None
        )
        self.prefer_model_batch = prefer_model_batch
        self.fallback_to_single_on_batch_error = fallback_to_single_on_batch_error
        self.bundle_root = bundle_root
        self.bundle_metadata = dict(bundle_metadata or {})
        self.artifacts = dict(artifacts or {})

    @classmethod
    def from_config(
        cls,
        inference_config: Mapping[str, Any] | DictConfig,
        *,
        enable_user_code: bool | None = None,
        trust_user_code: bool = False,
        user_code_paths: Sequence[str] | None = None,
        prefer_model_batch: bool = False,
        fallback_to_single_on_batch_error: bool = True,
    ) -> "InferenceSession":
        """Build a session directly from an ESPnet3 inference config.

        Args:
            inference_config: Inference config containing at least ``model`` and
                optionally ``input_key``, ``output_fn``, and ``recipe_dir``.
            enable_user_code: Whether to add recipe-local user code such as
                ``src/`` to ``sys.path`` before backend construction.
            trust_user_code: Explicit opt-in required when
                ``enable_user_code=True``.
            user_code_paths: Relative import roots under ``recipe_dir`` to add
                when user code is enabled. Defaults to ``("src",)``.
            prefer_model_batch: Whether batched backend execution should be
                attempted by default in :meth:`forward_batch`.
            fallback_to_single_on_batch_error: Whether failed batched calls
                should fall back to per-sample execution.

        Returns:
            InferenceSession: Config-backed inference session.

        Raises:
            RuntimeError: If user code was requested but ``recipe_dir`` is
                unavailable.
            ValueError: If untrusted user code execution was requested.

        Notes:
            Backend construction reuses the same device-resolution logic as the
            stage inference provider.
        """
        config = _to_dict_config(inference_config)
        bundle_root = None

        recipe_dir = getattr(config, "recipe_dir", None)
        if recipe_dir not in (None, ""):
            bundle_root = Path(recipe_dir).resolve()
        if _should_enable_user_code(
            explicit_enable=enable_user_code,
            trust_user_code=trust_user_code,
            config=config,
            metadata={},
            user_code_paths=user_code_paths,
        ):
            if bundle_root is None:
                raise RuntimeError(
                    "from_config() requires recipe_dir when bundled user code is used."
                )
            import_paths = _resolve_user_code_entries(
                bundle_root=bundle_root,
                metadata={},
                user_code_paths=user_code_paths,
            )
            _enable_user_code_paths(import_paths)

        model = InferenceProvider.build_model(config)
        return cls(
            model,
            input_key=getattr(config, "input_key", "speech"),
            output_fn_path=getattr(config, "output_fn", None),
            prefer_model_batch=prefer_model_batch,
            fallback_to_single_on_batch_error=fallback_to_single_on_batch_error,
            bundle_root=bundle_root,
        )

    @classmethod
    def from_artifacts(
        cls,
        artifacts: Mapping[str, Any],
        *,
        backend_class: str | type | None = None,
        input_key: str | Sequence[str] | None = None,
        output_fn_path: str | None = None,
        enable_user_code: bool | None = None,
        trust_user_code: bool = False,
        user_code_paths: Sequence[str] | None = None,
        prefer_model_batch: bool = False,
        fallback_to_single_on_batch_error: bool = True,
        **backend_kwargs: Any,
    ) -> "InferenceSession":
        """Build a session from already resolved model artifact paths.

        Args:
            artifacts: Mapping returned by a model downloader or unpacker.
            backend_class: Optional backend class override. When omitted, this
                method first looks for an inference config in the bundle and
                instantiates ``config.model`` from there.
            input_key: Optional input field override. When omitted and an
                inference config is available, ``config.input_key`` is used.
            output_fn_path: Optional dotted recipe output-function path.
            enable_user_code: Whether to activate bundled user code before
                backend construction.
            trust_user_code: Explicit opt-in required when
                ``enable_user_code=True``.
            user_code_paths: Relative import roots under the bundle root.
            prefer_model_batch: Whether :meth:`forward_batch` should try a
                single batched backend call first.
            fallback_to_single_on_batch_error: Whether failed batched calls
                should fall back to per-sample execution.
            **backend_kwargs: Additional constructor kwargs forwarded to the
                backend class.

        Returns:
            InferenceSession: Artifact-backed inference session.

        Raises:
            RuntimeError: If user code was requested but the bundle root could
                not be inferred.
            ValueError: If untrusted user code execution was requested.
        """
        bundle_root = _find_bundle_root(artifacts)
        metadata = _load_bundle_metadata(bundle_root)

        inference_config_path = _resolve_inference_config_path(
            artifacts=artifacts,
            bundle_root=bundle_root,
            metadata=metadata,
        )
        inference_config = None
        if inference_config_path is not None:
            inference_config = _load_inference_config(
                inference_config_path,
                bundle_root=bundle_root,
            )

        if _should_enable_user_code(
            explicit_enable=enable_user_code,
            trust_user_code=trust_user_code,
            config=inference_config,
            metadata=metadata,
            user_code_paths=user_code_paths,
        ):
            if bundle_root is None:
                raise RuntimeError(
                    "Could not infer bundle_root from artifacts for user code."
                )
            import_paths = _resolve_user_code_entries(
                bundle_root=bundle_root,
                metadata=metadata,
                user_code_paths=user_code_paths,
            )
            _enable_user_code_paths(import_paths)
            if inference_config_path is not None:
                inference_config = _load_inference_config(
                    inference_config_path,
                    bundle_root=bundle_root,
                )

        if inference_config is not None and backend_class is None:
            model = InferenceProvider.build_model(inference_config)
            resolved_input_key = (
                input_key
                if input_key is not None
                else getattr(inference_config, "input_key", "speech")
            )
            resolved_output_fn_path = (
                output_fn_path
                if output_fn_path is not None
                else getattr(inference_config, "output_fn", None)
            )
        else:
            if backend_class is None:
                raise RuntimeError(
                    "Could not determine backend construction. "
                    "Provide an inference_config in the bundle or pass backend_class."
                )
            backend_cls = (
                _import_object(backend_class)
                if isinstance(backend_class, str)
                else backend_class
            )
            init_kwargs = dict(artifacts)
            init_kwargs.update(backend_kwargs)
            if hasattr(backend_cls, "from_pretrained"):
                model = backend_cls.from_pretrained(model_tag=None, **dict(init_kwargs))
            else:
                model = backend_cls(**dict(init_kwargs))
            resolved_input_key = input_key if input_key is not None else "speech"
            resolved_output_fn_path = output_fn_path

        return cls(
            model,
            input_key=resolved_input_key,
            output_fn_path=resolved_output_fn_path,
            prefer_model_batch=prefer_model_batch,
            fallback_to_single_on_batch_error=fallback_to_single_on_batch_error,
            bundle_root=bundle_root,
            bundle_metadata=metadata,
            artifacts=artifacts,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_tag: str,
        *,
        backend_class: str | type | None = None,
        downloader_class: str | type = _DEFAULT_DOWNLOADER_CLASS,
        input_key: str | Sequence[str] | None = None,
        output_fn_path: str | None = None,
        enable_user_code: bool | None = None,
        trust_user_code: bool = False,
        user_code_paths: Sequence[str] | None = None,
        prefer_model_batch: bool = False,
        fallback_to_single_on_batch_error: bool = True,
        **backend_kwargs: Any,
    ) -> "InferenceSession":
        """Download a packaged model and build an inference session from it.

        Args:
            model_tag: Pretrained model identifier understood by
                ``espnet_model_zoo``.
            backend_class: Optional backend class override. When omitted, the
                published inference config is used to instantiate ``model``.
            downloader_class: Dotted model-downloader class path or class
                object. Defaults to ``espnet_model_zoo.downloader.ModelDownloader``.
            input_key: Optional input field override.
            output_fn_path: Optional dotted recipe output-function path.
            enable_user_code: Whether to activate bundled user code such as
                ``src/`` before backend construction.
            trust_user_code: Explicit opt-in required when
                ``enable_user_code=True``.
            user_code_paths: Relative import roots under the unpacked bundle.
            prefer_model_batch: Whether :meth:`forward_batch` should try a
                single batched backend call first.
            fallback_to_single_on_batch_error: Whether failed batched calls
                should fall back to per-sample execution.
            **backend_kwargs: Additional constructor kwargs forwarded to the
                backend class.

        Returns:
            InferenceSession: Downloaded inference session.

        Raises:
            Any exception raised by the configured downloader or backend
            constructor.

        Examples:
            >>> session = InferenceSession.from_pretrained(
            ...     "espnet/some_model",
            ...     trust_user_code=True,
            ... )
            >>> text = session(audio_array)
        """
        backend_cls = (
            _import_object(backend_class)
            if isinstance(backend_class, str)
            else backend_class
        )

        if backend_cls is not None and enable_user_code is False:
            if not hasattr(backend_cls, "from_pretrained"):
                raise RuntimeError(
                    "backend_class must provide from_pretrained() when "
                    "enable_user_code is disabled."
                )
            model = backend_cls.from_pretrained(
                model_tag=model_tag,
                **backend_kwargs,
            )
            return cls(
                model,
                input_key=input_key if input_key is not None else "speech",
                output_fn_path=output_fn_path,
                prefer_model_batch=prefer_model_batch,
                fallback_to_single_on_batch_error=fallback_to_single_on_batch_error,
            )

        downloader_cls = (
            _import_object(downloader_class)
            if isinstance(downloader_class, str)
            else downloader_class
        )
        downloader = downloader_cls()
        artifacts = downloader.download_and_unpack(model_tag)
        return cls.from_artifacts(
            artifacts,
            backend_class=backend_cls,
            input_key=input_key,
            output_fn_path=output_fn_path,
            enable_user_code=enable_user_code,
            trust_user_code=trust_user_code,
            user_code_paths=user_code_paths,
            prefer_model_batch=prefer_model_batch,
            fallback_to_single_on_batch_error=fallback_to_single_on_batch_error,
            **backend_kwargs,
        )

    @property
    def primary_input_key(self) -> str:
        """Return the single configured input key.

        Raises:
            RuntimeError: If the session expects multiple input keys.
        """
        if isinstance(self.input_key, list):
            if len(self.input_key) != 1:
                raise RuntimeError(
                    "A scalar sample requires exactly one configured input_key."
                )
            return self.input_key[0]
        return self.input_key

    def _build_single_inputs(self, sample: Any) -> tuple[dict[str, Any], Any]:
        """Normalize a single input sample into backend kwargs."""
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

    def _build_batch_inputs(
        self, samples: Sequence[Any]
    ) -> tuple[dict[str, list[Any]], Any]:
        """Normalize a batch of samples into backend kwargs."""
        if not samples:
            raise ValueError("forward_batch requires at least one sample.")

        if all(isinstance(sample, Mapping) for sample in samples):
            keys = (
                self.input_key if isinstance(self.input_key, list) else [self.input_key]
            )
            inputs = {}
            for key in keys:
                values = []
                for sample in samples:
                    if key not in sample:
                        raise KeyError(f"Input key '{key}' not found in sample.")
                    values.append(sample[key])
                inputs[key] = values
            return inputs, list(samples)

        key = self.primary_input_key
        return {key: list(samples)}, [{key: sample} for sample in samples]

    def _apply_output_fn(self, *, data: Any, model_output: Any, idx: Any) -> Any:
        """Apply the recipe output function when configured."""
        if self.output_fn is None:
            return model_output
        return self.output_fn(data=data, model_output=model_output, idx=idx)

    def forward(self, sample: Any, *, idx: Any = 0) -> Any:
        """Run inference for a single sample.

        Args:
            sample: Either a raw input value for single-input models or a
                mapping containing the configured input key(s).
            idx: Optional sample identifier forwarded to ``output_fn``.

        Returns:
            Any: Backend output, or the transformed output from ``output_fn``.

        Raises:
            KeyError: If a required input field is missing.
            RuntimeError: If a scalar sample is used with multiple input keys.
        """
        inputs, data = self._build_single_inputs(sample)
        model_output = self.model(**inputs)
        return self._apply_output_fn(data=data, model_output=model_output, idx=idx)

    def __call__(self, sample: Any, *, idx: Any = 0) -> Any:
        """Alias for :meth:`forward`."""
        return self.forward(sample, idx=idx)

    def forward_batch(
        self,
        samples: Sequence[Any],
        *,
        indices: Sequence[Any] | None = None,
        use_model_batch: bool | None = None,
        fallback_to_single_on_error: bool | None = None,
    ) -> list[Any]:
        """Run inference for a batch of samples.

        Args:
            samples: Sequence of raw inputs or sample mappings.
            indices: Optional per-sample identifiers forwarded to ``output_fn``
                during fallback execution.
            use_model_batch: Whether to try one batched backend call. When
                ``None``, uses ``prefer_model_batch`` configured on the session.
            fallback_to_single_on_error: Whether to fall back to per-sample
                execution after a failed batched call. When ``None``, uses the
                session default.

        Returns:
            list[Any]: One output per sample.

        Raises:
            RuntimeError: If batched backend execution fails and fallback is
                disabled.
            ValueError: If ``indices`` length does not match ``samples``.
        """
        sample_list = list(samples)
        if indices is None:
            index_list = list(range(len(sample_list)))
        else:
            index_list = list(indices)
            if len(index_list) != len(sample_list):
                raise ValueError("indices must have the same length as samples.")

        prefer_batch = (
            self.prefer_model_batch if use_model_batch is None else use_model_batch
        )
        allow_fallback = (
            self.fallback_to_single_on_batch_error
            if fallback_to_single_on_error is None
            else fallback_to_single_on_error
        )

        if prefer_batch:
            batch_inputs, batch_data = self._build_batch_inputs(sample_list)
            try:
                model_output = self.model(**batch_inputs)
                batch_result = self._apply_output_fn(
                    data=batch_data,
                    model_output=model_output,
                    idx=index_list,
                )
                if isinstance(batch_result, list):
                    if len(batch_result) != len(sample_list):
                        raise RuntimeError(
                            "Batched inference returned the wrong number of outputs: "
                            f"expected {len(sample_list)}, got {len(batch_result)}."
                        )
                    return batch_result
                if len(sample_list) != 1:
                    raise RuntimeError(
                        "Batched inference returned the wrong number of outputs: "
                        f"expected {len(sample_list)}, got 1."
                    )
                return [batch_result]
            except Exception as exc:  # noqa: BLE001
                if not allow_fallback:
                    raise RuntimeError(f"Batched inference failed: {exc}") from exc
                logger.info(
                    "Batched inference failed; falling back to per-sample execution."
                )

        return [
            self.forward(sample, idx=idx)
            for sample, idx in zip(sample_list, index_list)
        ]
