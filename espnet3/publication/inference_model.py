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

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from espnet_model_zoo.downloader import ModelDownloader
from hydra.utils import get_class
from omegaconf import DictConfig, ListConfig, OmegaConf

from espnet3.publication.demo.config import load_demo_config
from espnet3.systems.base.inference_provider import InferenceProvider
from espnet3.systems.base.inference_runner import InferenceRunner, _load_output_fn
from espnet3.utils.config_utils import load_config_with_defaults


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


def _resolve_runtime_device(provider_cls: type, inference_config: DictConfig) -> str | None:
    """Resolve the runtime device string for one inference model."""
    resolve_device = getattr(provider_cls, "_resolve_device", None)
    if callable(resolve_device):
        return resolve_device(inference_config)
    device = getattr(inference_config, "device", None)
    if device in (None, ""):
        return None
    return str(device)


class InferenceModel:
    """User-facing inference wrapper for packaged ESPnet models.

    This class is the public runtime API for a bundle produced by
    ``espnet3.utils.publish.pack_model()``. It sits on the publication side of
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

    def __init__(
        self,
        inference_config: DictConfig,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the inference model from a resolved inference config.

        Called by :meth:`from_packed` and :meth:`from_pretrained` after bundle
        discovery and trust checks are complete. This constructor instantiates
        the backend model, normalizes ``input_key`` into either a string or a
        list of strings, and loads the optional recipe ``output_fn``.

        Args:
            inference_config: Resolved inference config loaded from the packed
                bundle.
            config_overrides: Optional runtime overrides applied before
                backend construction, such as ``{"device": "cuda:0"}`` or
                ``{"model": {"beam_size": 1}}``.
        """
        if config_overrides:
            inference_config = OmegaConf.merge(
                inference_config,
                OmegaConf.create(config_overrides),
            )
        provider_target = getattr(
            getattr(inference_config, "provider", None), "_target_", None
        )
        provider_cls = (
            get_class(provider_target) if provider_target else InferenceProvider
        )
        runner_target = getattr(
            getattr(inference_config, "runner", None), "_target_", None
        )
        self.runner_cls = get_class(runner_target) if runner_target else InferenceRunner
        self.resolved_device = _resolve_runtime_device(provider_cls, inference_config)

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
        config_overrides: dict[str, Any] | None = None,
    ) -> "InferenceModel":
        """Build an inference model from a packed model directory.

        This is the main entry point for local publication bundles. It is
        called by external users, CI checks, and any runtime that already has
        an unpacked ``pack_model()`` output directory. The method validates the
        bundle layout, loads ``meta.yaml``, finds ``yaml_files.inference_config``,
        resolves the inference config against the bundle root, and then
        instantiates :class:`InferenceModel`.

        If the config references modules bundled alongside the model, the load
        is blocked unless ``trust_user_code=True``. In that case the bundle root
        is inserted into ``sys.path`` and the config is reloaded so import-based
        objects resolve against the newly trusted code.

        Args:
            pack_dir: Path to the output directory created by
                ``espnet3.utils.publish.pack_model()``. This directory must
                contain ``conf/inference.yaml`` and any files referenced by
                that config.
            trust_user_code: Set to ``True`` to allow importing bundled recipe
                code from the pack directory. Required when the inference
                config references modules shipped inside the bundle.
            config_overrides: Optional top-level runtime overrides applied
                before backend construction.

        Returns:
            InferenceModel: Inference model loaded from ``pack_model`` output.

        Raises:
            FileNotFoundError: If the bundle directory, ``meta.yaml``, or the
                referenced inference config is missing.
            ValueError: If the config requires bundled user code but
                ``trust_user_code`` is ``False``.

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

        return cls(inference_config, config_overrides=config_overrides)

    @classmethod
    def from_pretrained(
        cls,
        model_tag: str,
        trust_user_code: bool = False,
        config_overrides: dict[str, Any] | None = None,
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
            config_overrides: Forwarded to :meth:`from_packed`.

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
        return cls.from_packed(
            pack_dir,
            trust_user_code=trust_user_code,
            config_overrides=config_overrides,
        )

    @classmethod
    def from_dir_or_tag(
        cls,
        dir_or_tag: str | Path,
        trust_user_code: bool = False,
        base_dir: str | Path | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> "InferenceModel":
        """Build an inference model from a local pack dir or model tag.

        This helper supports configs that want one field for both local
        development and remote deployment. It first checks whether
        ``dir_or_tag`` resolves to an existing local directory. If so, it
        delegates to :meth:`from_packed`. Otherwise, it treats the value as a
        pretrained model tag and delegates to :meth:`from_pretrained`.

        Args:
            dir_or_tag: Local packed-model directory or pretrained model tag.
            trust_user_code: Forwarded to :meth:`from_packed` or
                :meth:`from_pretrained`.
            base_dir: Optional base directory used to resolve relative local
                paths before checking whether they exist.
            config_overrides: Forwarded to :meth:`from_packed` or
                :meth:`from_pretrained`.

        Returns:
            InferenceModel: Inference model loaded from the matching source.

        Raises:
            FileNotFoundError: If ``dir_or_tag`` resolves to an existing local
                path that is not a directory.

        Notes:
            Path detection is existence-based. When no local directory matches,
            the value is passed through to :meth:`from_pretrained`.

        Examples:
            >>> model = InferenceModel.from_dir_or_tag("/path/to/model_pack")
            >>> result = model(audio_array)

            >>> model = InferenceModel.from_dir_or_tag(
            ...     "espnet/some_model",
            ...     trust_user_code=True,
            ... )
        """
        raw_ref = str(dir_or_tag)
        candidate = Path(raw_ref).expanduser()
        candidates = []
        if base_dir is not None and not candidate.is_absolute():
            candidates.append((Path(base_dir).resolve() / candidate).resolve())
        if candidate.is_absolute():
            candidates.append(candidate.resolve())
        else:
            candidates.append(candidate.resolve())

        for path in candidates:
            if not path.exists():
                continue
            if not path.is_dir():
                raise FileNotFoundError(
                    "dir_or_tag resolved to a filesystem path, but it is not "
                    f"a directory: {path}"
                )
            return cls.from_packed(
                path,
                trust_user_code=trust_user_code,
                config_overrides=config_overrides,
            )

        return cls.from_pretrained(
            raw_ref,
            trust_user_code=trust_user_code,
            config_overrides=config_overrides,
        )

    @classmethod
    def from_demo(
        cls,
        demo_dir: str | Path,
        trust_user_code: bool = False,
        demo_config_path: str | Path | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> "InferenceModel":
        """Build an inference model from a demo directory created by pack_demo().

        This entry point loads the packed demo config from ``demo_dir``, reads
        ``model.dir_or_tag``, and delegates to :meth:`from_dir_or_tag`.

        Args:
            demo_dir: Path to the directory created by ``pack_demo()``. This
                directory must contain one packed demo config with
                ``model.dir_or_tag``.
            trust_user_code: Set to ``True`` to allow importing bundled recipe
                code from the selected model source. This flag is combined with
                the packed demo config ``model.trust_user_code``.
            demo_config_path: Optional explicit packed demo config path.
                Relative paths are resolved from ``demo_dir``.
            config_overrides: Forwarded to :meth:`from_dir_or_tag`.

        Returns:
            InferenceModel: Inference model loaded from the demo config.

        Raises:
            FileNotFoundError: If the demo directory or packed demo config is
                missing.
            ValueError: If the packed demo config does not define
                ``model.dir_or_tag``.

        Examples:
            >>> model = InferenceModel.from_demo("egs3/mini_an4/asr/demo")
            >>> result = model(audio_array)

            >>> model = InferenceModel.from_demo(
            ...     "egs3/mini_an4/asr/demo",
            ...     trust_user_code=True,
            ... )
        """
        demo_root = Path(demo_dir).resolve()
        _, demo_cfg = load_demo_config(demo_root, demo_config_path)
        model_cfg = getattr(demo_cfg, "model", None)
        dir_or_tag = getattr(model_cfg, "dir_or_tag", None) if model_cfg else None
        if not dir_or_tag:
            raise ValueError(
                "packed demo config must contain model.dir_or_tag, "
                f"but it was missing under: {demo_root}"
            )
        model_trust_user_code = bool(
            getattr(model_cfg, "trust_user_code", False) if model_cfg else False
        )
        return cls.from_dir_or_tag(
            dir_or_tag,
            trust_user_code=trust_user_code or model_trust_user_code,
            base_dir=demo_root,
            config_overrides=config_overrides,
        )

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
                runner (e.g. ``beam_size`` for demo overrides).

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
            **extra_kwargs,
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
            except (TypeError, NotImplementedError, RuntimeError):
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
