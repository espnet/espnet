"""Inference-model helpers for packaged ESPnet models."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from espnet_model_zoo.downloader import ModelDownloader
from omegaconf import DictConfig, ListConfig, OmegaConf

from espnet3.systems.base.inference_provider import InferenceProvider
from espnet3.systems.base.inference_runner import _load_output_fn
from espnet3.utils.config_utils import load_config_with_defaults


def _load_inference_config(
    config_path: Path,
    bundle_root: Path | None,
) -> DictConfig:
    """Load an inference config and bind ``recipe_dir`` to the bundle root."""
    config = load_config_with_defaults(str(config_path), resolve=False)
    if bundle_root is not None:
        config.recipe_dir = str(bundle_root)
    OmegaConf.resolve(config)
    return config


def _get_bundled_module_names(bundle_root: Path) -> set[str]:
    """Return importable top-level module names shipped in the bundle."""
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
    """Return whether the config references importable modules in the bundle."""
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


class InferenceModel:
    """User-facing inference wrapper for packaged ESPnet models.

    This class exposes a small direct-inference API around packaged ESPnet
    backends such as ``espnet2.bin.asr_inference.Speech2Text``. It is intended
    for use outside the stage runner, for example from a ``pixi shell`` session
    or a standalone Python environment after installing the model dependencies.

    The inference model can be built from:

    - a packaged model tag via :meth:`from_pretrained`
    - a packed model directory via :meth:`from_packed`

    When bundled user code is enabled, the packed bundle root is added to
    ``sys.path`` before backend construction. This is meant for explicitly
    trusted recipe code bundled with the published model.

    Args:
        model: Instantiated inference backend.
        input_key: Input field name or names expected by the backend.
        output_fn: Optional output function compatible with the recipe
            ``output_fn(data=..., model_output=..., idx=...)`` contract.

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
        """Initialize the inference model wrapper."""
        self.model = InferenceProvider.build_model(inference_config)
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
    ) -> "InferenceModel":
        """Build an inference model from a packed model directory.

        Args:
            pack_dir: Path to the output directory created by
                ``espnet3.utils.publish.pack_model()``. This directory must
                contain ``conf/inference.yaml`` and any files referenced by
                that config.
            trust_user_code: Set to ``True`` to allow importing bundled recipe
                code from the pack directory. Required when the inference
                config references modules shipped inside the bundle.

        Returns:
            InferenceModel: Inference model loaded from ``pack_model`` output.

        Raises:
            FileNotFoundError: If ``pack_dir/conf/inference.yaml`` is missing.
            ValueError: If the config requires bundled user code but
                ``trust_user_code`` is ``False``.

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

        return cls(inference_config)

    @classmethod
    def from_pretrained(
        cls,
        model_tag: str,
        trust_user_code: bool = False,
    ) -> "InferenceModel":
        """Download a packaged model and build an inference model from it.

        Args:
            model_tag: Pretrained model identifier understood by
                ``espnet_model_zoo``.
            trust_user_code: Forwarded to :meth:`from_packed`.

        Returns:
            InferenceModel: Downloaded inference model.

        Raises:
            RuntimeError: If the downloaded artifacts do not include an
                ``inference_config`` entry.

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
        inference_config_path = Path(artifacts["inference_config"]).resolve()
        pack_dir = inference_config_path.parent.parent
        return cls.from_packed(pack_dir, trust_user_code=trust_user_code)

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

    def _apply_output_fn(self, *, data: Any, model_output: Any, idx: Any) -> Any:
        """Apply the recipe output function when configured."""
        if self.output_fn is None:
            return model_output
        return self.output_fn(data=data, model_output=model_output, idx=idx)

    def forward(self, sample: Any, idx: Any = 0) -> Any:
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

    def __call__(self, sample: Any, idx: Any = 0) -> Any:
        """Alias for :meth:`forward`."""
        return self.forward(sample, idx=idx)

    def forward_batch(
        self,
        samples: Sequence[Any],
        indices: Sequence[Any] | None = None,
    ) -> list[Any]:
        """Run inference for a batch of samples.

        Args:
            samples: Sequence of raw inputs or sample mappings.
            indices: Optional per-sample identifiers forwarded to
                ``output_fn``. Defaults to ``range(len(samples))``.

        Returns:
            list[Any]: One output per sample.

        Raises:
            ValueError: If ``indices`` length does not match ``samples``.

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
        return [
            self.forward(sample, idx=idx)
            for sample, idx in zip(sample_list, index_list)
        ]
