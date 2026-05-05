"""Demo UI asset classes, registry, and asset registration."""

from __future__ import annotations

import logging
from typing import Any

try:
    import gradio as gr
except ImportError:
    gr = None

logger = logging.getLogger(__name__)


class UIAsset:
    """Base class for one demo UI asset type.

    Demo configs refer to UI assets by ``ui.inputs[].type`` and
    ``ui.outputs[].type``. Each asset implementation is responsible for
    building the corresponding Gradio component and for converting raw UI
    values to or from the model-facing payload.

    Recipe-local demo extensions should subclass :class:`UIAsset` and register
    the subclass with :func:`register_asset`.
    """

    def check_gradio(self):
        """Return the imported ``gradio`` module.

        Returns:
            The imported ``gradio`` module object.

        Raises:
            ImportError: If ``gradio`` is not installed in the current environment.
        """
        if gr is None:
            raise ImportError(
                "gradio is required to build demo UI components. "
                "Install gradio before launching the demo."
            )
        return gr

    def build_input(self, spec: dict[str, Any]) -> Any:
        """Build one input component for a demo spec.

        Args:
            spec: One resolved entry from ``demo_config.ui.inputs``.

        Returns:
            Any: Gradio component object for the input.

        Raises:
            ValueError: If the asset does not support input components.
        """
        raise ValueError(
            f"{self.__class__.__name__} does not support input components."
        )

    def build_output(self, spec: dict[str, Any]) -> Any:
        """Build one output component for a demo spec.

        Args:
            spec: One resolved entry from ``demo_config.ui.outputs``.

        Returns:
            Any: Gradio component object for the output.

        Raises:
            ValueError: If the asset does not support output components.
        """
        raise ValueError(
            f"{self.__class__.__name__} does not support output components."
        )

    def normalize_input(self, value: Any, spec: dict[str, Any]) -> Any:
        """Normalize one raw UI input value before model inference.

        Args:
            value: Raw value returned by the Gradio input component.
            spec: Resolved input spec from ``demo_config.ui.inputs``.

        Returns:
            Any: Value forwarded to the inference model under ``spec["key"]``.

        Examples:
            >>> asset.normalize_input("hello", {"key": "text", "type": "text"})
            'hello'
        """
        _ = spec
        return value

    def format_output(self, value: Any, spec: dict[str, Any]) -> Any:
        """Format one model output value before returning it to the UI.

        Args:
            value: Model output value selected for one output spec.
            spec: Resolved output spec from ``demo_config.ui.outputs``.

        Returns:
            Any: Value returned to the Gradio output component.

        Examples:
            >>> asset.format_output("ok", {"key": "hyp", "type": "text"})
            'ok'
        """
        _ = spec
        return value


AssetRegistration = UIAsset | type[UIAsset]


class DefaultAudioUI(UIAsset):
    """Default Gradio audio asset.

    This is the built-in asset used by the standard demo template for speech
    input and optional audio output. Input values are requested from Gradio as
    NumPy arrays so they can be passed directly to the inference model after a
    small normalization step.
    """

    def build_input(self, spec: dict[str, Any]) -> Any:
        """Build a ``gr.Audio`` input component."""
        gradio_module = self.check_gradio()
        return gradio_module.Audio(label=spec["label"], type="numpy")

    def build_output(self, spec: dict[str, Any]) -> Any:
        """Build a ``gr.Audio`` output component."""
        gradio_module = self.check_gradio()
        return gradio_module.Audio(label=spec["label"])

    def normalize_input(self, value: Any, spec: dict[str, Any]) -> Any:
        """Extract and convert the NumPy array from a Gradio audio tuple."""
        _ = spec
        import numpy as np

        if isinstance(value, (list, tuple)) and len(value) == 2:
            _, audio = value
            if isinstance(audio, np.ndarray):
                return audio.astype(np.float32)
        return value


class DefaultTextUI(UIAsset):
    """Default Gradio text asset.

    This built-in asset covers plain text input and output fields used by the
    standard transcription demo and simple recipe-local text prompts.
    """

    def build_input(self, spec: dict[str, Any]) -> Any:
        """Build a ``gr.Textbox`` input component."""
        gradio_module = self.check_gradio()
        return gradio_module.Textbox(label=spec["label"])

    def build_output(self, spec: dict[str, Any]) -> Any:
        """Build a ``gr.Textbox`` output component."""
        gradio_module = self.check_gradio()
        return gradio_module.Textbox(label=spec["label"])


class UIAssetRegistry:
    """Registry of named UI asset definitions.

    The registry maps ``type`` strings from demo configs to concrete
    :class:`UIAsset` instances. A demo session clones the default registry and
    then optionally augments it with recipe-local asset registrations.
    """

    def __init__(self, assets: dict[str, UIAsset] | None = None) -> None:
        """Initialize with an optional pre-populated asset mapping."""
        self._assets = dict(assets or {})

    def register(
        self,
        name: str,
        asset: AssetRegistration,
        replace: bool = False,
    ) -> None:
        """Register one asset definition.

        Args:
            name: Config-facing asset name such as ``"audio"`` or
                ``"prompt_text"``.
            asset: Asset instance or asset subclass to register.
            replace: Whether an existing registration may be overwritten.

        Raises:
            ValueError: If ``name`` is already registered and ``replace`` is
                False.
            TypeError: If ``asset`` is neither a :class:`UIAsset` instance nor
                a :class:`UIAsset` subclass.
        """
        if not replace and name in self._assets:
            raise ValueError(f"UI asset already registered: {name}")
        if isinstance(asset, UIAsset):
            self._assets[name] = asset
        elif isinstance(asset, type) and issubclass(asset, UIAsset):
            self._assets[name] = asset()
        else:
            raise TypeError(
                "asset must be a UIAsset instance or UIAsset subclass, "
                f"but got: {type(asset)}"
            )

    def get(self, name: str) -> UIAsset:
        """Return one registered asset.

        Args:
            name: Asset type name from demo config.

        Returns:
            UIAsset: Registered asset instance.

        Raises:
            KeyError: If the asset type is unknown.
        """
        if name not in self._assets:
            raise KeyError(f"Unknown UI asset type: {name}")
        return self._assets[name]

    def clone(self) -> UIAssetRegistry:
        """Return a shallow copy for one demo session."""
        return UIAssetRegistry(self._assets)


_ACTIVE_REGISTRY: UIAssetRegistry | None = None
DEFAULT_UI_ASSETS: UIAssetRegistry = UIAssetRegistry()


def register_asset(
    name: str,
    asset: AssetRegistration,
    replace: bool = False,
) -> None:
    """Register a UI asset in the active recipe registry or the default one.

    Recipe-local ``ui_assets.py`` modules call this function to add custom
    asset implementations during demo load. When no recipe registry is active,
    the registration updates the shared default registry.

    Args:
        name: Config-facing asset name.
        asset: Asset instance or asset subclass.
        replace: Whether an existing registration may be overwritten.
    """
    registry = _ACTIVE_REGISTRY or DEFAULT_UI_ASSETS
    registry.register(name, asset, replace=replace)


register_asset("audio", DefaultAudioUI)
register_asset("text", DefaultTextUI)
