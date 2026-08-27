"""Demo UI asset classes, registry, and built-in asset registration."""

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
    building the corresponding Gradio component.

    ESPnet3 developers can add new built-in assets by subclassing
    :class:`UIAsset` and registering the subclass in
    :data:`DEFAULT_UI_ASSETS` in this module.
    """

    def __init__(self) -> None:
        """Initialize one UI asset and cache the imported ``gradio`` module."""
        self.gradio_module = self._load_gradio()

    def _load_gradio(self):
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


AssetRegistration = UIAsset | type[UIAsset]


class DefaultAudioUI(UIAsset):
    """Default Gradio audio asset.

    This is the built-in asset used by the standard demo template for speech
    input and optional audio output.
    """

    def build_input(self, spec: dict[str, Any]) -> Any:
        """Build a ``gr.Audio`` input component."""
        return self.gradio_module.Audio(label=spec["label"])

    def build_output(self, spec: dict[str, Any]) -> Any:
        """Build a ``gr.Audio`` output component."""
        return self.gradio_module.Audio(label=spec["label"])


class DefaultTextUI(UIAsset):
    """Default Gradio text asset.

    This built-in asset covers plain text input and output fields used by the
    standard transcription demo and simple recipe-local text prompts.
    """

    def build_input(self, spec: dict[str, Any]) -> Any:
        """Build a ``gr.Textbox`` input component."""
        return self.gradio_module.Textbox(label=spec["label"])

    def build_output(self, spec: dict[str, Any]) -> Any:
        """Build a ``gr.Textbox`` output component."""
        return self.gradio_module.Textbox(label=spec["label"])


class UIAssetRegistry:
    """Registry of named UI asset definitions.

    The registry maps ``type`` strings from demo configs to concrete
    :class:`UIAsset` instances. A demo session clones the default registry so
    built-in asset definitions stay isolated from per-session state.
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


DEFAULT_UI_ASSETS: UIAssetRegistry = UIAssetRegistry()
DEFAULT_UI_ASSETS.register("audio", DefaultAudioUI)
DEFAULT_UI_ASSETS.register("text", DefaultTextUI)
