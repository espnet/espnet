"""ASR demo UI builder."""

from __future__ import annotations


def build_ui_default():
    """Return the default ASR demo UI configuration.

    Args:
        None

    Returns:
        dict: A UI config dictionary suitable for ``demo.yaml`` (``ui`` section).

    Example:
        >>> cfg = build_ui_default()
        >>> [c[\"name\"] for c in cfg[\"inputs\"]]
        ['speech']
    """
    return {
        "title": "ASR Demo",
        "inputs": [
            {"name": "speech", "type": "audio", "sources": ["mic", "upload"]},
        ],
        "outputs": [
            {"name": "text", "type": "textbox"},
        ],
    }


def build_inference_default():
    """Return the default inference mapping for the ASR demo.

    Args:
        None

    Returns:
        dict: Default inference settings, including ``output_keys`` and
            ``extra_kwargs``.

    Example:
        >>> build_inference_default()[\"output_keys\"][\"text\"]
        'hyp'
    """
    return {
        "output_keys": {"text": "hyp"},
        "extra_kwargs": {},
    }


def build_ui(demo_cfg):
    """Build ASR UI configuration derived from the demo config.

    Args:
        demo_cfg: Demo configuration object (OmegaConf or attribute-like).

    Returns:
        dict: UI config dictionary with optional overrides applied.

    Example:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.create({\"ui\": {\"title\": \"My ASR\"}})
        >>> build_ui(cfg)[\"title\"]
        'My ASR'
    """
    title = getattr(getattr(demo_cfg, "ui", None), "title", None)
    ui = build_ui_default()
    if title:
        ui["title"] = title
    return ui
