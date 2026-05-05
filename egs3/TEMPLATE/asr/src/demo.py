"""Recipe-local Gradio launcher for ESPnet3 demos."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gradio as gr

from espnet3.publication.demo.session import (
    build_runtime_overrides,
    load_demo_session,
)
from espnet3.utils.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def build_demo(
    demo_dir: Path,
    demo_config_path: Path | None = None,
    device: str | None = None,
    config_overrides: dict[str, object] | None = None,
):
    """Build the default Gradio Blocks app for one packed demo."""
    if demo_config_path is None:
        demo_config_path = demo_dir / "demo.yaml"
    logger.info(
        "Building recipe demo UI | demo_dir=%s demo_config_path=%s device=%s overrides=%s",
        demo_dir,
        demo_config_path,
        device,
        config_overrides,
    )
    base_overrides = dict(config_overrides or {})
    if device:
        base_overrides["device"] = device
    model_overrides = build_runtime_overrides(
        base_overrides=base_overrides or None,
    )
    session = load_demo_session(
        demo_dir,
        demo_config_path,
        model_overrides=model_overrides,
    )
    input_specs = session.input_specs
    output_specs = session.output_specs
    logger.info("Resolved demo specs | inputs=%s outputs=%s", input_specs, output_specs)
    inference_fn = session.create_inference_fn(input_specs, output_specs)

    with gr.Blocks(title=session.title) as app:
        if session.title:
            gr.Markdown(f"# {session.title}")
        if session.description:
            gr.Markdown(session.description)

        input_components = []
        with gr.Column():
            for spec in input_specs:
                logger.info("Building input component | spec=%s", spec)
                input_components.append(session.build_input_component(spec))

        submit_button = gr.Button("Run")

        output_components = []
        with gr.Column():
            for spec in output_specs:
                logger.info("Building output component | spec=%s", spec)
                output_components.append(session.build_output_component(spec))

        logger.info("Binding Run button click handler")
        submit_button.click(
            fn=inference_fn,
            inputs=input_components,
            outputs=output_components,
        )

    logger.info("Recipe demo UI ready")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch an ESPnet3 demo.")
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to the demo directory. Defaults to this script's directory.",
    )
    parser.add_argument(
        "--demo-config",
        type=Path,
        default=None,
        help="Optional packed demo config path. Relative paths use --demo-dir.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional inference device override, such as cpu or cuda:0.",
    )
    parser.add_argument(
        "--config-override",
        action="append",
        default=None,
        help="Optional inference config override, such as model.beam_size=1.",
    )
    args = parser.parse_args()
    configure_logging(log_dir=args.demo_dir, filename="demo.log")
    logger.info("Starting recipe demo CLI | args=%s", args)
    config_overrides = build_runtime_overrides(
        override_args=args.config_override,
    )
    demo_config_path = args.demo_config or (args.demo_dir / "demo.yaml")
    app = build_demo(
        args.demo_dir,
        demo_config_path=demo_config_path,
        device=args.device,
        config_overrides=config_overrides,
    )
    logger.info("Launching Gradio app")
    app.launch()


if __name__ == "__main__":
    main()
