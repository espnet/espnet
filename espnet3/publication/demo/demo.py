"""Default Gradio launcher for packed ESPnet3 demos."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gradio as gr

from espnet3.publication.demo.assets import (
    build_runtime_overrides,
    load_demo_session,
)

logger = logging.getLogger(__name__)


def _emit_demo_message(message: str, *args) -> None:
    text = message % args if args else message
    print(text, flush=True)
    logger.info(text)


def build_demo(
    demo_dir: Path,
    demo_config_path: Path | None = None,
    device: str | None = None,
    config_overrides: dict[str, object] | None = None,
):
    """Build the default Gradio Blocks app for one packed demo."""
    _emit_demo_message(
        "Building demo UI | demo_dir=%s demo_config_path=%s device=%s overrides=%s",
        demo_dir,
        demo_config_path,
        device,
        config_overrides,
    )
    model_overrides = build_runtime_overrides(
        device=device,
        base_overrides=config_overrides,
    )
    session = load_demo_session(
        demo_dir,
        demo_config_path,
        model_overrides=model_overrides,
    )
    input_specs = session.resolve_input_specs()
    output_specs = session.resolve_output_specs()
    _emit_demo_message(
        "Resolved demo specs | inputs=%s outputs=%s", input_specs, output_specs
    )
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
                print(f"Building input component | spec={spec}", flush=True)
                input_components.append(session.build_input_component(spec))

        submit_button = gr.Button("Run")

        output_components = []
        with gr.Column():
            for spec in output_specs:
                logger.info("Building output component | spec=%s", spec)
                print(f"Building output component | spec={spec}", flush=True)
                output_components.append(session.build_output_component(spec))

        _emit_demo_message("Binding Run button click handler")
        submit_button.click(
            fn=inference_fn,
            inputs=input_components,
            outputs=output_components,
        )

    _emit_demo_message("Demo UI ready")
    return app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Launch an ESPnet3 demo.")
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to the packed demo directory.",
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
    _emit_demo_message("Starting demo CLI | args=%s", args)
    config_overrides = build_runtime_overrides(
        override_args=args.config_override,
    )
    app = build_demo(
        args.demo_dir,
        demo_config_path=args.demo_config,
        device=args.device,
        config_overrides=config_overrides,
    )
    _emit_demo_message("Launching Gradio app")
    app.launch()


if __name__ == "__main__":
    main()
