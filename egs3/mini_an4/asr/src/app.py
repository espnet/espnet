"""Recipe-local Gradio launcher for ESPnet3 demos."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gradio as gr

from espnet3.publication.demo.session import load_demo_session
from espnet3.utils.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def build_demo(
    demo_dir: Path,
    demo_config_path: Path | None = None,
):
    """Build the default Gradio Blocks app for one packed demo."""
    assert demo_config_path is not None
    logger.info(
        "Building recipe demo UI | demo_dir=%s demo_config_path=%s",
        demo_dir,
        demo_config_path,
    )
    session = load_demo_session(demo_dir, demo_config_path)
    logger.info(
        "Resolved demo specs | inputs=%s outputs=%s",
        session.input_specs,
        session.output_specs,
    )
    inference_fn = session.create_inference_fn(
        session.input_specs,
        session.output_specs,
    )

    with gr.Blocks(title=session.title) as app:
        if session.title:
            gr.Markdown(f"# {session.title}")

        if session.description:
            gr.Markdown(session.description)

        input_components = []
        with gr.Column():
            # Gradio click handlers bind positional values, not a dict keyed by
            # spec name. Keep this list in the same order as
            # session.input_specs so create_inference_fn(*values) can zip each
            # incoming value back to the matching spec/key.
            for spec in session.input_specs:
                logger.info("Building input component | spec=%s", spec)
                # build_input_component() returns one Gradio input component
                # instance (for example gr.Audio or gr.Textbox). That component
                # object is what Gradio expects in click(..., inputs=[...]).
                input_components.append(session.build_input_component(spec))

        submit_button = gr.Button("Run")

        output_components = []
        with gr.Column():
            # Outputs also stay positional. inference_fn returns one value per
            # spec in this exact order, and Gradio routes each returned value
            # to the component at the same list index.
            for spec in session.output_specs:
                logger.info("Building output component | spec=%s", spec)
                # build_output_component() returns one Gradio output component
                # instance that Gradio can target from click(..., outputs=[...]).
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
    args = parser.parse_args()
    configure_logging(log_dir=args.demo_dir, filename="demo.log")
    logger.info("Starting recipe demo CLI | args=%s", args)
    demo_config_path = args.demo_config or (args.demo_dir / "demo.yaml")
    app = build_demo(
        args.demo_dir,
        demo_config_path=demo_config_path,
    )
    logger.info("Launching Gradio app")
    app.launch()


if __name__ == "__main__":
    main()
