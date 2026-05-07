"""Custom Gradio launcher used by publication integration tests."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gradio as gr
import numpy as np

from espnet3.publication.demo.session import load_demo_session
from espnet3.utils.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def build_demo(
    demo_dir: Path,
    demo_config_path: Path | None = None,
):
    """Build a demo app with an extra image block."""
    assert demo_config_path is not None
    session = load_demo_session(demo_dir, demo_config_path)
    inference_fn = session.create_inference_fn(
        input_keys=["speech"],
        output_keys=["hyp"],
    )

    with gr.Blocks(title=session.title) as app:
        if session.title:
            gr.Markdown(f"# {session.title}")

        with gr.Row():
            with gr.Column():
                input_components = [
                    session.build_input_component(spec) for spec in session.input_specs
                ]
                submit_button = gr.Button("Run")
            with gr.Column():
                gr.Image(
                    value=np.zeros((16, 16, 3), dtype=np.uint8),
                    label="Reference Image",
                    interactive=False,
                    show_download_button=False,
                    show_share_button=False,
                )
                output_components = [
                    session.build_output_component(spec)
                    for spec in session.output_specs
                ]

        if session.description:
            gr.Markdown(session.description)

        submit_button.click(
            fn=inference_fn,
            inputs=input_components,
            outputs=output_components,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a custom ESPnet3 demo.")
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--demo-config",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    configure_logging(log_dir=args.demo_dir, filename="demo.log")
    demo_config_path = args.demo_config or (args.demo_dir / "demo.yaml")
    app = build_demo(args.demo_dir, demo_config_path=demo_config_path)
    logger.info("Launching custom Gradio app")
    app.launch()


if __name__ == "__main__":
    main()
