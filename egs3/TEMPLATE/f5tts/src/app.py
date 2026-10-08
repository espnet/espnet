"""Recipe-local Gradio launcher for ESPnet3 F5-TTS demos.

The generic launcher (``egs3/TEMPLATE/esp2_asr/src/app.py``) binds the demo
session's inference function straight to the Gradio components; the session
already turns the model's ``Audio`` into the ``(sample_rate, samples)`` pair
``gr.Audio`` plays. This launcher adds what a text-to-speech UI needs on top:
an untouched text box arrives as ``""``, which is passed on as "not given" so
the contract refuses it by name instead of synthesizing from an empty
transcript, and that refusal is shown in the UI rather than as a bare error.
"""

from __future__ import annotations

# ZeroGPU Spaces kill an app that declares no `@spaces.GPU` function and need
# `spaces` imported before any package that touches CUDA, so this import comes
# first. The package exists only on Hugging Face Spaces.
try:
    import spaces  # isort: skip
except ImportError:  # local runs and plain CPU Spaces
    spaces = None

import argparse  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import gradio as gr  # noqa: E402

from espnet3.publication.demo.session import load_demo_session  # noqa: E402
from espnet3.utils.logging_utils import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)


def build_demo(
    demo_dir: Path,
    demo_config_path: Path | None = None,
):
    """Build the Gradio Blocks app for one packed F5-TTS demo.

    Input and output components are read from the packed ``demo.yaml``. With
    the template's ``conf/demo.yaml`` that is: target text, reference speech
    and reference transcript in, synthesized speech out.

    Args:
        demo_dir: Packed demo directory, containing ``demo.yaml`` and the
            model reference it points at.
        demo_config_path: Optional demo config override. Defaults to
            ``demo_dir / "demo.yaml"``.

    Returns:
        gradio.Blocks: App with one component per input/output spec, bound
        positionally to the session's inference function.

    Example:
        .. code-block:: python

            app = build_demo(Path("exp/training/demo"))
            app.launch()

    Note:
        Reference speech is passed to the model as Gradio delivers it, a
        ``(rate, samples)`` pair; ``Inference`` resamples it to the model's
        rate. A packed config that names the bare ``F5TTSInference`` engine
        instead does no such conversion and is not supported by this app.
    """
    # Resolve first: a relative config path is joined onto the demo directory
    # by `load_demo_session`, which would double a relative `demo_dir`.
    demo_dir = Path(demo_dir).resolve()
    if demo_config_path is None:
        demo_config_path = demo_dir / "demo.yaml"
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

    def synthesize(*values):
        """Run the packed model on the UI values; the session shapes the outputs."""
        # An untouched Gradio text box holds "", which the model would take
        # as an empty transcript; None means "not given", which the contract
        # refuses for a required field with a message naming it.
        values = [None if value == "" else value for value in values]
        try:
            return inference_fn(*values)
        except TypeError as error:
            # A missing or malformed input, e.g. no reference speech or no
            # transcript: show the reason in the UI instead of a bare "Error".
            raise gr.Error(str(error)) from error

    if spaces is not None:
        # On ZeroGPU the model runs on a GPU leased per call; the decorated
        # function is what the Space's scheduler sees.
        logger.info("ZeroGPU detected; registering the handler with spaces.GPU")
        synthesize = spaces.GPU(synthesize)

    with gr.Blocks(title=session.title) as app:
        if session.title:
            gr.Markdown(f"# {session.title}")

        input_components = []
        with gr.Column():
            # Gradio click handlers bind positional values, so this list keeps
            # the order of session.input_specs.
            for spec in session.input_specs:
                logger.info("Building input component | spec=%s", spec)
                input_components.append(session.build_input_component(spec))

        submit_button = gr.Button("Synthesize")

        output_components = []
        with gr.Column():
            # Outputs stay positional too: one value per output spec, in order.
            for spec in session.output_specs:
                logger.info("Building output component | spec=%s", spec)
                output_components.append(session.build_output_component(spec))

        if session.description:
            gr.Markdown(session.description)

        logger.info("Binding Synthesize button click handler")
        submit_button.click(
            fn=synthesize,
            inputs=input_components,
            outputs=output_components,
        )

    logger.info("Recipe demo UI ready")
    return app


def main() -> None:
    """Parse CLI arguments and launch the packed demo.

    Returns:
        None. Blocks while the Gradio server is running.

    Example:
        .. code-block:: bash

            python app.py --demo-dir exp/training/demo
    """
    parser = argparse.ArgumentParser(description="Launch an ESPnet3 F5-TTS demo.")
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
