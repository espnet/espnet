"""Recipe-local Gradio launcher for ESPnet3 codec demos.

The generic launcher (``egs3/TEMPLATE/esp2_asr/src/app.py``) binds the demo
session's inference function straight to the Gradio components; the session
builds the components from the packed inference contract, which also turns
whatever ``gr.Audio`` delivers into the waveform the codec takes, and hands
the resynthesized ``Audio`` back as the ``(sample_rate, samples)`` pair
``gr.Audio`` plays. This launcher adds two things a codec demo needs: the
output samples are passed to Gradio as int16, because Gradio peak-normalizes
float arrays and would change the level the codec reproduced, and on ZeroGPU
Spaces the handler is registered with ``spaces.GPU``.
"""

from __future__ import annotations

# ZeroGPU Spaces kill any app that never declares a `@spaces.GPU` function and
# require `spaces` to be imported before any package that touches CUDA, so this
# import has to come first. The package only exists on Hugging Face Spaces.
try:
    import spaces  # isort: skip
except ImportError:  # local runs and plain CPU Spaces
    spaces = None

import argparse  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Sequence  # noqa: E402

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

from espnet3.publication.demo.session import load_demo_session  # noqa: E402
from espnet3.utils.logging_utils import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)


def convert_demo_audio_to_int16(value: Any) -> tuple[int, np.ndarray]:
    """Return a ``(sample_rate, samples)`` pair with int16 samples.

    The demo session returns a contract ``audio`` output as the
    ``(sample_rate, samples)`` pair ``gr.Audio`` plays, with float samples in
    ``[-1, 1]``. Gradio peak-normalizes float arrays when it converts them to
    16-bit audio, which would change the level of a codec's output; int16
    samples are passed through unchanged.

    Args:
        value: ``(sample_rate, samples)`` with float samples, samples first
            for multichannel audio as the session lays them out.

    Returns:
        ``(sample_rate, samples)`` with the same shape and int16 samples.

    Raises:
        TypeError: If ``value`` is not a ``(sample_rate, samples)`` pair.

    Examples:
        ```python
        rate, samples = convert_demo_audio_to_int16((24000, np.full(240, 0.5)))
        samples.dtype  # -> dtype('int16')
        ```
    """
    if not (isinstance(value, (tuple, list)) and len(value) == 2):
        raise TypeError(
            "Expected a (sample_rate, samples) pair from the demo session, "
            f"got {type(value).__name__}"
        )
    sample_rate, samples = value
    waveform = np.asarray(samples, dtype=np.float32)
    waveform = np.clip(waveform, -1.0, 1.0) * np.iinfo(np.int16).max
    return int(sample_rate), waveform.astype(np.int16)


def wrap_inference_fn(
    inference_fn: Callable[..., Any],
    output_specs: Sequence[dict[str, Any]],
) -> Callable[..., Any]:
    """Pass every ``type: audio`` output to Gradio as int16 samples.

    The session function created by ``DemoSession.create_inference_fn``
    returns one value per output spec (a bare value when there is exactly
    one). Inputs are left to the packed contract; this wrapper only converts
    the audio outputs with :func:`convert_demo_audio_to_int16`, other types
    pass through untouched.

    Args:
        inference_fn: Positional inference callable from the demo session.
        output_specs: Resolved output specs, in UI order.

    Returns:
        Callable suitable for ``gr.Button.click``.

    Examples:
        ```python
        run_inference = wrap_inference_fn(
            lambda audio: (24000, np.full(240, 0.5)),
            [{"key": "wav", "type": "audio", "label": "Out"}],
        )
        rate, samples = run_inference((24000, np.zeros(240, dtype=np.int16)))
        ```
    """
    output_is_audio = [spec.get("type") == "audio" for spec in output_specs]

    def run_inference(*values: Any) -> Any:
        result = inference_fn(*values)
        outputs = list(result) if len(output_specs) > 1 else [result]
        outputs = [
            convert_demo_audio_to_int16(value) if is_audio else value
            for value, is_audio in zip(outputs, output_is_audio)
        ]
        return outputs[0] if len(outputs) == 1 else outputs

    return run_inference


def build_demo(
    demo_dir: Path,
    demo_config_path: Path | None = None,
):
    """Build the default Gradio Blocks app for one packed demo.

    Input and output components are built from the packed inference config's
    contract (or from ``ui.inputs`` / ``ui.outputs`` when the demo config
    sets them), so this launcher is task-agnostic: a codec demo wires audio
    in to audio out purely through that config. Audio outputs reach Gradio
    as int16 samples, and on ZeroGPU Spaces the inference function is
    registered with ``spaces.GPU``.

    Args:
        demo_dir: Packed demo directory, containing ``demo.yaml`` and the
            model reference it points at.
        demo_config_path: Optional demo config override. Defaults to
            ``demo_dir / "demo.yaml"``.

    Returns:
        gradio.Blocks: App with one component per input/output spec, bound
        positionally to the session's inference function.

    Examples:
        ```python
        app = build_demo(Path("exp/train_encodec_libritts/demo"))
        app.launch()
        ```
    """
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
    inference_fn = wrap_inference_fn(
        session.create_inference_fn(
            session.input_specs,
            session.output_specs,
        ),
        session.output_specs,
    )
    if spaces is not None:
        # Every Run then draws on the ZeroGPU quota even for a CPU-only codec;
        # it is the price of hosting a Gradio Space on a free account.
        logger.info("ZeroGPU detected; registering the inference fn with spaces.GPU")
        inference_fn = spaces.GPU(inference_fn)

    with gr.Blocks(title=session.title) as app:
        if session.title:
            gr.Markdown(f"# {session.title}")
        # What the inputs are and what to expect, before the inputs themselves.
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
    """Parse CLI arguments and launch the packed demo.

    Returns:
        None. Blocks while the Gradio server is running.

    Examples:
        ```shell
        python app.py --demo-dir exp/train_encodec_libritts/demo
        ```
    """
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
