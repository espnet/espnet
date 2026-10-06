"""Recipe-local Gradio launcher for ESPnet3 F5-TTS demos.

The generic launcher (``egs3/TEMPLATE/esp2_asr/src/app.py``) hands each model
output straight to its Gradio component. That is not enough for speech
synthesis: ``gr.Audio`` plays a ``(sample_rate, samples)`` pair, while the
packed model returns the samples alone. This launcher adds that pairing, and
passes an empty text box as "not given" so the optional reference transcript
can be left blank.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gradio as gr
import numpy as np

from espnet3.publication.demo.session import load_demo_session
from espnet3.utils.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def resolve_sample_rate(model) -> int:
    """Return the rate of the waveforms a packed F5-TTS model synthesizes.

    Args:
        model: The backend built from the packed ``conf/inference.yaml``:
            ``espnet3.systems.f5tts.inference.Inference`` (``sample_rate``)
            or the bare ``F5TTSInference`` engine (``target_sample_rate``).

    Returns:
        int: Samples per second.

    Raises:
        TypeError: If the model exposes neither attribute, so the rate of its
            output cannot be told.

    Example:
        .. code-block:: python

            session = load_demo_session(demo_dir, demo_dir / "demo.yaml")
            resolve_sample_rate(session.model.model)  # -> 24000
    """
    for name in ("sample_rate", "target_sample_rate"):
        sample_rate = getattr(model, name, None)
        if sample_rate:
            return int(sample_rate)
    raise TypeError(
        f"Cannot tell the output sample rate of {type(model).__name__}: it has "
        "neither `sample_rate` nor `target_sample_rate`."
    )


def build_gradio_audio(wav, sample_rate: int):
    """Pair a synthesized waveform with its rate, as ``gr.Audio`` expects.

    Args:
        wav: The model's ``wav`` output: a float array, or an
            :class:`espnet3.api.inference.Audio`, whose own rate is then used.
        sample_rate: Rate of a bare array.

    Returns:
        tuple[int, numpy.ndarray]: ``(sample_rate, float32 samples)``.

    Example:
        .. code-block:: python

            build_gradio_audio(np.zeros(24000, dtype=np.float32), 24000)
            # -> (24000, array([0., 0., ...], dtype=float32))
    """
    samples = getattr(wav, "array", wav)
    rate = int(getattr(wav, "rate", sample_rate))
    return rate, np.asarray(samples, dtype=np.float32)


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
    sample_rate = resolve_sample_rate(session.model.model)
    is_audio_output = [spec["type"] == "audio" for spec in session.output_specs]

    def synthesize(*values):
        """Run the packed model on the UI values and pair audio with its rate."""
        # An untouched Gradio text box holds "", which the model would take
        # as an empty transcript; None means "not given" instead.
        values = [None if value == "" else value for value in values]
        try:
            outputs = inference_fn(*values)
        except TypeError as error:
            # A missing or malformed input, e.g. no reference speech: show the
            # reason in the UI instead of a bare "Error".
            raise gr.Error(str(error)) from error
        # The session returns a bare value for a single output spec.
        if len(is_audio_output) == 1:
            outputs = [outputs]
        outputs = [
            build_gradio_audio(output, sample_rate) if is_audio else output
            for output, is_audio in zip(outputs, is_audio_output)
        ]
        return outputs[0] if len(outputs) == 1 else outputs

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
