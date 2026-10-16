"""Recipe-local Gradio launcher for ESPnet3 demos."""

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


def convert_gradio_audio_to_waveform(value: Any, sample_rate: int) -> np.ndarray | None:
    """Convert a ``gr.Audio`` value into the waveform the codec expects.

    ``gr.Audio`` (``type="numpy"``) delivers ``(rate, samples)`` with integer
    samples at the rate of the recording or file, and stereo material as a
    ``(frames, channels)`` array. ``AudioCoding`` wants a float32 mono array at
    the model's sampling rate, so scale integers to ``[-1, 1]``, average the
    channels, and resample when the rates differ.

    Args:
        value: ``(rate, samples)`` tuple from Gradio, or ``None`` when the
            component is empty.
        sample_rate: Sampling rate the model was trained at.

    Returns:
        float32 mono waveform at ``sample_rate``, or ``None`` for ``None``.

    Raises:
        TypeError: If ``value`` is not a ``(rate, samples)`` tuple.

    Examples:
        ```python
        waveform = convert_gradio_audio_to_waveform(
            (48000, np.zeros((48000, 2), dtype=np.int16)), 24000
        )
        waveform.shape  # -> (24000,)
        ```
    """
    if value is None:
        return None
    if not (isinstance(value, (tuple, list)) and len(value) == 2):
        raise TypeError(
            "Expected a (sample_rate, samples) tuple from gr.Audio, "
            f"got {type(value).__name__}"
        )
    source_sample_rate, samples = value
    waveform = np.asarray(samples)
    if np.issubdtype(waveform.dtype, np.integer):
        waveform = waveform.astype(np.float32) / np.iinfo(waveform.dtype).max
    else:
        waveform = waveform.astype(np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    waveform = waveform.reshape(-1)
    if int(source_sample_rate) != int(sample_rate):
        import librosa

        waveform = librosa.resample(
            waveform,
            orig_sr=int(source_sample_rate),
            target_sr=int(sample_rate),
        )
    return np.ascontiguousarray(waveform, dtype=np.float32)


def convert_waveform_to_gradio_audio(
    value: Any, sample_rate: int
) -> tuple[int, np.ndarray]:
    """Wrap a model waveform as the ``(rate, samples)`` pair ``gr.Audio`` plays.

    Samples are returned as int16. Gradio peak-normalizes float arrays when it
    converts them to 16-bit audio, which would change the level of a codec's
    output; int16 samples are passed through unchanged.

    Args:
        value: Waveform in ``[-1, 1]`` as an array-like (torch tensors and
            ``(1, T)`` shapes included).
        sample_rate: Rate the waveform was generated at.

    Returns:
        ``(sample_rate, samples)`` with a flat int16 array.

    Examples:
        ```python
        rate, samples = convert_waveform_to_gradio_audio(
            np.zeros((1, 240)), 24000
        )
        samples.shape  # -> (240,)
        ```
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    waveform = np.asarray(value, dtype=np.float32).reshape(-1)
    waveform = np.clip(waveform, -1.0, 1.0) * np.iinfo(np.int16).max
    return int(sample_rate), waveform.astype(np.int16)


def wrap_inference_fn(
    inference_fn: Callable[..., Any],
    input_specs: Sequence[dict[str, Any]],
    output_specs: Sequence[dict[str, Any]],
    sample_rate: int | None,
) -> Callable[..., Any]:
    """Adapt a session inference function to Gradio's audio value format.

    The session function created by ``DemoSession.create_inference_fn`` maps
    UI values positionally onto model input keys and returns one value per
    output spec (a bare value when there is exactly one). This wrapper
    converts every ``type: audio`` input with
    :func:`convert_gradio_audio_to_waveform` and every ``type: audio`` output
    with :func:`convert_waveform_to_gradio_audio`; other types pass through
    untouched.

    Args:
        inference_fn: Positional inference callable from the demo session.
        input_specs: Resolved ``ui.inputs`` specs, in UI order.
        output_specs: Resolved ``ui.outputs`` specs, in UI order.
        sample_rate: Model sampling rate (``ui.sample_rate``); required as soon
            as any spec is audio.

    Returns:
        Callable suitable for ``gr.Button.click``.

    Raises:
        ValueError: If an audio spec is present but ``sample_rate`` is unset.

    Examples:
        ```python
        run_inference = wrap_inference_fn(
            lambda audio: audio,
            [{"key": "audio", "type": "audio", "label": "In"}],
            [{"key": "wav", "type": "audio", "label": "Out"}],
            sample_rate=24000,
        )
        rate, samples = run_inference((24000, np.zeros(240, dtype=np.int16)))
        ```
    """
    input_is_audio = [spec.get("type") == "audio" for spec in input_specs]
    output_is_audio = [spec.get("type") == "audio" for spec in output_specs]
    if (any(input_is_audio) or any(output_is_audio)) and sample_rate is None:
        raise ValueError(
            "ui.sample_rate is required in demo.yaml when an input or output "
            "spec has type: audio; it is the model's sampling rate."
        )

    def run_inference(*values: Any) -> Any:
        converted = [
            convert_gradio_audio_to_waveform(value, sample_rate) if is_audio else value
            for value, is_audio in zip(values, input_is_audio)
        ]
        result = inference_fn(*converted)
        outputs = list(result) if len(output_specs) > 1 else [result]
        outputs = [
            convert_waveform_to_gradio_audio(value, sample_rate) if is_audio else value
            for value, is_audio in zip(outputs, output_is_audio)
        ]
        return outputs[0] if len(outputs) == 1 else outputs

    return run_inference


def build_demo(
    demo_dir: Path,
    demo_config_path: Path | None = None,
):
    """Build the default Gradio Blocks app for one packed demo.

    Input and output components are read from the packed ``demo.yaml``, so
    this launcher is task-agnostic: a codec demo wires audio in to audio out
    purely through that config. Audio values are converted between Gradio's
    ``(rate, samples)`` format and the model's float32 waveform at
    ``ui.sample_rate``, and on ZeroGPU Spaces the inference function is
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
    ui_cfg = getattr(session.demo_cfg, "ui", None)
    sample_rate = ui_cfg.get("sample_rate") if ui_cfg is not None else None
    inference_fn = wrap_inference_fn(
        session.create_inference_fn(
            session.input_specs,
            session.output_specs,
        ),
        session.input_specs,
        session.output_specs,
        sample_rate,
    )
    if spaces is not None:
        # Every Run then draws on the ZeroGPU quota even for a CPU-only codec;
        # it is the price of hosting a Gradio Space on a free account.
        logger.info("ZeroGPU detected; registering the inference fn with spaces.GPU")
        inference_fn = spaces.GPU(inference_fn)

    with gr.Blocks(title=session.title) as app:
        if session.title:
            gr.Markdown(f"# {session.title}")

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

        if session.description:
            gr.Markdown(session.description)

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
