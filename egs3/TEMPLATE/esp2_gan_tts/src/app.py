"""Gradio launcher for ESPnet3 espnet2 GAN-TTS demos.

The generic launcher (``egs3/TEMPLATE/esp2_asr/src/app.py``) binds the demo
session's inference function straight to the Gradio components built from the
packed inference contract. That is not enough here: a multi-speaker VITS wants
a speaker embedding (``spembs`` in the contract), while the only thing a
visitor can supply through a browser is a reference recording. This launcher
closes that gap: the page shows a text box and an audio widget (``ui.inputs``
in ``demo.yaml``), the reference recording is turned into an embedding by the
same extractor the ``compute_xvectors`` stage uses (``xvector`` in
``demo.yaml``), and the model is called through the session with the
contract's own inputs, so the checked boundary is the one ``infer`` and
``InferenceModel`` use. The synthesized ``Audio`` is handed to Gradio as the
``(sample_rate, samples)`` pair ``gr.Audio`` plays.
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
from typing import Any, Callable  # noqa: E402

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

from espnet3.contract.kinds import KINDS  # noqa: E402
from espnet3.contract.kinds.audio import Audio  # noqa: E402
from espnet3.publication.demo.session import load_demo_session  # noqa: E402
from espnet3.utils.logging_utils import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

#: The contract inputs of an esp2_gan_tts bundle the launcher fills in.
TEXT_INPUT = "text"
SPEAKER_EMBEDDING_INPUT = "spembs"


def build_xvector_extractor(demo_cfg) -> Callable[[Any], np.ndarray]:
    """Build the callable that turns a reference recording into a speaker embedding.

    Args:
        demo_cfg: The packed demo config. Its ``xvector`` block names the
            ``toolkit``, ``pretrained_model`` and ``device`` of the extractor,
            which must match the ``xvector`` block the model was trained
            with; the defaults are the training template's.

    Returns:
        A callable taking the reference in any form
        :meth:`~espnet3.contract.kinds.audio.Audio.coerce` accepts (the
        ``(rate, samples)`` pair ``gr.Audio`` returns, a path, an ``Audio``)
        and returning a 1-D float32 embedding.

    Raises:
        ValueError: If ``xvector.toolkit`` is not one of ``espnet``,
            ``speechbrain`` or ``rawnet``.

    Note:
        The provider and runner are imported lazily so that importing this
        module (as ``pack_demo`` does when copying it) does not pull in the
        speaker-embedding toolchain.

    Example:
        .. code-block:: python

            extract = build_xvector_extractor(session.demo_cfg)
            spembs = extract((22050, np.zeros(22050, dtype=np.float32)))
            spembs.shape  # -> (192,) for ECAPA-TDNN, (256,) for RawNet3
    """
    from espnet3.systems.esp2_gan_tts.src.xvector_provider import XVectorProvider
    from espnet3.systems.esp2_gan_tts.src.xvector_runner import XVectorRunner

    xvector_cfg = getattr(demo_cfg, "xvector", None)
    toolkit = "espnet"
    pretrained_model = "espnet/voxcelebs12_rawnet3"
    device = "cpu"
    if xvector_cfg is not None:
        toolkit = xvector_cfg.get("toolkit", toolkit)
        pretrained_model = xvector_cfg.get("pretrained_model", pretrained_model)
        device = xvector_cfg.get("device", device)

    logger.info(
        "Loading speaker embedding model | toolkit=%s model=%s device=%s",
        toolkit,
        pretrained_model,
        device,
    )
    model = XVectorProvider._build_model(toolkit, pretrained_model, device)

    def extract(reference: Any) -> np.ndarray:
        """Return the embedding of ``reference``, mixed down to mono first."""
        audio = Audio.coerce(reference, None)
        waveform = audio.array.mean(axis=0) if audio.array.ndim == 2 else audio.array
        embedding = XVectorRunner._extract_embedding(
            np.ascontiguousarray(waveform, dtype=np.float32),
            audio.rate,
            model,
            toolkit,
            device,
        )
        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu().numpy()
        return np.asarray(embedding, dtype=np.float32).reshape(-1)

    return extract


def _missing_inputs(specs: list[dict], values: tuple) -> list[str]:
    """Return the labels of the inputs the visitor left empty.

    An untouched text box arrives as ``""`` and an empty audio widget as
    ``None``; both inputs are required, so either is reported by the label
    the page shows rather than handed to the model.

    Args:
        specs: The session's input specs, in widget order.
        values: The widget values, in the same order.

    Returns:
        The labels of the empty inputs, in widget order.

    Example:
        .. code-block:: python

            _missing_inputs(specs, ("hello", None))
            # -> ["Reference recording (voice to clone)"]
    """
    return [
        spec["label"]
        for spec, value in zip(specs, values)
        if value is None or (isinstance(value, str) and not value.strip())
    ]


def _widget_keys(input_specs: list[dict]) -> tuple[str, str]:
    """Return the keys of the text widget and of the reference audio widget.

    Args:
        input_specs: The session's input specs (``ui.inputs`` of the demo
            config).

    Returns:
        ``(text_key, reference_key)``.

    Raises:
        ValueError: Unless the specs hold exactly one ``text`` widget and one
            ``audio`` widget.

    Example:
        .. code-block:: python

            _widget_keys(
                [{"key": "text", "type": "text", "label": "Text"},
                 {"key": "reference_speech", "type": "audio", "label": "Voice"}]
            )
            # -> ("text", "reference_speech")
    """
    by_type: dict[str, list[str]] = {}
    for spec in input_specs:
        by_type.setdefault(spec["type"], []).append(spec["key"])
    if sorted(by_type) != ["audio", "text"] or any(
        len(keys) != 1 for keys in by_type.values()
    ):
        raise ValueError(
            "demo.yaml ui.inputs must hold exactly one text widget and one "
            f"audio widget, got {[(s['key'], s['type']) for s in input_specs]}"
        )
    return by_type["text"][0], by_type["audio"][0]


def build_demo(
    demo_dir: Path,
    demo_config_path: Path | None = None,
):
    """Build the Gradio Blocks app for one packed GAN-TTS demo.

    Args:
        demo_dir: Packed demo directory, containing ``demo.yaml`` and the
            model reference it points at.
        demo_config_path: Optional demo config override. Defaults to
            ``demo_dir / "demo.yaml"``.

    Returns:
        gradio.Blocks: App with a text box and a reference-audio input wired
        to one component per output of the packed inference contract.

    Raises:
        ValueError: If ``ui.inputs`` does not hold one text and one audio
            widget.

    Example:
        .. code-block:: python

            app = build_demo(Path("exp/train_vits_libritts/demo"))
            app.launch()

    Note:
        The reference recording is passed to the extractor as Gradio delivers
        it, a ``(rate, samples)`` pair; the extractor resamples it to the
        embedding model's rate. The model is called through the session with
        ``{"text": ..., "spembs": ...}``, so the packed contract checks both
        and returns the synthesized ``Audio``, shown as the ``(rate,
        samples)`` pair ``gr.Audio`` plays.
    """
    demo_dir = Path(demo_dir).resolve()
    if demo_config_path is None:
        demo_config_path = demo_dir / "demo.yaml"
    logger.info(
        "Building GAN-TTS demo UI | demo_dir=%s demo_config_path=%s",
        demo_dir,
        demo_config_path,
    )
    session = load_demo_session(demo_dir, demo_config_path)
    input_specs = session.input_specs
    output_specs = session.output_specs
    text_key, reference_key = _widget_keys(input_specs)
    logger.info("Resolved demo specs | inputs=%s outputs=%s", input_specs, output_specs)
    extract_xvector = build_xvector_extractor(session.demo_cfg)

    def synthesize(*values):
        """Embed the reference, run the packed model, shape the outputs for the UI."""
        missing = _missing_inputs(input_specs, values)
        if missing:
            raise gr.Error("Required: " + ", ".join(missing))
        by_key = {spec["key"]: value for spec, value in zip(input_specs, values)}
        try:
            spembs = extract_xvector(by_key[reference_key])
            result = session.model(
                {TEXT_INPUT: str(by_key[text_key]), SPEAKER_EMBEDDING_INPUT: spembs}
            )
        except (TypeError, ValueError) as error:
            # A malformed input the contract or the model refused, e.g. a
            # file that is not audio: show the reason in the UI instead of a
            # bare "Error".
            raise gr.Error(str(error)) from error
        outputs = [
            KINDS[spec["type"]].to_demo_output(result[spec["key"]])
            for spec in output_specs
        ]
        return outputs[0] if len(outputs) == 1 else outputs

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
            # the order of input_specs.
            for spec in input_specs:
                logger.info("Building input component | spec=%s", spec)
                input_components.append(session.build_input_component(spec))

        submit_button = gr.Button("Synthesize")

        output_components = []
        with gr.Column():
            # Outputs stay positional too: one value per output spec, in order.
            for spec in output_specs:
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

    logger.info("GAN-TTS demo UI ready")
    return app


def main() -> None:
    """Parse CLI arguments and launch the packed demo.

    Returns:
        None. Blocks while the Gradio server is running.

    Example:
        .. code-block:: bash

            python app.py --demo-dir exp/train_vits_libritts/demo
    """
    parser = argparse.ArgumentParser(description="Launch an ESPnet3 GAN-TTS demo.")
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
    logger.info("Starting GAN-TTS demo CLI | args=%s", args)
    demo_config_path = args.demo_config or (args.demo_dir / "demo.yaml")
    app = build_demo(args.demo_dir, demo_config_path=demo_config_path)
    logger.info("Launching Gradio app")
    app.launch()


if __name__ == "__main__":
    main()
