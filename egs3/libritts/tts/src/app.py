"""Gradio launcher for the multi-speaker LibriTTS VITS demo.

The generic template launcher (``egs3/TEMPLATE/tts/src/app.py``) maps each UI
value straight onto a model input key. That is not enough here: this recipe's
VITS is multi-speaker, so the model wants a speaker embedding (``spembs``),
while the only thing a person can supply through a browser is a reference
waveform. This launcher closes that gap by running the same SpeechBrain ECAPA
extractor the ``compute_xvectors`` stage uses before calling the model.
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


def build_xvector_extractor(demo_cfg):
    """Build the callable that turns reference audio into a speaker embedding.

    Args:
        demo_cfg: Loaded demo config. The defaults below match the
            ``compute_xvectors`` stage, so a recipe that trains against the
            standard ECAPA embeddings needs no config at all. A recipe using
            a different embedding model overrides them under
            ``demo_cfg.xvector`` (``toolkit``, ``pretrained_model``,
            ``device``).

    Returns:
        Callable taking ``(wav, sample_rate)`` and returning a 1-D float32
        ``numpy.ndarray`` speaker embedding.

    Raises:
        ValueError: If ``xvector.toolkit`` is not one of ``espnet``,
            ``speechbrain`` or ``rawnet``.

    Notes:
        Imports the provider/runner lazily so that merely importing this
        module (as ``pack_demo`` does when copying it) does not pull in the
        speaker-embedding toolchain.

    Examples:
        ```python
        extract = build_xvector_extractor(session.demo_cfg)
        spembs = extract(np.zeros(22050, dtype=np.float32), 22050)
        spembs.shape  # -> (192,) for ECAPA-TDNN
        ```
    """
    from espnet3.systems.tts.xvector_provider import XVectorProvider
    from espnet3.systems.tts.xvector_runner import XVectorRunner

    xvec_cfg = getattr(demo_cfg, "xvector", None)
    toolkit = "speechbrain"
    pretrained_model = "speechbrain/spkrec-ecapa-voxceleb"
    device = "cpu"
    if xvec_cfg is not None:
        toolkit = xvec_cfg.get("toolkit", toolkit)
        pretrained_model = xvec_cfg.get("pretrained_model", pretrained_model)
        device = xvec_cfg.get("device", device)

    logger.info(
        "Loading speaker embedding model | toolkit=%s model=%s device=%s",
        toolkit,
        pretrained_model,
        device,
    )
    model = XVectorProvider._build_model(toolkit, pretrained_model, device)

    def extract(wav: np.ndarray, sample_rate: int) -> np.ndarray:
        embedding = XVectorRunner._extract_embedding(
            np.asarray(wav, dtype=np.float32), sample_rate, model, toolkit, device
        )
        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu().numpy()
        return np.asarray(embedding, dtype=np.float32).reshape(-1)

    return extract


def build_demo(
    demo_dir: Path,
    demo_config_path: Path | None = None,
):
    """Build the multi-speaker TTS Gradio app for one packed demo.

    Args:
        demo_dir: Packed demo directory, containing ``demo.yaml`` and the
            model reference it points at.
        demo_config_path: Optional demo config override. Defaults to
            ``demo_dir / "demo.yaml"``.

    Returns:
        gradio.Blocks: App with a text box and a reference-audio input wired
        to a synthesized-audio output.

    Notes:
        Labels come from ``ui.inputs`` / ``ui.outputs`` in the packed demo
        config, but the wiring is fixed here rather than positional, because
        the reference audio is converted before it reaches the model.

    Examples:
        ```python
        app = build_demo(Path("exp/train_vits_libritts/demo"))
        app.launch()
        ```
    """
    if demo_config_path is None:
        demo_config_path = demo_dir / "demo.yaml"
    logger.info(
        "Building LibriTTS VITS demo UI | demo_dir=%s demo_config_path=%s",
        demo_dir,
        demo_config_path,
    )
    session = load_demo_session(demo_dir, demo_config_path)
    extract_xvector = build_xvector_extractor(session.demo_cfg)
    # Matches output_artifacts.wav.sample_rate in conf/inference.yaml; a recipe
    # synthesizing at another rate sets `output_sample_rate` in its demo config.
    output_sample_rate = int(session.demo_cfg.get("output_sample_rate", 22050))

    labels = {spec["key"]: spec["label"] for spec in session.input_specs}
    output_labels = {spec["key"]: spec["label"] for spec in session.output_specs}

    def synthesize(text, reference_audio):
        if not text or not str(text).strip():
            raise gr.Error("Please enter some text to synthesize.")
        if reference_audio is None:
            raise gr.Error("Please provide a reference audio clip.")

        # gr.Audio(type="numpy") hands back (sample_rate, waveform).
        sample_rate, wav = reference_audio
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:  # mixdown to mono
            wav = wav.mean(axis=1)
        peak = np.abs(wav).max()
        if peak > 1.0:  # gradio returns int16-scaled data for uploaded files
            wav = wav / peak

        logger.info(
            "Synthesizing | chars=%d reference_sr=%d reference_len=%d",
            len(text),
            sample_rate,
            wav.shape[0],
        )
        spembs = extract_xvector(wav, sample_rate)
        result = session.model({"text": str(text), "spembs": spembs})
        synthesized = result["wav"] if isinstance(result, dict) else result
        if hasattr(synthesized, "detach"):
            synthesized = synthesized.detach().cpu().numpy()
        synthesized = np.asarray(synthesized, dtype=np.float32).reshape(-1)
        logger.info("Synthesized %d samples", synthesized.shape[0])
        return output_sample_rate, synthesized

    with gr.Blocks(title=session.title) as app:
        if session.title:
            gr.Markdown(f"# {session.title}")

        with gr.Column():
            text_input = gr.Textbox(label=labels.get("text", "Text to synthesize"))
            reference_input = gr.Audio(
                label=labels.get("spembs", "Reference audio (voice to clone)")
            )

        submit_button = gr.Button("Synthesize")

        with gr.Column():
            audio_output = gr.Audio(label=output_labels.get("wav", "Synthesized Audio"))

        if session.description:
            gr.Markdown(session.description)

        submit_button.click(
            fn=synthesize,
            inputs=[text_input, reference_input],
            outputs=[audio_output],
        )

    logger.info("LibriTTS VITS demo UI ready")
    return app


def main() -> None:
    """Parse CLI arguments and launch the packed demo.

    Returns:
        None. Blocks while the Gradio server is running.

    Examples:
        ```shell
        python app.py --demo-dir exp/train_vits_libritts/demo
        ```
    """
    parser = argparse.ArgumentParser(description="Launch the LibriTTS VITS demo.")
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
    logger.info("Starting LibriTTS VITS demo CLI | args=%s", args)
    demo_config_path = args.demo_config or (args.demo_dir / "demo.yaml")
    app = build_demo(args.demo_dir, demo_config_path=demo_config_path)
    logger.info("Launching Gradio app")
    app.launch()


if __name__ == "__main__":
    main()
