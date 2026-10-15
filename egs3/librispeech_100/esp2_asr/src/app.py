"""Recipe-local Gradio launcher using the native ASR backend directly."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import gradio as gr
import librosa
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

from espnet2.torch_utils.resolve_device import resolve_device
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.logging_utils import configure_logging


def build_model(demo_dir: Path, model_config):
    """Build the ASR backend configured in a local or Hub recipe archive."""
    reference = str(model_config.dir_or_tag)
    pack_dir = Path(reference).expanduser()
    if not pack_dir.is_absolute():
        pack_dir = demo_dir / pack_dir
    if not pack_dir.is_dir():
        if Path(reference).is_absolute():
            raise FileNotFoundError(pack_dir)
        pack_dir = Path(snapshot_download(repo_id=reference))
    pack_dir = pack_dir.resolve()
    config = load_config_with_defaults(
        str(pack_dir / "conf" / "inference.yaml"), resolve=False
    )
    config.recipe_dir = str(pack_dir)
    parameters = OmegaConf.to_container(config.model, resolve=True)
    target = parameters.pop("_target_")
    if target == "espnet2.bin.asr_inference.Speech2Text":
        from espnet2.bin.asr_inference import Speech2Text
    elif target == "espnet2.bin.asr_transducer_inference.Speech2Text":
        from espnet2.bin.asr_transducer_inference import Speech2Text

        parameters["return_decoded_hyp"] = True
    else:
        raise ValueError(f"The ASR demo does not support backend {target!r}")
    parameters["device"] = resolve_device(model_config.get("device", "auto"))
    return Speech2Text(**parameters)


def create_predict(model, sample_rate: int):
    """Read uploaded audio at the configured rate and return the best ASR text."""

    def predict(audio_path):
        if audio_path is None:
            raise gr.Error("Record or upload audio first.")
        speech, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
        best = model(speech)[0]
        if isinstance(best, list):
            best = best[0]
        return best[0] or ""

    return predict


def build_demo(demo_dir: Path, demo_config_path: Path | None = None):
    """Build an audio-to-text app from a packed demo's explicit configuration."""
    demo_dir = Path(demo_dir).resolve()
    config_path = Path(demo_config_path) if demo_config_path else Path("demo.yaml")
    if not config_path.is_absolute():
        config_path = demo_dir / config_path
    config = load_config_with_defaults(str(config_path))
    sample_rate = int(config.model.get("sample_rate", 16000))
    if sample_rate <= 0:
        raise ValueError("model.sample_rate must be positive")
    model = build_model(demo_dir, config.model)
    title = config.ui.get("title") or "Speech recognition"
    description = config.ui.get("description")
    if description:
        path = Path(str(description))
        if not path.is_absolute():
            path = demo_dir / path
        if path.is_file():
            description = path.read_text(encoding="utf-8")
            description = re.sub(
                r"^---\s*\n.*?\n---\s*\n", "", description, count=1, flags=re.DOTALL
            )
    input_spec = next(
        (spec for spec in config.ui.get("inputs", []) if spec.get("key") == "speech"),
        {},
    )
    output_spec = next(
        (spec for spec in config.ui.get("outputs", []) if spec.get("key") == "text"),
        {},
    )
    with gr.Blocks(title=title) as app:
        gr.Markdown(f"# {title}")
        audio = gr.Audio(type="filepath", label=input_spec.get("label", "Input audio"))
        submit = gr.Button("Run")
        text = gr.Textbox(label=output_spec.get("label", "Transcription"))
        if description:
            gr.Markdown(str(description))
        submit.click(create_predict(model, sample_rate), inputs=audio, outputs=text)
    return app


def main() -> None:
    """Launch this script from either a recipe or a packed demo directory."""
    parser = argparse.ArgumentParser(description="Launch an ESPnet3 ASR demo.")
    parser.add_argument(
        "--demo-dir", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--demo-config", type=Path, default=None)
    args = parser.parse_args()
    configure_logging(log_dir=args.demo_dir, filename="demo.log")
    build_demo(args.demo_dir, args.demo_config).launch()


if __name__ == "__main__":
    main()
