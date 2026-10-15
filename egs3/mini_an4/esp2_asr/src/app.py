"""Recipe-local Gradio launcher using the native ASR backend directly."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import gradio as gr
import librosa
import yaml
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

from espnet2.torch_utils.resolve_device import resolve_device
from espnet3.publication.demo.assets import DEFAULT_UI_ASSETS, DefaultAudioUI
from espnet3.publication.demo.session import DemoSession
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def _bundle_paths(value, pack_dir, key=""):
    """Resolve existing recipe artifacts inside native ASR/LM YAML configs."""
    if isinstance(value, dict):
        return {
            name: _bundle_paths(item, pack_dir, name) for name, item in value.items()
        }
    if isinstance(value, list):
        return [_bundle_paths(item, pack_dir) for item in value]
    is_path = key in {"token_list", "bpemodel"} or key.endswith(
        ("_file", "_path", "_dir", "_config")
    )
    if is_path and isinstance(value, str) and value and not Path(value).is_absolute():
        candidate = pack_dir / value
        if candidate.exists():
            return str(candidate.resolve())
    return value


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
    # Packed configs retain Hydra controls; native constructors do not take them.
    for key in ("_convert_", "_recursive_"):
        parameters.pop(key, None)
    if target == "espnet2.bin.asr_inference.Speech2Text":
        from espnet2.bin.asr_inference import Speech2Text
    elif target == "espnet2.bin.asr_transducer_inference.Speech2Text":
        from espnet2.bin.asr_transducer_inference import Speech2Text

        parameters["return_decoded_hyp"] = True
    else:
        raise ValueError(f"The ASR demo does not support backend {target!r}")
    parameters["device"] = resolve_device(model_config.get("device", "auto"))
    # Native task configs use plain YAML, so bind their relative artifact paths
    # before construction. This also keeps lazily loaded tokenizer paths valid.
    with TemporaryDirectory(prefix="espnet-asr-demo-") as tmp_dir:
        for key, weights_key in (
            ("asr_train_config", "asr_model_file"),
            ("lm_train_config", "lm_file"),
        ):
            path = Path(parameters[key]) if parameters.get(key) else None
            if path is None and parameters.get(weights_key):
                path = Path(parameters[weights_key]).parent / "config.yaml"
            if path is None or not path.is_file():
                continue
            native_config = yaml.safe_load(path.read_text(encoding="utf-8"))
            resolved = _bundle_paths(native_config, pack_dir)
            if resolved != native_config:
                bound_path = Path(tmp_dir) / f"{key}.yaml"
                bound_path.write_text(yaml.safe_dump(resolved), encoding="utf-8")
                parameters[key] = str(bound_path)
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


class _ASRAudioUI(DefaultAudioUI):
    """Give the native ASR recipe a file path for its audio loader."""

    def build_input(self, spec):
        """Keep the configured label and let Gradio save uploaded audio."""
        return self.gradio_module.Audio(type="filepath", label=spec["label"])


def build_demo(
    demo_dir: Path,
    demo_config_path: Path | None = None,
):
    """Build the default Gradio Blocks app for one packed demo."""
    demo_dir = Path(demo_dir).resolve()
    if demo_config_path is None:
        demo_config_path = demo_dir / "demo.yaml"
    logger.info(
        "Building recipe demo UI | demo_dir=%s demo_config_path=%s",
        demo_dir,
        demo_config_path,
    )
    demo_config_path = Path(demo_config_path)
    if not demo_config_path.is_absolute():
        demo_config_path = demo_dir / demo_config_path
    config = load_config_with_defaults(str(demo_config_path))
    sample_rate = int(config.model.get("sample_rate", 16000))
    if sample_rate <= 0:
        raise ValueError("model.sample_rate must be positive")
    predict = create_predict(build_model(demo_dir, config.model), sample_rate)

    def decode(*, speech):
        return {"text": predict(speech)}

    registry = DEFAULT_UI_ASSETS.clone()
    registry.register("audio", _ASRAudioUI, replace=True)
    session = DemoSession(demo_dir, config, decode, registry)
    logger.info(
        "Resolved demo specs | inputs=%s outputs=%s",
        session.input_specs,
        session.output_specs,
    )
    inference_fn = session.create_inference_fn(
        session.input_specs,
        session.output_specs,
    )

    with gr.Blocks(title=session.title or "Speech recognition") as app:
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
