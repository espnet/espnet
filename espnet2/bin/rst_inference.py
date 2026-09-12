#!/usr/bin/env python3
"""Run an ESPnet restoration predictor with a Sidon vocoder.

The vocoder is either the official TorchScript decoder (--sidon_vocoder) or
one trained by recipe stages 7-8 (--vocoder_train_config, --vocoder_model_file).
"""

import argparse
import io
import logging
import subprocess
from argparse import Namespace
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from espnet2.tasks.rst import RestorationTask

logger = logging.getLogger(__name__)


def _load_feature_predictor(config_path, model_path, device):
    """Load native Sidon and legacy SpeechCleaner FP checkpoints."""
    import yaml

    with open(config_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    ssl_conf = dict(config.get("ssl_encoder_conf") or {})
    # keys of the legacy SpeechCleaner configs that this recipe does not have
    for key in (
        "use_flash_attention",
        "use_multilayer_loss",
        "multilayer_mode",
        "use_bf16",
    ):
        ssl_conf.pop(key, None)
    task_args = Namespace(
        ssl_encoder=config.get("ssl_encoder", "w2v_bert2"),
        ssl_encoder_conf=ssl_conf,
        lora_rank=config.get("lora_rank", 64),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.1),
        input_sr=config.get("input_sr", 16000),
    )
    model = RestorationTask.build_model(task_args)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    expected = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in expected and expected[key].shape == value.shape
    }
    # A checkpoint tensor whose name matches the model but whose shape does not
    # is a different failure from one the model simply does not have. Because
    # load_state_dict(strict=False) leaves unmatched parameters at their
    # initialised values, a silent shape mismatch would run inference on random
    # weights, so it is an error. Names absent from the model stay a warning:
    # that is how legacy and adapter-only checkpoints are meant to load.
    mismatched = sorted(
        key
        for key in state
        if key in expected and expected[key].shape != state[key].shape
    )
    if mismatched:
        detail = ", ".join(
            f"{key}: checkpoint {tuple(state[key].shape)} vs model "
            f"{tuple(expected[key].shape)}"
            for key in mismatched[:5]
        )
        raise RuntimeError(
            f"{len(mismatched)} checkpoint tensors have shapes incompatible "
            f"with the configured model and would be silently replaced by "
            f"randomly initialised weights: {detail}"
            + (" ..." if len(mismatched) > 5 else "")
        )
    unused = sorted(set(state) - set(compatible))
    missing = sorted(set(expected) - set(compatible))
    model.load_state_dict(compatible, strict=False)
    missing_lora = [key for key in missing if "lora_" in key]
    if missing_lora:
        raise RuntimeError(
            "Checkpoint is incompatible with the configured LoRA architecture; "
            f"missing {len(missing_lora)} LoRA tensors. Use its original "
            "lora_rank and lora_alpha configuration."
        )
    if unused or missing:
        logger.warning(
            "Loaded compatible legacy tensors; "
            "unused_in_checkpoint=%d, left_at_init=%d",
            len(unused),
            len(missing),
        )
    return model.eval().to(device)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        help="YAML of default option values (conf/decode.yaml). "
        "Explicit command-line flags take precedence.",
    )
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--model_file", required=True)
    parser.add_argument(
        "--sidon_vocoder",
        default=None,
        help="Official TorchScript decoder_cpu.pt or decoder_cuda.pt",
    )
    parser.add_argument(
        "--vocoder_train_config",
        default=None,
        help="config.yaml of a vocoder trained by stages 7-8 "
        "(alternative to --sidon_vocoder)",
    )
    parser.add_argument(
        "--vocoder_model_file",
        default=None,
        help="checkpoint of that vocoder, e.g. valid.loss_mel.best.pth",
    )
    parser.add_argument("--wav_scp", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk_sec", type=float, default=20.0)
    parser.add_argument("--overlap_sec", type=float, default=0.5)
    return parser


def _load_vocoder(args, input_dim, device):
    """Either vocoder as a callable (B, D, T) -> (B, 1, T * 960)."""
    espnet_args = (args.vocoder_train_config, args.vocoder_model_file)
    if args.sidon_vocoder and any(espnet_args):
        raise ValueError(
            "give either --sidon_vocoder or --vocoder_train_config with "
            "--vocoder_model_file, not both"
        )
    if args.sidon_vocoder:
        return torch.jit.load(args.sidon_vocoder, map_location=device).eval()
    if not all(espnet_args):
        raise ValueError(
            "a vocoder is required: --sidon_vocoder (official TorchScript) or "
            "--vocoder_train_config and --vocoder_model_file (stages 7-8)"
        )
    import yaml

    from espnet2.rst.decoder.dac_vocoder import build_vocoder

    with open(args.vocoder_train_config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    vocoder = build_vocoder(
        config.get("vocoder_type", "dac"), input_dim, config.get("vocoder_conf")
    )
    checkpoint = torch.load(
        args.vocoder_model_file, map_location="cpu", weights_only=True
    )
    state = checkpoint.get("model", checkpoint)
    prefix = "vocoder."
    state = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    if not state:
        raise RuntimeError(f"no 'vocoder.*' tensors in {args.vocoder_model_file}")
    # Training checkpoints hold weight-normalised convolutions; a checkpoint
    # saved after folding holds plain weights. Load whichever this is, then
    # fold for inference.
    if not any(k.endswith("weight_g") for k in state):
        vocoder.remove_weight_norm()
    vocoder.load_state_dict(state, strict=True)
    vocoder.remove_weight_norm()
    logger.info("loaded ESPnet-trained vocoder from %s", args.vocoder_model_file)
    return vocoder.eval().to(device)


def _read_audio(value: str, sample_rate: int = 16000):
    value = value.strip()
    if value.endswith("|"):
        process = subprocess.run(
            value[:-1], shell=True, check=True, stdout=subprocess.PIPE
        )
        waveform, source_rate = sf.read(
            io.BytesIO(process.stdout), dtype="float32", always_2d=True
        )
    else:
        waveform, source_rate = sf.read(value, dtype="float32", always_2d=True)
    waveform = waveform.mean(1)
    if source_rate != sample_rate:
        import torchaudio.functional as AF

        waveform = AF.resample(
            torch.from_numpy(waveform), source_rate, sample_rate
        ).numpy()
    return waveform


@torch.inference_mode()
def _restore_chunk(waveform, model, vocoder, device):
    wav_tensor = torch.from_numpy(waveform).float().to(device)
    lengths = torch.tensor([len(waveform)], device=device)
    ssl_inputs = model.ssl_encoder._wav_to_ssl_inputs(wav_tensor.unsqueeze(0), lengths)
    features, _ = model.ssl_encoder(ssl_inputs)
    output = vocoder(features.transpose(1, 2))
    return output.reshape(-1).float().cpu().numpy()


def _restore(waveform, model, vocoder, device, chunk_sec, overlap_sec):
    chunk = int(chunk_sec * 16000)
    overlap = int(overlap_sec * 16000)
    if len(waveform) <= chunk:
        return _restore_chunk(waveform, model, vocoder, device)
    hop = chunk - overlap
    output_length = int(len(waveform) * 3)
    output = np.zeros(output_length, np.float32)
    weight = np.zeros(output_length, np.float32)
    for start in range(0, len(waveform), hop):
        piece = waveform[start : start + chunk]
        if len(piece) < 1600:
            continue
        restored = _restore_chunk(piece, model, vocoder, device)
        destination = start * 3
        restored = restored[: output_length - destination]
        envelope = np.ones(len(restored), np.float32)
        # fade == 0 (--overlap_sec 0) means plain concatenation: envelope[-0:]
        # would select the whole array, so the fades are only applied when > 0.
        fade = min(overlap * 3, len(restored))
        if fade > 0 and start:
            envelope[:fade] = np.linspace(0, 1, fade)
        if fade > 0 and start + chunk < len(waveform):
            envelope[-fade:] = np.minimum(envelope[-fade:], np.linspace(1, 0, fade))
        output[destination : destination + len(restored)] += restored * envelope
        weight[destination : destination + len(restored)] += envelope
    valid = weight > 1e-6
    output[valid] /= weight[valid]
    return output


def main(cmd=None):
    import yaml

    parser = get_parser()
    args = parser.parse_args(cmd)
    if args.config is not None:
        with open(args.config, encoding="utf-8") as stream:
            defaults = yaml.safe_load(stream) or {}
        unknown = set(defaults) - set(vars(args))
        if unknown:
            raise ValueError(f"unknown keys in {args.config}: {sorted(unknown)}")
        # set_defaults then reparse, so anything given on the command line
        # still overrides the file.
        parser.set_defaults(**defaults)
        args = parser.parse_args(cmd)
    logging.basicConfig(level=logging.INFO)
    device = args.device if torch.cuda.is_available() else "cpu"
    model = _load_feature_predictor(args.train_config, args.model_file, device)
    vocoder = _load_vocoder(args, model.ssl_encoder.ssl_dim, device)

    output = Path(args.output_dir)
    wav_dir = output / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    # Progress goes to wav.scp.partial one line per finished utterance, and
    # wav.scp is only published once every input is done, so an interrupted
    # run leaves no manifest that a later stage could mistake for complete.
    # Rerunning into the same output_dir resumes from the partial manifest.
    partial = output / "wav.scp.partial"
    done = {}
    if partial.is_file():
        with open(partial, encoding="utf-8") as stream:
            for line in stream:
                utterance, path = line.rstrip().split(maxsplit=1)
                if Path(path).is_file():
                    done[utterance] = path
        if done:
            logger.info("Resuming: %d utterances already restored", len(done))
    with open(args.wav_scp, encoding="utf-8") as stream:
        inputs = [line.rstrip().split(maxsplit=1) for line in stream if line.strip()]
    with open(partial, "a", encoding="utf-8") as manifest:
        for utterance, source in inputs:
            if utterance in done:
                continue
            waveform = _read_audio(source)
            restored = _restore(
                waveform,
                model,
                vocoder,
                device,
                args.chunk_sec,
                args.overlap_sec,
            )
            path = wav_dir / f"{utterance}.wav"
            sf.write(path, restored, 48000)
            manifest.write(f"{utterance} {path}\n")
            manifest.flush()
            done[utterance] = str(path)
    with open(output / "wav.scp", "w", encoding="utf-8") as stream:
        stream.writelines(f"{utterance} {done[utterance]}\n" for utterance, _ in inputs)
    logger.info("Restored %d utterances", len(inputs))


if __name__ == "__main__":
    main()
