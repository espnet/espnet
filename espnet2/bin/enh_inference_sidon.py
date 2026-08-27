#!/usr/bin/env python3
"""Run an ESPnet-Sidon predictor with the official Sidon vocoder."""

import argparse
import logging
import subprocess
from argparse import Namespace
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from espnet2.tasks.sidon import SidonTask

logger = logging.getLogger(__name__)


def _load_feature_predictor(config_path, model_path, device):
    """Load native Sidon and legacy SpeechCleaner FP checkpoints."""
    import yaml

    with open(config_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    ssl_conf = dict(config.get("ssl_encoder_conf") or {})
    for key in ("target_layer", "use_flash_attention", "use_multilayer_loss",
                "multilayer_mode", "use_bf16"):
        ssl_conf.pop(key, None)
    task_args = Namespace(
        ssl_encoder="w2v_bert2",
        ssl_encoder_conf=ssl_conf,
        lora_rank=config.get("lora_rank", 64),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.1),
        input_sr=config.get("input_sr", 16000),
    )
    model = SidonTask.build_model(task_args)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    expected = model.state_dict()
    compatible = {
        key: value for key, value in state.items()
        if key in expected and expected[key].shape == value.shape
    }
    skipped = sorted(set(state) - set(compatible))
    missing = sorted(set(expected) - set(compatible))
    model.load_state_dict(compatible, strict=False)
    missing_lora = [key for key in missing if "lora_" in key]
    if missing_lora:
        raise RuntimeError(
            "Checkpoint is incompatible with the configured LoRA architecture; "
            f"missing {len(missing_lora)} LoRA tensors. Use its original "
            "lora_rank and lora_alpha configuration."
        )
    if skipped or missing:
        logger.warning(
            "Loaded compatible legacy tensors; skipped=%d, missing=%d",
            len(skipped), len(missing),
        )
    return model.eval().to(device)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--sidon_vocoder", required=True,
                        help="Official decoder_cpu.pt or decoder_cuda.pt")
    parser.add_argument("--wav_scp", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk_sec", type=float, default=20.0)
    parser.add_argument("--overlap_sec", type=float, default=0.5)
    return parser


def _read_audio(value: str, sample_rate: int = 16000):
    value = value.strip()
    if value.endswith("|"):
        process = subprocess.run(
            value[:-1], shell=True, check=True, stdout=subprocess.PIPE
        )
        waveform = np.frombuffer(process.stdout, np.int16).astype(np.float32) / 32768
        return waveform
    waveform, source_rate = sf.read(value, dtype="float32", always_2d=True)
    waveform = waveform.mean(1)
    if source_rate != sample_rate:
        import torchaudio.functional as AF
        waveform = AF.resample(
            torch.from_numpy(waveform), source_rate, sample_rate
        ).numpy()
    return waveform


@torch.inference_mode()
def _restore_chunk(waveform, model, vocoder, processor, device):
    inputs = processor(
        [np.pad(waveform, (40, 40))],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    features, _ = model.ssl_encoder(inputs)
    output = vocoder(features.transpose(1, 2))
    return output.reshape(-1).float().cpu().numpy()


def _restore(waveform, model, vocoder, processor, device, chunk_sec, overlap_sec):
    chunk = int(chunk_sec * 16000)
    overlap = int(overlap_sec * 16000)
    if len(waveform) <= chunk:
        return _restore_chunk(waveform, model, vocoder, processor, device)
    hop = chunk - overlap
    output_length = int(len(waveform) * 3)
    output = np.zeros(output_length, np.float32)
    weight = np.zeros(output_length, np.float32)
    for start in range(0, len(waveform), hop):
        piece = waveform[start : start + chunk]
        if len(piece) < 1600:
            continue
        restored = _restore_chunk(piece, model, vocoder, processor, device)
        destination = start * 3
        restored = restored[: output_length - destination]
        envelope = np.ones(len(restored), np.float32)
        fade = min(overlap * 3, len(restored))
        if start:
            envelope[:fade] = np.linspace(0, 1, fade)
        if start + chunk < len(waveform):
            envelope[-fade:] = np.minimum(envelope[-fade:], np.linspace(1, 0, fade))
        output[destination : destination + len(restored)] += restored * envelope
        weight[destination : destination + len(restored)] += envelope
    valid = weight > 1e-6
    output[valid] /= weight[valid]
    return output


def main(cmd=None):
    args = get_parser().parse_args(cmd)
    logging.basicConfig(level=logging.INFO)
    device = args.device if torch.cuda.is_available() else "cpu"
    model = _load_feature_predictor(
        args.train_config, args.model_file, device
    )
    vocoder = torch.jit.load(args.sidon_vocoder, map_location=device).eval()
    from transformers import SeamlessM4TFeatureExtractor
    processor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    output = Path(args.output_dir)
    wav_dir = output / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    scp = []
    with open(args.wav_scp, encoding="utf-8") as stream:
        for line in stream:
            utterance, source = line.rstrip().split(maxsplit=1)
            waveform = _read_audio(source)
            restored = _restore(
                waveform, model, vocoder, processor, device,
                args.chunk_sec, args.overlap_sec,
            )
            path = wav_dir / f"{utterance}.wav"
            sf.write(path, restored, 48000)
            scp.append(f"{utterance} {path}\n")
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "wav.scp", "w", encoding="utf-8") as stream:
        stream.writelines(scp)
    logger.info("Restored %d utterances", len(scp))


if __name__ == "__main__":
    main()
