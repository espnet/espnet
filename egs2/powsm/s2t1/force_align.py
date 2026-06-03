"""Forced alignment using ESPnet Speech2Text model.

Usage:
    python force_align.py \
        --config_path config.yaml \
        --model_path valid.acc.ave_5best.till40epoch.pth \
        --bpe_model_path bpe.model \
        --audio_path recording/s01_10-5178.flac \
        --text "/k//l//ʌ///ɹ//e//ɪ//n//i//ŋ/"

Output:
    Alignment Results:
--------------------------------------------------
▁                        0.00ms -   100.00ms
/k/                    220.00ms -   240.00ms
/l/                    260.00ms -   280.00ms
...
/i/                   3200.00ms -  3220.00ms
/ŋ/                   3320.00ms -  3340.00ms
"""

import argparse

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from espnet2.bin.s2t_inference import Speech2Text
from espnet2.torch_utils.device_funcs import to_device


def load_model(config_path, model_path, bpe_model_path, device="cpu"):
    """
    Load the Speech2Text model for forced alignment.

    Args:
        config_path: Path to config.yaml
        model_path: Path to model checkpoint (.pth)
        bpe_model_path: Path to BPE model
        device: "cpu" or "cuda"

    Returns:
        Speech2Text model instance
    """
    model = Speech2Text(
        s2t_train_config=config_path,
        s2t_model_file=model_path,
        bpemodel=bpe_model_path,
        beam_size=1,
        ctc_weight=0.3,  # dummy, we only use ctc posteriors
        device=device,
    )
    return model


def prepare_speech(speech, model, device):
    """
    Prepare speech tensor for model input.

    Args:
        speech: Audio waveform (numpy array or torch tensor)
        model: Speech2Text model instance
        device: Device to place tensor on

    Returns:
        Tuple of (speech_tensor, speech_lengths)
    """
    if isinstance(speech, np.ndarray):
        speech = torch.tensor(speech)

    if speech.dim() > 1:
        assert (
            speech.dim() == 2 and speech.size(1) == 1
        ), f"Speech of size {speech.size()} is not supported"
        speech = speech.squeeze(1)

    speech_length = int(
        model.preprocessor_conf["fs"] * model.preprocessor_conf["speech_length"]
    )
    original_length = speech.size(-1)

    if original_length >= speech_length:
        speech = speech[:speech_length]
    else:
        speech = F.pad(speech, (0, speech_length - original_length))
    speech = speech.unsqueeze(0).to(getattr(torch, model.dtype))
    speech_lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.shape[1])
    return speech, speech_lengths


def prepare_text(text, model, device):
    """
    Prepare text tensor for model input.

    Args:
        text: Text string to align
        model: Speech2Text model instance
        device: Device to place tensor on

    Returns:
        Tuple of (text_tensor, text_lengths)
    """
    tokens = model.tokenizer.text2tokens(text)
    token_ids = model.converter.tokens2ids(tokens)
    text_tensor = torch.tensor([token_ids], device=device)
    text_lengths = text_tensor.new_full(
        [1], dtype=torch.long, fill_value=text_tensor.shape[1]
    )
    return text_tensor, text_lengths


def forced_align(speech, text, model, device="cpu", time_hop=0.02):
    """
    Perform forced alignment between speech and text.

    Args:
        speech: Audio waveform (numpy array or torch tensor)
        text: Text string to align with speech
        model: Speech2Text model instance
        device: Device to run inference on
        time_hop: Time hop in seconds per frame (default: 0.02)

    Returns:
        List of tuples: [(token, [start_ms, end_ms]), ...]
    """
    speech_tensor, speech_lengths = prepare_speech(speech, model, device)
    text_tensor, text_lengths = prepare_text(text, model, device)
    batch = {
        "speech": speech_tensor,
        "speech_lengths": speech_lengths,
        "text": text_tensor,
        "text_lengths": text_lengths,
    }
    batch = to_device(batch, device)
    align_label, align_score = model.s2t_model.forced_align(**batch)
    align_label_spans = torchaudio.functional.merge_tokens(
        align_label[0], align_score[0]
    )
    alignments = []
    for span in align_label_spans:
        token = model.converter.ids2tokens([span.token])[0]
        start_time_ms = span.start * time_hop * 1000
        end_time_ms = span.end * time_hop * 1000
        alignments.append((token, [start_time_ms, end_time_ms]))
    return alignments


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    argparser = argparse.ArgumentParser(
        description="Forced alignment using ESPnet Speech2Text model"
    )
    argparser.add_argument("--config_path", type=str, help="Path to config.yaml")
    argparser.add_argument(
        "--model_path", type=str, help="Path to model checkpoint (.pth)"
    )
    argparser.add_argument("--bpe_model_path", type=str, help="Path to BPE model")
    argparser.add_argument(
        "--audio_path", type=str, default="test.wav", help="Path to input audio file"
    )
    argparser.add_argument(
        "--text", type=str, default="hello world", help="Text to align with audio"
    )
    args = argparser.parse_args()

    config_path = args.config_path
    model_path = args.model_path
    bpe_model_path = args.bpe_model_path
    audio_path = args.audio_path
    text = args.text

    print(f"Loading model on {device}...")
    model = load_model(config_path, model_path, bpe_model_path, device)
    speech, sample_rate = sf.read(audio_path)
    print(f"Audio sample rate: {sample_rate}, waveform shape: {speech.shape}")
    alignments = forced_align(speech, text, model, device)
    print("\nAlignment Results:")
    print("-" * 50)
    for token, (start_ms, end_ms) in alignments:
        print(f"{token:20s} {start_ms:8.2f}ms - {end_ms:8.2f}ms")
