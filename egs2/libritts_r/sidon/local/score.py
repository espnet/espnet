#!/usr/bin/env python3
"""Sidon evaluation script.

Metrics following the paper:
  WER     — word error rate via mms-1b-all ASR model
  DNSMOS  — P.835 overall MOS estimate (microsoft/DNSMOS)
  NISQA   — neural speech quality assessment
  SpkSim  — cosine similarity of speaker embeddings (wavlm-base-plus-sv)

Usage
-----
python local/score.py \
    --restored_dir  exp/restored/test-other \
    --ref_wav_scp   data/test-other/wav.scp \
    --noisy_wav_scp data/test-other_noisy/noisy/wav.scp \
    --output_dir    exp/scores/test-other \
    --nj            8
"""

import argparse
import json
import logging
import os
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers: load wav.scp
# ---------------------------------------------------------------------------

def load_wav_scp(path: str) -> Dict[str, str]:
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                d[parts[0]] = parts[1]
    return d


def read_wav(path: str, target_sr: int = None) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if target_sr is not None and sr != target_sr:
        import torchaudio.functional as AF
        wav = AF.resample(
            torch.from_numpy(wav), sr, target_sr
        ).numpy()
    return wav


# ---------------------------------------------------------------------------
# DNSMOS
# ---------------------------------------------------------------------------

def compute_dnsmos(wav_paths: List[str], sr: int = 16000) -> Dict[str, float]:
    """Compute DNSMOS P.835 OVRL scores.

    Requires: pip install requests  (uses DNSMOS REST API or local model).
    Falls back to torchDNSMOS if available.
    """
    try:
        from torchDNSMOS import DNSMOS as DNSMOSModel
        model = DNSMOSModel()
        scores = {}
        for path in wav_paths:
            uttid = os.path.splitext(os.path.basename(path))[0]
            wav   = read_wav(path, sr)
            score = model(wav, sr)
            scores[uttid] = float(score["ovrl"])
        return scores
    except ImportError:
        logger.warning(
            "torchDNSMOS not found; DNSMOS skipped. "
            "Install with: pip install torchDNSMOS"
        )
        return {}


# ---------------------------------------------------------------------------
# NISQA
# ---------------------------------------------------------------------------

def compute_nisqa(wav_paths: List[str]) -> Dict[str, float]:
    """Compute NISQA overall quality scores."""
    try:
        from nisqa.NISQA_model import nisqaModel
    except ImportError:
        logger.warning("nisqa not found; NISQA skipped.")
        return {}

    scores = {}
    for path in wav_paths:
        uttid = os.path.splitext(os.path.basename(path))[0]
        try:
            # nisqa expects a single file path and returns a dict
            result = nisqaModel(path)
            scores[uttid] = float(result.get("mos_pred", result.get("mos", 0.0)))
        except Exception as e:
            logger.debug("NISQA failed for %s: %s", path, e)
    return scores


# ---------------------------------------------------------------------------
# Speaker Similarity
# ---------------------------------------------------------------------------

def compute_spk_sim(
    restored_paths: List[str],
    noisy_paths: List[str],
) -> Dict[str, float]:
    """Cosine similarity between speaker embeddings (noisy vs restored).

    Uses wavlm-base-plus-sv from HuggingFace following the Sidon paper.
    """
    try:
        from transformers import Wav2Vec2FeatureExtractor, WavLMModel
    except ImportError:
        logger.warning("transformers not installed; SpkSim skipped.")
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "microsoft/wavlm-base-plus-sv"
    logger.info("Loading %s for speaker similarity...", model_id)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    model     = WavLMModel.from_pretrained(model_id).eval().to(device)

    def embed(path):
        wav = read_wav(path, target_sr=16000)
        inp = extractor(wav, sampling_rate=16000, return_tensors="pt",
                        padding=True)
        with torch.no_grad():
            out = model(inp["input_values"].to(device))
        # Mean-pool last hidden state as speaker embedding
        emb = out.last_hidden_state.mean(dim=1)
        return torch.nn.functional.normalize(emb, dim=-1).cpu()

    scores = {}
    for rp, np_ in zip(restored_paths, noisy_paths):
        uttid = os.path.splitext(os.path.basename(rp))[0]
        try:
            e_r = embed(rp)
            e_n = embed(np_)
            sim = (e_r * e_n).sum().item()
            scores[uttid] = sim
        except Exception as e:
            logger.debug("SpkSim failed for %s: %s", uttid, e)
    return scores


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------

def compute_wer(
    restored_paths: List[str],
    ref_texts: Dict[str, str],
) -> Dict[str, float]:
    """Compute WER using facebook/mms-1b-all (covers 1162 languages)."""
    try:
        from transformers import AutoProcessor, Wav2Vec2ForCTC
    except ImportError:
        logger.warning("transformers not installed; WER skipped.")
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "facebook/mms-1b-all"
    logger.info("Loading %s for WER...", model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model     = Wav2Vec2ForCTC.from_pretrained(model_id).eval().to(device)

    import editdistance

    total_err, total_ref = 0, 0
    per_utt = {}

    for path in restored_paths:
        uttid = os.path.splitext(os.path.basename(path))[0]
        if uttid not in ref_texts:
            continue
        wav = read_wav(path, target_sr=16000)
        inp = processor(wav, sampling_rate=16000, return_tensors="pt",
                        padding=True)
        with torch.no_grad():
            logits = model(inp["input_values"].to(device)).logits
        pred_ids  = torch.argmax(logits, dim=-1)
        hyp = processor.batch_decode(pred_ids)[0].lower().split()
        ref = ref_texts[uttid].lower().split()
        err = editdistance.eval(hyp, ref)
        per_utt[uttid] = err / max(len(ref), 1)
        total_err += err
        total_ref += len(ref)

    overall_wer = total_err / max(total_ref, 1)
    logger.info("WER = %.4f (%d / %d)", overall_wer, total_err, total_ref)
    return per_utt, overall_wer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_parser():
    p = argparse.ArgumentParser(description="Sidon evaluation")
    p.add_argument("--restored_dir",  required=True)
    p.add_argument("--ref_wav_scp",   required=True)
    p.add_argument("--noisy_wav_scp", required=True)
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--text",          default=None,
                   help="Optional Kaldi text file for WER computation")
    p.add_argument("--nj",            type=int, default=8)
    return p


def main():
    args = get_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect restored WAV paths
    restored_files = sorted([
        os.path.join(args.restored_dir, f)
        for f in os.listdir(args.restored_dir)
        if f.endswith(".wav")
    ])
    if not restored_files:
        logger.error("No WAV files found in %s", args.restored_dir)
        return

    logger.info("Evaluating %d utterances", len(restored_files))

    noisy_scp = load_wav_scp(args.noisy_wav_scp)
    noisy_paths = []
    for rpath in restored_files:
        uttid = os.path.splitext(os.path.basename(rpath))[0]
        if uttid in noisy_scp:
            noisy_paths.append(noisy_scp[uttid])
        else:
            noisy_paths.append(rpath)  # fallback

    results = {}

    # DNSMOS
    logger.info("Computing DNSMOS...")
    dnsmos = compute_dnsmos(restored_files)
    if dnsmos:
        avg = float(np.mean(list(dnsmos.values())))
        results["dnsmos_mean"] = avg
        logger.info("DNSMOS (restored) = %.4f", avg)

    # NISQA
    logger.info("Computing NISQA...")
    nisqa = compute_nisqa(restored_files)
    if nisqa:
        avg = float(np.mean(list(nisqa.values())))
        results["nisqa_mean"] = avg
        logger.info("NISQA (restored) = %.4f", avg)

    # SpkSim
    logger.info("Computing SpkSim...")
    spksim = compute_spk_sim(restored_files, noisy_paths)
    if spksim:
        avg = float(np.mean(list(spksim.values())))
        results["spksim_mean"] = avg
        logger.info("SpkSim = %.4f", avg)

    # WER (optional)
    if args.text is not None:
        ref_texts = {}
        with open(args.text) as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    ref_texts[parts[0]] = parts[1]
        logger.info("Computing WER...")
        per_utt_wer, overall_wer = compute_wer(restored_files, ref_texts)
        results["wer"] = overall_wer

    # Save results
    out_json = os.path.join(args.output_dir, "scores.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_json)

    # Print summary
    print("\n=== Evaluation Summary ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    print("==========================\n")


if __name__ == "__main__":
    main()
