#!/usr/bin/env python3
"""Sidon evaluation script: the four numbers reported in the paper.

Metrics following the paper:
  WER     — word error rate via mms-1b-all ASR model
  DNSMOS  — P.835 overall MOS estimate (microsoft/DNSMOS)
  NISQA   — neural speech quality assessment
  SpkSim  — cosine similarity of speaker embeddings (wavlm-base-plus-sv)

VERSA (recipe stage 11, ``conf/versa_enh.yaml`` reference-free and
``conf/versa_enh_ref_based.yaml`` reference-based) scores the same
restorations with these metrics and many more variants (UTMOS, SQUIM, PESQ,
STOI, SDR/SI-SNR, ...) in one pass and is the recommended scorer. Keep this
script for a dependency-light reproduction of exactly the paper's table.

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

        wav = AF.resample(torch.from_numpy(wav), sr, target_sr).numpy()
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
            wav = read_wav(path, sr)
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


def compute_nisqa(
    wav_dir: str,
    nisqa_model: str = None,
    output_dir: str = None,
) -> Dict[str, float]:
    """Compute NISQA MOS predictions for every WAV in ``wav_dir``.

    NISQA is not pip-installable and is not an ESPnet dependency: it has to be
    cloned from github.com/gabrielmittag/NISQA and put on PYTHONPATH, and it
    needs an explicit weights file (``weights/nisqa.tar``). Its public entry
    point takes an argument dict and is driven through ``predict()`` -- it is
    not callable on a path -- and one ``predict_dir`` pass over the directory
    is far cheaper than reloading the model per utterance.

    Returns an empty dict when NISQA is unavailable. VERSA (stage 8) already
    reports neural MOS predictors, so NISQA here is strictly optional.
    """
    if not nisqa_model:
        logger.info("--nisqa_model not set; NISQA skipped.")
        return {}
    if not os.path.isfile(nisqa_model):
        logger.warning("NISQA weights not found at %s; NISQA skipped.", nisqa_model)
        return {}
    try:
        from nisqa.NISQA_model import nisqaModel
    except ImportError:
        logger.warning(
            "nisqa not importable; NISQA skipped. Clone "
            "github.com/gabrielmittag/NISQA and add it to PYTHONPATH."
        )
        return {}

    nisqa_args = {
        "mode": "predict_dir",
        "pretrained_model": nisqa_model,
        "data_dir": wav_dir,
        "output_dir": output_dir,
        "csv_file": None,
        "csv_deg": None,
        "num_workers": 0,
        "bs": 1,
        "ms_channel": None,
    }
    # Deliberately not wrapped in a bare except: if weights were supplied and
    # NISQA still fails, that is a configuration error worth surfacing rather
    # than a silently empty column in the results table.
    frame = nisqaModel(nisqa_args).predict()
    scores = {}
    for _, row in frame.iterrows():
        uttid = os.path.splitext(os.path.basename(str(row["deg"])))[0]
        scores[uttid] = float(row["mos_pred"])
    logger.info("NISQA scored %d utterances", len(scores))
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
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    except ImportError:
        logger.warning("transformers not installed; SpkSim skipped.")
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "microsoft/wavlm-base-plus-sv"
    logger.info("Loading %s for speaker similarity...", model_id)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    model = WavLMForXVector.from_pretrained(model_id).eval().to(device)

    def embed(path):
        wav = read_wav(path, target_sr=16000)
        inp = extractor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(inp["input_values"].to(device))
        emb = out.embeddings
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
    model = Wav2Vec2ForCTC.from_pretrained(model_id).eval().to(device)

    import editdistance

    total_err, total_ref = 0, 0
    per_utt = {}

    for path in restored_paths:
        uttid = os.path.splitext(os.path.basename(path))[0]
        if uttid not in ref_texts:
            continue
        wav = read_wav(path, target_sr=16000)
        inp = processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inp["input_values"].to(device)).logits
        pred_ids = torch.argmax(logits, dim=-1)
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
    p.add_argument("--restored_dir", required=True)
    p.add_argument("--ref_wav_scp", required=True)
    p.add_argument("--noisy_wav_scp", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--text", default=None, help="Optional Kaldi text file for WER computation"
    )
    p.add_argument(
        "--nisqa_model",
        default=None,
        help="Path to NISQA weights (e.g. NISQA/weights/nisqa.tar). "
        "NISQA is optional and is skipped when unset.",
    )
    p.add_argument("--nj", type=int, default=8)
    return p


def main():
    args = get_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect restored WAV paths
    restored_files = sorted(
        [
            os.path.join(args.restored_dir, f)
            for f in os.listdir(args.restored_dir)
            if f.endswith(".wav")
        ]
    )
    if not restored_files:
        logger.error("No WAV files found in %s", args.restored_dir)
        return

    logger.info("Evaluating %d utterances", len(restored_files))

    # Speaker similarity needs a genuine (restored, input) pair. Falling back
    # to the restored file itself would embed the same audio on both sides and
    # fold a cosine of ~1.0 into the mean, silently inflating the score, so
    # unmatched utterances are dropped and counted instead.
    noisy_scp = load_wav_scp(args.noisy_wav_scp)
    spksim_restored, spksim_noisy, unmatched = [], [], []
    for rpath in restored_files:
        uttid = os.path.splitext(os.path.basename(rpath))[0]
        if uttid in noisy_scp:
            spksim_restored.append(rpath)
            spksim_noisy.append(noisy_scp[uttid])
        else:
            unmatched.append(uttid)
    if unmatched:
        logger.warning(
            "%d/%d utterances missing from %s; excluded from SpkSim: %s%s",
            len(unmatched),
            len(restored_files),
            args.noisy_wav_scp,
            ", ".join(unmatched[:5]),
            " ..." if len(unmatched) > 5 else "",
        )

    results = {}
    results["spksim_num_scored"] = len(spksim_restored)
    results["spksim_num_unmatched"] = len(unmatched)

    # DNSMOS
    logger.info("Computing DNSMOS...")
    dnsmos = compute_dnsmos(restored_files)
    if dnsmos:
        avg = float(np.mean(list(dnsmos.values())))
        results["dnsmos_mean"] = avg
        logger.info("DNSMOS (restored) = %.4f", avg)

    # NISQA
    logger.info("Computing NISQA...")
    nisqa = compute_nisqa(args.restored_dir, args.nisqa_model, args.output_dir)
    if nisqa:
        avg = float(np.mean(list(nisqa.values())))
        results["nisqa_mean"] = avg
        logger.info("NISQA (restored) = %.4f", avg)

    # SpkSim
    logger.info("Computing SpkSim...")
    spksim = compute_spk_sim(spksim_restored, spksim_noisy)
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
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("==========================\n")


if __name__ == "__main__":
    main()
