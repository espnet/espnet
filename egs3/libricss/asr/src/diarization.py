"""Diarization stage for the LibriCSS recipe.

Reads the segment manifests written by the ``segment`` stage
(``${exp_dir}/segments/<split>/<reco>.json``) and produces diarized manifests
in ``${exp_dir}/diarized/<split>/<reco>.json`` plus an optional RTTM file,
porting egs/libri_css/asr1/diarization/diarize.sh with
``--diarizer_type spectral``:

1. Sliding-window subsegmentation of each VAD segment
   (``extract_xvectors.sh --window 1.5 --period 0.75 --min-segment 0.5``).
2. Speaker embedding per subsegment with a pretrained ESPnet
   ``Speech2Embedding`` model (replaces the Kaldi 0012_diarization_v1
   x-vector extractor; results will therefore differ from egs1).
3. Per-recording mean-centering + L2 normalization, then cosine similarity
   (``calc_cossim_scores.py``).
4. NME spectral clustering (``spec_clust.py``, see
   ``src.spectral_clustering``).
5. Flat turns from the overlapping labeled subsegments: overlaps cut at the
   midpoint, contiguous same-label turns merged (``make_rttm.py``).
6. Utterance IDs and times quantized exactly as
   ``convert_rttm_to_utt2spk_and_segments.py --append-reco-id-to-spkr=true``:
   ``{label}_{reco}_{start_cs:06d}_{end_cs:06d}`` with truncated centiseconds
   derived from 3-decimal RTTM times. Unlike egs1's ``make_rttm.py`` (whose
   index handling can drop a segment for two-segment recordings), the flat
   turn construction here is a clean sequential reimplementation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

try:  # recipe-dir import (run.py adds the recipe dir to sys.path)
    from src.spectral_clustering import nme_spectral_clustering
except ImportError:  # package-style import fallback
    from .spectral_clustering import nme_spectral_clustering

_EPS = 1e-8


def _libricss_cfg(eval_config) -> Any:
    cfg = OmegaConf.select(eval_config, "libricss")
    if cfg is None:
        raise RuntimeError(
            "eval_config.libricss is required by the diarize stage; "
            "pass --eval_config conf/eval.yaml."
        )
    return cfg


def _resolve_device(device: str) -> str:
    """Resolve ``auto`` to cuda when available, else cpu."""
    if device in ("auto", "", None):
        import torch  # noqa: PLC0415

        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def _load_speaker_encoder(model_tag: str, device: str, dtype: str):
    """Load a pretrained ESPnet speaker embedding model."""
    try:
        from espnet2.bin.spk_inference import Speech2Embedding  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(
            "The `diarize` stage requires the ESPnet speaker toolkit "
            "(espnet2.bin.spk_inference): install espnet with the egs2 "
            "extras, e.g. `pip install espnet[egs2]`."
        ) from e
    logger.info("Loading speaker encoder %s on %s (%s)", model_tag, device, dtype)
    return Speech2Embedding.from_pretrained(
        model_tag=model_tag, device=device, dtype=dtype
    )


def subsegment(
    start: float,
    end: float,
    window: float = 1.5,
    period: float = 0.75,
    min_segment: float = 0.5,
) -> List[Tuple[float, float]]:
    """Sliding-window subsegments of one segment (Kaldi extract_xvectors port).

    Segments shorter than ``min_segment`` are dropped; segments up to
    ``window`` long are kept whole; longer ones emit ``window``-long
    subsegments every ``period``, plus a final ``window``-long subsegment
    ending at ``end`` if the sliding window left a gap.
    """
    dur = end - start
    if dur < min_segment:
        return []
    if dur <= window:
        return [(start, end)]
    subs: List[Tuple[float, float]] = []
    t = start
    while t + window <= end + _EPS:
        subs.append((t, t + window))
        t += period
    tail = end - window
    if subs[-1][0] < tail - _EPS:
        subs.append((tail, end))
    return subs


def embed_subsegments(
    spk_model: Any,
    audio: np.ndarray,
    sample_rate: int,
    subsegs: List[Tuple[float, float]],
    desc: str = "",
) -> np.ndarray:
    """Extract one speaker embedding per subsegment.

    Args:
        spk_model: Pretrained ``Speech2Embedding`` instance.
        audio: Full recording, 1-D float32.
        sample_rate: Sample rate of ``audio``.
        subsegs: (start, end) times in seconds.
        desc: tqdm progress bar description.

    Returns:
        Embedding matrix of shape (len(subsegs), D).
    """
    from tqdm import tqdm  # noqa: PLC0415

    embeddings = []
    for start, end in tqdm(subsegs, desc=desc, leave=False):
        s = max(0, int(round(start * sample_rate)))
        e = min(len(audio), int(round(end * sample_rate)))
        chunk = np.asarray(audio[s:e], dtype=np.float32)
        out = spk_model(chunk)
        if hasattr(out, "detach"):  # torch.Tensor
            out = out.detach().cpu().numpy()
        embeddings.append(np.asarray(out, dtype=np.float32).reshape(-1))
    return np.stack(embeddings)


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Per-recording normalized cosine similarity (calc_cossim_scores port).

    Mean-centers the embeddings across the recording, L2-normalizes them
    (egs1 does not guard against zero norms; this port floors at 1e-10),
    and returns ``1 - squareform(pdist(X, "cosine"))``.
    """
    from scipy.spatial.distance import pdist, squareform  # noqa: PLC0415

    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-10)
    return 1.0 - squareform(pdist(X, "cosine"))


def make_flat_turns(
    turns: List[Tuple[float, float, str]], merge_eps: float = 1e-6
) -> List[Tuple[float, float, str]]:
    """Turn overlapping labeled segments into a flat segmentation.

    Reimplementation of egs1 ``make_rttm.py``: the boundary between two
    overlapping segments with different speakers is placed at the midpoint
    of the overlap; contiguous same-label turns are then merged. Degenerate
    (empty or inverted) turns are dropped.

    Args:
        turns: (start, end, spk) triples sorted by start time.
        merge_eps: Tolerance for treating turns as contiguous when merging.

    Returns:
        Non-overlapping (start, end, spk) triples.
    """
    cut: List[Tuple[float, float, str]] = []
    for start, end, spk in turns:
        if cut and start < cut[-1][1] - merge_eps:
            mid = (start + cut[-1][1]) / 2.0
            prev_start, _prev_end, prev_spk = cut[-1]
            cut[-1] = (prev_start, mid, prev_spk)
            start = mid
        if end <= start:
            continue
        cut.append((start, end, spk))

    merged: List[Tuple[float, float, str]] = []
    for start, end, spk in cut:
        if (
            merged
            and merged[-1][2] == spk
            and abs(start - merged[-1][1]) <= merge_eps
        ):
            merged[-1] = (merged[-1][0], end, spk)
        else:
            merged.append((start, end, spk))
    return merged


def quantize_turn(reco: str, start: float, end: float, spk: str) -> Optional[Dict[str, Any]]:
    """Quantize one flat turn into a manifest segment dict.

    Faithful to egs1's RTTM round trip: ``make_rttm.py`` writes times with
    3 decimals, and ``convert_rttm_to_utt2spk_and_segments.py`` recomputes
    ``end = tbeg + tdur`` and truncates to centiseconds for the utterance ID.
    Returns ``None`` for turns that quantize to zero length.
    """
    rt_start = float(f"{start:.3f}")
    rt_dur = float(f"{end - start:.3f}")
    end_time = rt_start + rt_dur
    st_cs = int(rt_start * 100)
    en_cs = int(end_time * 100)
    if en_cs <= st_cs:
        return None
    return {
        "utt_id": f"{spk}_{reco}_{st_cs:06d}_{en_cs:06d}",
        "start": float(f"{rt_start:.2f}"),
        "end": float(f"{end_time:.2f}"),
        "spk": spk,
        "_rttm": (rt_start, rt_dur),
    }


def diarize_recording(
    manifest: Dict[str, Any],
    spk_model: Any,
    window: float,
    period: float,
    min_segment: float,
    num_clusters: Optional[int],
    max_num_clusters: int,
    pbest: int,
    pmax: int,
    random_state: int,
    desc: str,
) -> List[Dict[str, Any]]:
    """Diarize one recording manifest; return diarized segment dicts."""
    wav = manifest["wav"]
    reco = manifest["reco"]
    sample_rate = int(manifest["sample_rate"])

    subsegs: List[Tuple[float, float]] = []
    for seg in manifest["segments"]:
        subsegs.extend(
            subsegment(
                float(seg["start"]),
                float(seg["end"]),
                window=window,
                period=period,
                min_segment=min_segment,
            )
        )
    if not subsegs:
        logger.warning("%s: no subsegments >= %.2fs; writing empty manifest", reco, min_segment)
        return []

    audio, file_sr = sf.read(wav, dtype="float32")
    if audio.ndim != 1:
        raise ValueError(f"{wav}: expected mono audio, got shape {audio.shape}.")
    if file_sr != sample_rate:
        sample_rate = int(file_sr)

    X = embed_subsegments(spk_model, audio, sample_rate, subsegs, desc=desc)
    A = cosine_similarity_matrix(X)
    labels = nme_spectral_clustering(
        A,
        num_clusters=num_clusters,
        max_num_clusters=max_num_clusters,
        pbest=pbest,
        pmax=pmax,
        random_state=random_state,
    )

    turns = sorted(
        (s, e, str(int(lab) + 1)) for (s, e), lab in zip(subsegs, labels)
    )
    segments = []
    for start, end, spk in make_flat_turns(turns):
        seg = quantize_turn(reco, start, end, spk)
        if seg is not None:
            segments.append(seg)
    logger.info(
        "%s: %d subsegments, %d speakers, %d turns",
        reco,
        len(subsegs),
        len(set(labels.tolist())),
        len(segments),
    )
    return segments


def run_diarization(eval_config) -> None:
    """Execute the ``diarize`` stage.

    Args:
        eval_config: Resolved eval config (``conf/eval.yaml``, carried in
            the framework's ``training_config`` slot) with ``exp_dir`` and
            the ``libricss`` parameter block (``libricss.diarize.*``).
    """
    cfg = _libricss_cfg(eval_config)
    dia_cfg = OmegaConf.select(cfg, "diarize") or OmegaConf.create({})
    spk_model_tag = str(
        dia_cfg.get("spk_model_tag", "espnet/voxcelebs12_xvector_mel")
    )
    device = _resolve_device(str(dia_cfg.get("device", "auto")))
    dtype = str(dia_cfg.get("dtype", "float32"))
    window = float(dia_cfg.get("window_sec", 1.5))
    period = float(dia_cfg.get("period_sec", 0.75))
    min_segment = float(dia_cfg.get("min_segment_sec", 0.5))
    num_clusters = dia_cfg.get("num_clusters", None)
    num_clusters = None if num_clusters is None else int(num_clusters)
    max_num_clusters = int(dia_cfg.get("max_num_clusters", 10))
    pmax = int(dia_cfg.get("pmax", 20))
    pbest = int(dia_cfg.get("pbest", 0))
    random_state = int(dia_cfg.get("random_state", 0))
    write_rttm = bool(dia_cfg.get("write_rttm", True))

    splits = list(cfg.get("splits", ["dev", "eval"]))
    seg_root = Path(str(eval_config.exp_dir)) / "segments"
    out_root = Path(str(eval_config.exp_dir)) / "diarized"
    if not seg_root.is_dir():
        raise FileNotFoundError(
            f"{seg_root} not found; run the `segment` stage first."
        )

    spk_model = _load_speaker_encoder(spk_model_tag, device, dtype)

    for split in splits:
        split_dir = seg_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"{split_dir} not found; run the `segment` stage for split "
                f"'{split}' first."
            )
        out_split = out_root / split
        out_split.mkdir(parents=True, exist_ok=True)
        rttm_lines: List[str] = []
        manifests = sorted(split_dir.glob("*.json"))
        if not manifests:
            logger.warning("No segment manifests in %s; skipping.", split_dir)
        for i, manifest_path in enumerate(manifests, 1):
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            reco = manifest["reco"]
            segments = diarize_recording(
                manifest,
                spk_model,
                window=window,
                period=period,
                min_segment=min_segment,
                num_clusters=num_clusters,
                max_num_clusters=max_num_clusters,
                pbest=pbest,
                pmax=pmax,
                random_state=random_state,
                desc=f"{split} {i}/{len(manifests)} {reco}",
            )
            for seg in segments:
                if write_rttm:
                    rt_start, rt_dur = seg.pop("_rttm")
                    rttm_lines.append(
                        f"SPEAKER {reco} 1 {rt_start:7.3f} {rt_dur:7.3f} "
                        f"<NA> <NA> {seg['spk']} <NA> <NA>"
                    )
                else:
                    seg.pop("_rttm", None)
            out_manifest = {
                "reco": reco,
                "wav": manifest["wav"],
                "sample_rate": int(manifest["sample_rate"]),
                "segments": segments,
            }
            with (out_split / f"{reco}.json").open("w", encoding="utf-8") as f:
                json.dump(out_manifest, f, indent=1)
        if write_rttm:
            with (out_split / "rttm").open("w", encoding="utf-8") as f:
                f.write("\n".join(rttm_lines) + ("\n" if rttm_lines else ""))
            logger.info("Wrote %d RTTM lines to %s", len(rttm_lines), out_split / "rttm")
        logger.info("Split %s: wrote %d diarized manifests to %s", split, len(manifests), out_split)
