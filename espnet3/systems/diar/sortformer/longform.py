"""Long-form (full-session) Sortformer diarization and scoring helpers.

The offline model has a bounded context (trained on <=90 s sessions and O(T^2)
attention), so a full ~50-min meeting cannot be diarized in a single pass. Two
strategies are provided to cover the whole session and produce globally
consistent speaker identities:

* ``"stitch"`` (``diarize_long``): run the offline model on overlapping chunks
  and stitch the per-chunk speaker channels into global identities using the
  predictions on the overlap region (Hungarian matching).
* ``"streaming"``: run the streaming model once over the whole session using its
  speaker cache, which keeps identities consistent without chunk stitching.

Either way the result is a single session-level activity matrix that is
binarized into segments and written to one RTTM per session, then scored with a
collar (the standard long-form DER protocol), unlike the per-window chunk DER.

The recipe's infer/measure stages call ``run_longform_inference`` (writes
``out_dir/{hyp,ref}/*.rttm``) and ``score_rttm_der`` (returns a DER dict).
``build_eval_model`` loads a model from one of several checkpoint sources.
"""

import math
from typing import List, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def _median_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Per-column temporal median filter (edge-padded). No-op when ``k <= 1``."""
    if k <= 1:
        return x
    # scipy's C implementation is far faster than a Python loop over the time
    # axis on long full-session sequences; ``mode="nearest"`` = edge padding.
    from scipy.ndimage import median_filter

    return median_filter(x, size=(k, 1), mode="nearest")


@torch.no_grad()
def diarize_long(
    model,
    wav: np.ndarray,
    sample_rate: int = 16000,
    chunk_sec: float = 90.0,
    overlap_sec: float = 30.0,
    frame_dur: float = 0.08,
    device: str = "cpu",
) -> np.ndarray:
    """Diarize a full session with the offline model via chunking and stitching.

    The waveform is split into overlapping ``chunk_sec`` windows. Each chunk is
    diarized independently, then its speaker columns are matched to the running
    global identities using the overlap region (Hungarian assignment on absolute
    activity difference). Overlapping predictions are averaged.

    Args:
        model: Sortformer model exposing ``diarize`` and ``num_spk``.
        wav: Mono waveform of shape ``(num_samples,)``.
        sample_rate: Sample rate of ``wav`` in Hz.
        chunk_sec: Chunk length in seconds.
        overlap_sec: Overlap between consecutive chunks in seconds.
        frame_dur: Model output frame duration in seconds (0.08 = 12.5 fps).
        device: Torch device for inference.

    Returns:
        Session activity probabilities of shape ``(T_frames, num_spk)``.
    """
    model.eval()
    num_spk = model.num_spk
    n = wav.shape[0]
    fps = 1.0 / frame_dur  # model output frame rate (12.5 fps @ 80 ms)
    total_frames = int(math.ceil(n / sample_rate * fps))

    acc = np.zeros((total_frames, num_spk), dtype=np.float64)
    cnt = np.zeros((total_frames, 1), dtype=np.float64)

    chunk_samp = int(chunk_sec * sample_rate)
    hop_samp = int((chunk_sec - overlap_sec) * sample_rate)
    starts = list(range(0, max(1, n), hop_samp))

    for ci, s in enumerate(starts):
        e = min(n, s + chunk_samp)
        if e - s < int(0.5 * sample_rate):  # skip <0.5s tail
            continue
        chunk = torch.tensor(wav[s:e], dtype=torch.float32, device=device).unsqueeze(0)
        ln = torch.tensor([chunk.shape[1]], device=device)
        preds, plen = model.diarize(chunk, ln)
        pc = preds[0, : plen[0]].float().cpu().numpy()  # (Tc, num_spk)
        g0 = int(round(s / sample_rate * fps))
        g1 = min(total_frames, g0 + pc.shape[0])
        pc = pc[: g1 - g0]

        if ci > 0:
            # Stitch: match this chunk's columns to the running global identities
            # using the already-written overlap region.
            ov_mask = cnt[g0:g1, 0] > 0
            if ov_mask.sum() > 3:
                gprev = (acc[g0:g1] / np.maximum(cnt[g0:g1], 1e-8))[ov_mask]
                cur = pc[ov_mask]
                cost = np.zeros((num_spk, num_spk))
                for i in range(num_spk):
                    for j in range(num_spk):
                        cost[i, j] = np.sum(np.abs(gprev[:, i] - cur[:, j]))
                ri, cj = linear_sum_assignment(cost)
                perm = np.zeros(num_spk, dtype=int)
                for i, j in zip(ri, cj):
                    perm[i] = j
                pc = pc[:, perm]

        acc[g0:g1] += pc
        cnt[g0:g1] += 1.0

    return (acc / np.maximum(cnt, 1e-8)).astype(np.float32)


def activity_to_segments(
    preds: np.ndarray,
    frame_dur: float = 0.08,
    threshold: float = 0.5,
    median_k: int = 11,
    min_dur: float = 0.1,
    fill_gap: float = 0.2,
) -> List[Tuple[int, float, float]]:
    """Convert per-speaker activity probabilities into timed segments.

    Smooths each speaker track with a temporal median filter, thresholds it,
    closes short gaps, and drops too-short runs.

    Args:
        preds: Activity probabilities of shape ``(T_frames, num_spk)``.
        frame_dur: Frame duration in seconds (maps frame indices to time).
        threshold: Activity threshold for binarization.
        median_k: Median-filter window in frames.
        min_dur: Minimum segment duration in seconds; shorter runs are dropped.
        fill_gap: Gaps up to this many seconds between runs are merged.

    Returns:
        List of ``(speaker_index, start_s, end_s)`` tuples.
    """
    sm = _median_filter(preds, median_k)
    biny = sm >= threshold
    segs = []
    fill = int(round(fill_gap / frame_dur))
    min_f = max(1, int(round(min_dur / frame_dur)))
    for spk in range(preds.shape[1]):
        active = biny[:, spk].astype(int)
        # close short gaps
        i = 0
        T = len(active)
        runs = []
        while i < T:
            if active[i]:
                j = i
                while j < T and active[j]:
                    j += 1
                runs.append([i, j])
                i = j
            else:
                i += 1
        merged = []
        for r in runs:
            if merged and r[0] - merged[-1][1] <= fill:
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        for st, en in merged:
            if en - st >= min_f:
                segs.append((spk, st * frame_dur, en * frame_dur))
    return segs


def write_rttm(path, session_id, segments, label_prefix="spk"):
    """Write hypothesis segments to an RTTM file.

    Args:
        path: Output RTTM path.
        session_id: Recording / session id written in column 2.
        segments: Iterable of ``(speaker_index, start_s, end_s)`` tuples.
        label_prefix: Prefix for the speaker label (e.g. ``"spk0"``).
    """
    with open(path, "w", encoding="utf-8") as f:
        for spk, st, en in sorted(segments, key=lambda x: (x[1], x[0])):
            f.write(
                f"SPEAKER {session_id} 1 {st:.3f} {en - st:.3f} <NA> <NA> "
                f"{label_prefix}{spk} <NA> <NA>\n"
            )


def supervisions_to_rttm(path, session_id, supervisions):
    """Write lhotse reference supervisions to an RTTM file.

    Args:
        path: Output RTTM path.
        session_id: Recording / session id written in column 2.
        supervisions: Iterable of lhotse supervision segments (each with
            ``start``, ``duration`` and ``speaker``).
    """
    with open(path, "w", encoding="utf-8") as f:
        for s in sorted(supervisions, key=lambda s: s.start):
            f.write(
                f"SPEAKER {session_id} 1 {s.start:.3f} {s.duration:.3f} <NA> <NA> "
                f"{s.speaker} <NA> <NA>\n"
            )


def build_eval_model(
    nemo_ckpt=None,
    ckpt=None,
    hf_model="nvidia/diar_sortformer_4spk-v1",
    nest=None,
    num_spk: int = 4,
    device: str = "cpu",
):
    """Build a Sortformer model for long-form eval from one of several sources.

    The source is chosen by priority: ``nemo_ckpt`` > ``ckpt`` > ``hf_model``.
    An optional NEST encoder can be overlaid afterwards. The model is returned on
    ``device`` in eval mode.

    Args:
        nemo_ckpt: Path to a NeMo streaming-v2 full state-dict (highest priority).
        ckpt: Path to an ESPnet ``.pth``/``.ckpt`` for the offline architecture;
            a leading ``model.`` prefix on keys is stripped and loaded
            non-strictly.
        hf_model: HuggingFace id of NVIDIA offline weights to convert (used when
            neither checkpoint is given).
        nest: Optional NEST checkpoint whose encoder weights overlay the model.
        num_spk: Number of speaker channels.
        device: Torch device to move the model to.

    Returns:
        The constructed Sortformer model in eval mode on ``device``.
    """
    if nemo_ckpt is not None:
        from .convert_nemo_sortformer import convert_nemo

        model, _ = convert_nemo(nemo_ckpt, num_spk=num_spk)
    elif ckpt is not None:
        from .model import build_sortformer_model

        model = build_sortformer_model(num_spk=num_spk)
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        sd = {
            k[len("model.") :] if k.startswith("model.") else k: v
            for k, v in sd.items()
        }
        model.load_state_dict(sd, strict=False)
    else:
        from .convert_hf_sortformer import convert

        model, _ = convert(hf_model, num_spk=num_spk)
    if nest is not None:
        from .convert_nest import load_nest_encoder

        load_nest_encoder(model, nest)
    return model.to(device).eval()


@torch.no_grad()
def run_longform_inference(
    model,
    cuts,
    out_dir,
    mode: str = "streaming",
    device: str = "cpu",
    chunk_sec: float = 90.0,
    overlap_sec: float = 30.0,
    threshold: float = 0.5,
    channel: int = 0,
    log=print,
):
    """Diarize a full-session CutSet and write hypothesis and reference RTTMs.

    For each cut, the chosen channel is diarized end-to-end, binarized into
    segments, and written to ``out_dir/hyp/{recording_id}.rttm``; the cut's
    reference supervisions are written to ``out_dir/ref/{recording_id}.rttm``.
    The two directories can then be scored by ``score_rttm_der``.

    Args:
        model: Sortformer model. Used via ``diarize_streaming`` when
            ``mode="streaming"``, else via ``diarize`` inside ``diarize_long``.
        cuts: lhotse ``CutSet`` of full-session cuts.
        out_dir: Output directory; ``hyp/`` and ``ref/`` are created inside it.
        mode: ``"streaming"`` for a single speaker-cache pass (globally
            consistent speakers), or ``"stitch"`` for overlap-Hungarian chunk
            stitching via ``diarize_long``.
        device: Torch device for inference.
        chunk_sec: Chunk length in seconds (``"stitch"`` mode only).
        overlap_sec: Chunk overlap in seconds (``"stitch"`` mode only).
        threshold: Activity threshold for segment binarization.
        channel: Audio channel index to load from each cut.
        log: Callable for progress logging, or falsy to silence it.

    Returns:
        ``out_dir``, now containing ``hyp/`` and ``ref/`` RTTM files.

    Example:
        >>> run_longform_inference(model, cuts, "exp/der", mode="streaming")
        >>> score_rttm_der("exp/der", collar=0.25)["DER"]
    """
    import os

    os.makedirs(os.path.join(out_dir, "hyp"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "ref"), exist_ok=True)
    ids = list(cuts.ids)
    for k, cid in enumerate(ids):
        cut = cuts[cid]
        wav = cut.load_audio()[channel].astype(np.float32)
        if mode == "streaming":
            p, plen = model.diarize_streaming(
                torch.tensor(wav, device=device).unsqueeze(0)
            )
            preds = p[0, : plen[0]].float().cpu().numpy()
        else:
            preds = diarize_long(
                model,
                wav,
                sample_rate=cut.sampling_rate,
                chunk_sec=chunk_sec,
                overlap_sec=overlap_sec,
                device=device,
            )
        segs = activity_to_segments(preds, threshold=threshold)
        write_rttm(
            os.path.join(out_dir, "hyp", f"{cut.recording_id}.rttm"),
            cut.recording_id,
            segs,
        )
        supervisions_to_rttm(
            os.path.join(out_dir, "ref", f"{cut.recording_id}.rttm"),
            cut.recording_id,
            cut.supervisions,
        )
        if log:
            log(
                f"[{k + 1}/{len(ids)}] {cut.recording_id} "
                f"dur={cut.duration / 60:.1f}min frames={preds.shape[0]} segs={len(segs)}"
            )
    return out_dir


def _load_rttm(path):
    """Parse an RTTM file into a list of ``(speaker, start, end)`` tuples."""
    segs = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 8 and p[0] == "SPEAKER":
                start, dur, spk = float(p[3]), float(p[4]), p[7]
                segs.append((spk, start, start + dur))
    return segs


def _rttm_to_frames(segs, total, frame_dur):
    """Rasterize ``(spk, start, end)`` segments into a ``(total, n_spk)`` mask."""
    spks = sorted({s for s, _, _ in segs})
    idx = {s: i for i, s in enumerate(spks)}
    m = np.zeros((total, max(1, len(spks))), dtype=np.float32)
    for s, st, en in segs:
        a, b = int(round(st / frame_dur)), int(round(en / frame_dur))
        m[max(0, a) : min(total, b), idx[s]] = 1.0
    return m


def score_rttm_der(
    out_dir, collar: float = 0.25, frame_dur: float = 0.01, use_pyannote: bool = True
):
    """Score session-level diarization error rate over the hyp/ref RTTM dirs.

    Pairs each ``out_dir/ref/*.rttm`` with the matching ``out_dir/hyp/*.rttm``.
    When pyannote is available (and ``use_pyannote``), it computes a collar-based
    DER and its breakdown; otherwise it falls back to a frame-level
    Hungarian-matched DER with no collar.

    Args:
        out_dir: Directory containing ``hyp/`` and ``ref/`` RTTM files.
        collar: Scoring collar in seconds (pyannote path only).
        frame_dur: Frame duration in seconds for the fallback frame-level DER.
        use_pyannote: If False, force the frame-level fallback scorer.

    Returns:
        Dict with the DER and metadata. With pyannote::

            {"DER", "MS", "FA", "SC", "collar", "scorer": "pyannote"}

        all percentages; ``MS``/``FA``/``SC`` are missed-detection, false-alarm
        and speaker-confusion rates. Fallback path returns
        ``{"DER", "collar": 0.0, "scorer": "frame-level ..."}``.
    """
    import glob
    import os

    ref_files = sorted(glob.glob(os.path.join(out_dir, "ref", "*.rttm")))
    try:
        if not use_pyannote:
            raise ImportError
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate

        metric = DiarizationErrorRate(collar=collar, skip_overlap=False)

        def ann(path):
            a = Annotation(uri=os.path.basename(path))
            for spk, st, en in _load_rttm(path):
                a[Segment(st, en)] = spk
            return a

        for rf in ref_files:
            hf = os.path.join(out_dir, "hyp", os.path.basename(rf))
            if os.path.exists(hf):
                metric(ann(rf), ann(hf))
        c = metric[:]
        tot = c["total"]
        return {
            "DER": round(100 * abs(metric), 2),
            "MS": round(100 * c["missed detection"] / tot, 2),
            "FA": round(100 * c["false alarm"] / tot, 2),
            "SC": round(100 * c["confusion"] / tot, 2),
            "collar": collar,
            "scorer": "pyannote",
        }
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        tot_ref = tot_err = 0.0
        for rf in ref_files:
            hf = os.path.join(out_dir, "hyp", os.path.basename(rf))
            if not os.path.exists(hf):
                continue
            ref_segs, hyp_segs = _load_rttm(rf), _load_rttm(hf)
            end = max([e for _, _, e in ref_segs + hyp_segs] + [0.0])
            total = int(np.ceil(end / frame_dur))
            ref = _rttm_to_frames(ref_segs, total, frame_dur)
            hyp = _rttm_to_frames(hyp_segs, total, frame_dur)
            s = max(ref.shape[1], hyp.shape[1])
            ref = np.pad(ref, ((0, 0), (0, s - ref.shape[1])))
            hyp = np.pad(hyp, ((0, 0), (0, s - hyp.shape[1])))
            cost = np.array(
                [
                    [np.sum(np.abs(ref[:, i] - hyp[:, j])) for j in range(s)]
                    for i in range(s)
                ]
            )
            ri, cj = linear_sum_assignment(cost)
            hp = np.zeros_like(ref)
            for i, j in zip(ri, cj):
                hp[:, i] = hyp[:, j]
            tot_ref += float(ref.sum())
            tot_err += float(
                np.sum((ref == 1) & (hp == 0)) + np.sum((ref == 0) & (hp == 1))
            )
        denom = max(tot_ref, 1.0)
        return {
            "DER": round(100 * tot_err / denom, 2),
            "collar": 0.0,
            "scorer": "frame-level (no collar; install pyannote.metrics for collar DER)",
        }
