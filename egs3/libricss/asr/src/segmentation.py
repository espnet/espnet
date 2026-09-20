"""Segmentation stage for the LibriCSS recipe.

Writes per-recording segment manifests consumed by the ``diarize`` stage and
by ``LibriCSSDataset``:

.. code-block:: text

    ${exp_dir}/segments/<split>/<reco>.json

Two modes (``libricss.segment.mode`` in conf/eval.yaml):

- ``vad``: webrtcvad speech activity detection, ported from
  egs/libri_css/asr1/local/segmentation/apply_webrtcvad.py (mode 0, 30 ms
  frames, 300 ms padding ring buffer, 90% trigger/detrigger). Segments carry
  no speaker labels; run the ``diarize`` stage next.
- ``oracle``: corpus-provided segments with speaker labels and transcripts
  from ``data/<split>/oracle/{segments,utt2spk,text}``, ported from
  egs/libri_css/asr1/local/segment_diarize_oracle.sh.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import soundfile as sf
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

_VALID_MODES = ("vad", "oracle")
_VALID_FRAME_MS = (10, 20, 30)
_VALID_SAMPLE_RATES = (8000, 16000, 32000, 48000)


def _zfill6(seconds: float) -> str:
    """Format seconds as zero-padded centiseconds (egs1 convention)."""
    return "{:.0f}".format(100 * seconds).zfill(6)


def _libricss_cfg(eval_config) -> Any:
    cfg = OmegaConf.select(eval_config, "libricss")
    if cfg is None:
        raise RuntimeError(
            "eval_config.libricss is required by the segment stage; "
            "pass --eval_config conf/eval.yaml."
        )
    return cfg


def _import_webrtcvad():
    """Import webrtcvad with an actionable error message."""
    try:
        import webrtcvad  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(
            "The `segment` stage with mode=vad requires webrtcvad: "
            "`pip install webrtcvad` (or the prebuilt `webrtcvad-wheels`). "
            "Alternatively use libricss.segment.mode: oracle."
        ) from e
    return webrtcvad


class _Frame:
    """A single audio frame for the VAD (port of egs1's Frame class)."""

    __slots__ = ("bytes", "timestamp", "duration")

    def __init__(self, data: bytes, timestamp: float, duration: float):
        self.bytes = data
        self.timestamp = timestamp
        self.duration = duration


def _read_wave_pcm(path: str | Path) -> tuple[bytes, int]:
    """Read a wav file as mono 16-bit PCM bytes (egs1 read_wave port).

    Raises:
        ValueError: If the file is not mono or not at a webrtcvad rate.
    """
    audio, sample_rate = sf.read(str(path), dtype="int16")
    if audio.ndim != 1:
        raise ValueError(
            f"{path}: expected mono audio (the builder extracts one mic "
            f"channel); got shape {audio.shape}."
        )
    if sample_rate not in _VALID_SAMPLE_RATES:
        raise ValueError(
            f"{path}: webrtcvad requires 8/16/32/48 kHz; got {sample_rate}."
        )
    return audio.tobytes(), int(sample_rate)


def _frame_generator(frame_duration_ms: int, audio: bytes, sample_rate: int):
    """Yield fixed-duration frames of PCM audio (egs1 port)."""
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield _Frame(audio[offset : offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def _vad_segments(
    sample_rate: int,
    frame_duration_ms: int,
    padding_duration_ms: int,
    vad: Any,
    frames: List[_Frame],
) -> List[tuple[float, float]]:
    """Collect voiced regions with a padded ring buffer (egs1 verbatim port).

    Triggers when more than 90% of the frames in the ring buffer are voiced
    and detriggers when more than 90% are unvoiced. Leftover voiced frames
    at the end of the recording close the final segment.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer: deque = deque(maxlen=num_padding_frames)
    triggered = False
    segments: List[tuple[float, float]] = []
    voiced_frames: List[_Frame] = []
    start_time = 0.0
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                for f, _s in ring_buffer:
                    voiced_frames.append(f)
                start_time = voiced_frames[0].timestamp
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                end_time = frame.timestamp + frame.duration
                triggered = False
                ring_buffer.clear()
                voiced_frames = []
                segments.append((start_time, end_time))
    if voiced_frames:
        # egs1 uses the final frame's start timestamp here (not + duration).
        end_time = voiced_frames[-1].timestamp
        segments.append((start_time, end_time))
    return segments


def vad_file_segments(
    wav: str | Path,
    vad: Any,
    reco: str,
    frame_ms: int = 30,
    padding_ms: int = 300,
) -> List[Dict[str, Any]]:
    """Run webrtcvad over one recording and return manifest segment dicts."""
    pcm, sample_rate = _read_wave_pcm(wav)
    frames = list(_frame_generator(frame_ms, pcm, sample_rate))
    raw_segments = _vad_segments(sample_rate, frame_ms, padding_ms, vad, frames)
    segments = []
    for start, end in raw_segments:
        start = float("{:.2f}".format(start))
        end = float("{:.2f}".format(end))
        if end <= start:
            continue
        segments.append(
            {
                "utt_id": f"{reco}_{_zfill6(start)}_{_zfill6(end)}",
                "start": start,
                "end": end,
            }
        )
    return segments


def _read_scp(path: Path) -> Dict[str, str]:
    """Read a two-column scp file into a dict."""
    entries = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(maxsplit=1)
            entries[key] = value
    return entries


def _load_oracle_segments(oracle_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Group oracle segments per recording with speaker labels and text."""
    utt2spk = _read_scp(oracle_dir / "utt2spk")
    utt2text: Dict[str, str] = {}
    with (oracle_dir / "text").open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if parts:
                utt2text[parts[0]] = parts[1] if len(parts) > 1 else ""

    per_reco: Dict[str, List[Dict[str, Any]]] = {}
    with (oracle_dir / "segments").open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            utt, reco, start, end = parts[0], parts[1], float(parts[2]), float(parts[3])
            per_reco.setdefault(reco, []).append(
                {
                    "utt_id": utt,
                    "start": start,
                    "end": end,
                    "spk": utt2spk[utt],
                    "text": utt2text.get(utt, ""),
                }
            )
    for segments in per_reco.values():
        segments.sort(key=lambda s: (s["start"], s["end"]))
    return per_reco


def run_segmentation(eval_config) -> None:
    """Execute the ``segment`` stage.

    Args:
        eval_config: Resolved eval config (``conf/eval.yaml``, carried in
            the framework's ``training_config`` slot) with ``data_dir``,
            ``exp_dir`` and the ``libricss`` parameter block.
    """
    cfg = _libricss_cfg(eval_config)
    seg_cfg = OmegaConf.select(cfg, "segment") or OmegaConf.create({})
    mode = str(seg_cfg.get("mode", "vad")).lower()
    if mode not in _VALID_MODES:
        raise ValueError(
            f"libricss.segment.mode must be one of {_VALID_MODES}, got {mode!r}."
        )
    vad_cfg = OmegaConf.select(seg_cfg, "vad") or OmegaConf.create({})
    vad_mode = int(vad_cfg.get("mode", 0))
    frame_ms = int(vad_cfg.get("frame_ms", 30))
    padding_ms = int(vad_cfg.get("padding_ms", 300))
    if vad_mode not in (0, 1, 2, 3):
        raise ValueError(f"webrtcvad mode must be in 0..3, got {vad_mode}.")
    if frame_ms not in _VALID_FRAME_MS:
        raise ValueError(f"webrtcvad frame_ms must be in {_VALID_FRAME_MS}.")

    splits = list(cfg.get("splits", ["dev", "eval"]))
    data_dir = Path(str(eval_config.data_dir))
    out_root = Path(str(eval_config.exp_dir)) / "segments"

    vad = None
    if mode == "vad":
        webrtcvad = _import_webrtcvad()
        vad = webrtcvad.Vad(vad_mode)
        logger.info(
            "webrtcvad segmentation: mode=%d frame=%dms padding=%dms",
            vad_mode,
            frame_ms,
            padding_ms,
        )

    for split in splits:
        split_dir = data_dir / split
        wav_scp_path = split_dir / "wav.scp"
        if not wav_scp_path.is_file():
            raise FileNotFoundError(
                f"{wav_scp_path} not found; run the create_dataset stage first."
            )
        wav_scp = _read_scp(wav_scp_path)

        oracle: Dict[str, List[Dict[str, Any]]] = {}
        if mode == "oracle":
            oracle_dir = split_dir / "oracle"
            if not oracle_dir.is_dir():
                raise FileNotFoundError(
                    f"{oracle_dir} not found; run the create_dataset stage first."
                )
            oracle = _load_oracle_segments(oracle_dir)

        out_split = out_root / split
        out_split.mkdir(parents=True, exist_ok=True)
        total_segments = 0
        for i, (reco, wav) in enumerate(sorted(wav_scp.items()), 1):
            if mode == "oracle":
                segments = [dict(s) for s in oracle.get(reco, [])]
            else:
                segments = vad_file_segments(
                    wav, vad, reco, frame_ms=frame_ms, padding_ms=padding_ms
                )
            total_segments += len(segments)
            manifest = {
                "reco": reco,
                "wav": str(Path(wav).resolve()),
                "sample_rate": int(sf.info(wav).samplerate),
                "segments": segments,
            }
            with (out_split / f"{reco}.json").open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=1)
            logger.info(
                "[%s %d/%d] %s: %d segments (%s)",
                split,
                i,
                len(wav_scp),
                reco,
                len(segments),
                mode,
            )
        logger.info(
            "Split %s: wrote %d manifests (%d segments) to %s",
            split,
            len(wav_scp),
            total_segments,
            out_split,
        )
