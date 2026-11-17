#!/usr/bin/env python3
"""Real-time simulation of a VAD: audio arrives in blocks, decisions are final at block end.

    python3 local/v2/realtime_sim.py --system <name> --block_ms 20 --out <dir> \
        --sets dump/raw/<set> ... [--task 1 --ntasks 1] [--window_s 2] [--limit_hours H]

A system gets a recording's samples in time order, b samples at a time. After block k it
may return the probabilities of frames whose interval ends by the block's end, using only
the samples it has been given; those decisions are final. simulate() feeds the blocks,
dates every returned frame with its emission time (the block end) and checks that no frame
is dated after the audio the system had: a frame interval [offset + (i - 0.5) shift,
offset + (i + 0.5) shift) must end by the emission time (CausalityError otherwise). The
latency of a frame is its emission time minus the end of its interval, between 0 and b.

Two kinds of systems:
- streaming (StreamingSystem): keeps its own state, returns the frames that the new samples
  complete; ours at look-ahead 0 (the TCN encoder's forward_streaming, frames dated at their
  window end), Silero (its stateful 512-sample call), WebRTC (480-sample frames);
- windowed (WindowedOffline): an offline model rerun at each block end on the last W seconds
  (start aligned to the model's hop), keeping only the frames that end in the new block;
  MarbleNet here. Its probabilities are computed for all block ends of a recording in
  batches (windowed_probs), which gives the same result as feeding the blocks one by one
  (tested), since each window holds only audio up to its block end.

Writes <out>/<system>/diar_eval/diarized_<set>/predictions/<rec>.npy and diarize.<task>.scp
(the layout of local/v2/vad_infer.py), <out>/<system>/frames.json (shift and offset, for
local/v2/score_vad.py --shift --offset; "latency_ms": the block) and latency.<task>.tsv
(per recording: frames, largest latency in ms, seconds of audio, seconds of compute).
"""

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

FS = 16000
ENTRY = re.compile(r"^(.+):(\d+):(\d+)$")


class CausalityError(AssertionError):
    """A system dated a decision after the audio it had been given."""


class StreamingSystem:
    """Interface: shift and offset (seconds) date frame i as [offset + (i - 0.5) shift,
    offset + (i + 0.5) shift); reset() starts a recording; accept(samples) returns the
    probabilities of the frames completed since the last call, in order."""

    shift: float
    offset: float

    def reset(self):
        raise NotImplementedError

    def accept(self, samples: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def frame_end(i, shift, offset):
    return offset + (np.asarray(i) + 0.5) * shift


def simulate(system: StreamingSystem, wav: np.ndarray, block: int):
    """Feed wav in blocks of `block` samples; return (probs, emission times in s)."""
    system.reset()
    probs, emit = [], []
    for start in range(0, len(wav), block):
        end = min(start + block, len(wav))
        out = np.asarray(system.accept(wav[start:end]), dtype=np.float32).reshape(-1)
        t = end / FS
        first = len(probs)
        last_end = frame_end(first + len(out) - 1, system.shift, system.offset)
        if len(out) and last_end > t + 0.5 / FS:
            raise CausalityError(
                f"frame {first + len(out) - 1} ends at {last_end:.4f} s but was returned "
                f"at {t:.4f} s, when the system had audio up to {t:.4f} s only"
            )
        probs.extend(out.tolist())
        emit.extend([t] * len(out))
    return np.asarray(probs, dtype=np.float32), np.asarray(emit)


def latencies(emit, shift, offset):
    return emit - frame_end(np.arange(len(emit)), shift, offset)


class WindowedOffline(StreamingSystem):
    """An offline model rerun at each block end on the last `window` samples.

    f(batch of equal-length windows (B, T)) -> list of frame probabilities per window;
    frame j of a window starting at sample s is dated with the model's own shift and
    offset relative to s. The window start is rounded down to a multiple of `hop`
    samples (the model's frame step), so window frames fall on the recording's grid.
    """

    def __init__(self, f, hop: int, shift: float, offset: float, window: int):
        self.f, self.hop, self.shift, self.offset, self.window = (
            f,
            hop,
            shift,
            offset,
            window,
        )

    def reset(self):
        self.audio = np.zeros(0, dtype=np.float32)
        self.emitted = 0

    def accept(self, samples):
        self.audio = np.concatenate([self.audio, np.asarray(samples, dtype=np.float32)])
        t = len(self.audio)
        s = max(0, (t - self.window) // self.hop * self.hop)
        p = np.asarray(self.f(self.audio[s:t][None])[0], dtype=np.float32).reshape(-1)
        return self._new(p, s, t)

    def _new(self, p, s, t):
        g0 = s // self.hop
        if g0 > self.emitted:
            raise ValueError(
                f"the window starts at frame {g0}, after frame {self.emitted}, the first "
                "one not yet returned: the window must be longer than the block"
            )
        ends = frame_end(g0 + np.arange(len(p)), self.shift, self.offset)
        last = int(np.searchsorted(ends, t / FS + 0.5 / FS, side="right")) - 1 + g0
        out = p[self.emitted - g0 : last - g0 + 1] if last >= self.emitted else p[:0]
        self.emitted = max(self.emitted, last + 1)
        return out


def windowed_probs(sys_: WindowedOffline, wav: np.ndarray, block: int, batch: int = 64):
    """simulate(sys_, wav, block) computed with the windows of equal length in batches."""
    n = len(wav)
    ends = list(range(block, n, block)) + [n]
    starts = [max(0, (t - sys_.window) // sys_.hop * sys_.hop) for t in ends]
    outs = [None] * len(ends)
    full = [k for k, (s, t) in enumerate(zip(starts, ends)) if t - s == sys_.window]
    for i in range(0, len(full), batch):
        ks = full[i : i + batch]
        x = np.stack([wav[starts[k] : ends[k]] for k in ks])
        for k, p in zip(ks, sys_.f(x)):
            outs[k] = p
    for k in range(len(ends)):
        if outs[k] is None:
            outs[k] = sys_.f(wav[starts[k] : ends[k]][None])[0]
    sys_.reset()
    probs, emit = [], []
    for k, t in enumerate(ends):
        out = sys_._new(np.asarray(outs[k], dtype=np.float32).reshape(-1), starts[k], t)
        probs.extend(out.tolist())
        emit.extend([t / FS] * len(out))
    return np.asarray(probs, dtype=np.float32), np.asarray(emit)


# --------------------------------------------------------------------------- systems
class OursStream(StreamingSystem):
    """Our model at look-ahead 0 through the TCN encoder's forward_streaming.

    Frame i of the STFT (512-sample window every 128 samples, not centred) reads samples
    [128 i, 128 i + 512) and is dated by the 8 ms interval ending at its last sample
    (local/v2/score_vad.py's default). Only the new frames are computed: frontend, global
    MVN and the linear decoder work per frame, the encoder keeps its convolution state.
    """

    shift, offset = 128 / FS, 448 / FS

    def __init__(self, exp, ckpt, device=None):
        import torch

        from espnet2.bin.diar_inference import DiarizeSpeech

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.diar = DiarizeSpeech(
            train_config=f"{exp}/config.yaml",
            model_file=f"{exp}/{ckpt}",
            num_spk=1,
            device=device,
            look_ahead=0,
        )
        self.model = self.diar.diar_model
        stft = self.model.frontend.stft
        assert (
            not stft.center
        ), "streaming needs an STFT that does not centre its frames"
        self.n_fft, self.hop = stft.n_fft, stft.hop_length
        assert (self.n_fft, self.hop) == (512, 128), (self.n_fft, self.hop)
        assert hasattr(self.model.encoder, "forward_streaming")
        self.device = device

    def reset(self):
        self.buffer = np.zeros(0, dtype=np.float32)
        self.state = None

    def accept(self, samples):
        import torch

        self.buffer = np.concatenate(
            [self.buffer, np.asarray(samples, dtype=np.float32)]
        )
        if len(self.buffer) < self.n_fft:
            return np.zeros(0, dtype=np.float32)
        new = (len(self.buffer) - self.n_fft) // self.hop + 1
        end = (new - 1) * self.hop + self.n_fft
        x = torch.from_numpy(self.buffer[:end]).to(self.device)[None]
        m = self.model
        with torch.no_grad():
            lengths = torch.tensor([x.size(1)], device=self.device)
            feats, flens = m._extract_feats(x, lengths)
            if m.normalize is not None:
                feats, flens = m.normalize(feats, flens)
            out, self.state = m.encoder.forward_streaming(feats, self.state)
            logits = m.decoder(out, flens)
        self.buffer = self.buffer[new * self.hop :]
        return torch.sigmoid(logits)[0, :, 0].float().cpu().numpy()


class SileroStream(StreamingSystem):
    """Silero VAD through its stateful call on 512-sample chunks (32 ms frames)."""

    shift, offset = 512 / FS, 256 / FS

    def __init__(self):
        import torch
        from silero_vad import load_silero_vad

        torch.set_num_threads(1)
        self.model = load_silero_vad()

    def reset(self):
        self.model.reset_states()
        self.buffer = np.zeros(0, dtype=np.float32)

    def accept(self, samples):
        import torch

        self.buffer = np.concatenate(
            [self.buffer, np.asarray(samples, dtype=np.float32)]
        )
        out = []
        while len(self.buffer) >= 512:
            out.append(float(self.model(torch.from_numpy(self.buffer[:512]), FS)))
            self.buffer = self.buffer[512:]
        return np.asarray(out, dtype=np.float32)


class WebRTCStream(StreamingSystem):
    """WebRTC VAD (aggressiveness 0) on 480-sample frames, a new detector per recording."""

    shift, offset = 480 / FS, 240 / FS

    def __init__(self, mode=0):
        self.mode = mode

    def reset(self):
        import webrtcvad

        self.vad = webrtcvad.Vad(self.mode)
        self.buffer = np.zeros(0, dtype=np.float32)

    def accept(self, samples):
        self.buffer = np.concatenate(
            [self.buffer, np.asarray(samples, dtype=np.float32)]
        )
        out = []
        while len(self.buffer) >= 480:
            pcm = (np.clip(self.buffer[:480], -1, 1) * 32767).astype("<i2")
            out.append(float(self.vad.is_speech(pcm.tobytes(), FS)))
            self.buffer = self.buffer[480:]
        return np.asarray(out, dtype=np.float32)


def marblenet(window_s, model_dir=None):
    """MarbleNet (local/baselines/models.py) as a windowed system (20 ms frames)."""
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
    from models import MarbleNet

    m = MarbleNet(model_dir=model_dir)

    def f(x):
        with torch.no_grad():
            xt = torch.from_numpy(np.ascontiguousarray(x)).to(m.device)
            n = torch.full((x.shape[0],), x.shape[1], device=m.device)
            logits = m.model(input_signal=xt, input_signal_length=n)
            return torch.softmax(logits, dim=-1)[..., 1].float().cpu().numpy()

    return WindowedOffline(
        f, 320, MarbleNet.shift, MarbleNet.offset, int(window_s * FS)
    )


SYSTEMS = {
    "silero": lambda a: SileroStream(),
    "webrtc0": lambda a: WebRTCStream(0),
    "marblenet": lambda a: marblenet(a.window_s, a.model_dir),
}


def load(entry: str):
    m = ENTRY.match(entry)
    if m and not Path(entry).exists():
        import soundfile

        with open(m.group(1), "rb") as f:
            f.seek(int(m.group(2)))
            data = f.read(int(m.group(3)) - int(m.group(2)))
        x, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    else:
        import soundfile

        x, rate = soundfile.read(entry, dtype="float32", always_2d=True)
    x = x[:, 0]
    if rate != FS:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(rate, FS)
        x = resample_poly(x, FS // g, rate // g).astype(np.float32)
    return x


def table(path):
    return dict(x.split(maxsplit=1) for x in Path(path).read_text().splitlines() if x)


def build(name: str, a) -> StreamingSystem:
    if name.startswith("ours:"):
        _, exp, ckpt = name.split(":")
        return OursStream(exp, ckpt)
    return SYSTEMS[name](a)


def main(extra_systems: Optional[dict] = None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--system",
        required=True,
        help="silero, webrtc0, marblenet or ours:<exp>:<ckpt>",
    )
    ap.add_argument(
        "--name", help="output name (default: the system, ours -> ours_la0)"
    )
    ap.add_argument("--block_ms", type=float, required=True)
    ap.add_argument("--window_s", type=float, default=2.0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--task", type=int, default=1)
    ap.add_argument("--ntasks", type=int, default=1)
    ap.add_argument("--model_dir", default=None)
    ap.add_argument(
        "--limit_hours",
        type=float,
        default=None,
        help="stop after this much audio (RTF measurement)",
    )
    a = ap.parse_args()
    if extra_systems:
        SYSTEMS.update(extra_systems)
    block = int(round(a.block_ms * FS / 1000))
    system = build(a.system, a)
    name = a.name or ("ours_la0" if a.system.startswith("ours:") else a.system)
    root = a.out / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "frames.json").write_text(
        json.dumps(
            dict(
                shift=system.shift,
                offset=system.offset,
                latency_ms=a.block_ms,
                window_s=a.window_s if isinstance(system, WindowedOffline) else None,
            )
        )
        + "\n"
    )
    items = []
    for s in map(Path, a.sets):
        n = {k: int(v) for k, v in table(s / "utt2num_samples").items()}
        items += [(n[r], s, r, e.strip()) for r, e in table(s / "wav.scp").items()]
    items.sort(key=lambda x: (-x[0], str(x[1]), x[2]))
    mine = items[a.task - 1 :: a.ntasks]
    scps, rows, audio_s, compute_s = {}, [], 0.0, 0.0
    for n, s, rid, entry in mine:
        x = load(entry)
        t0 = time.perf_counter()
        if isinstance(system, WindowedOffline):
            prob, emit = windowed_probs(system, x, block)
        else:
            prob, emit = simulate(system, x, block)
        dt = time.perf_counter() - t0
        lat = latencies(emit, system.shift, system.offset)
        od = root / "diar_eval" / f"diarized_{s.name}"
        (od / "predictions").mkdir(parents=True, exist_ok=True)
        path = od / "predictions" / f"{rid}.npy"
        np.save(path, prob[:, None])
        scps.setdefault(od, []).append(f"{rid} {path.resolve()}\n")
        rows.append(
            f"{s.name}\t{rid}\t{len(prob)}\t{1000 * lat.max() if len(lat) else 0:.2f}"
            f"\t{len(x) / FS:.2f}\t{dt:.2f}\n"
        )
        audio_s += len(x) / FS
        compute_s += dt
        if a.limit_hours and audio_s >= a.limit_hours * 3600:
            break
    for od, lines in scps.items():
        (od / f"diarize.{a.task}.scp").write_text("".join(lines))
    (root / f"latency.{a.task}.tsv").write_text(
        "set\trecording\tframes\tmax_latency_ms\taudio_s\tcompute_s\n" + "".join(rows)
    )
    print(
        f"{name} task {a.task}/{a.ntasks}: {len(rows)} recordings, {audio_s / 3600:.2f} h "
        f"in {compute_s:.0f} s (RTF {compute_s / max(audio_s, 1e-9):.5f}), block "
        f"{a.block_ms} ms"
    )


if __name__ == "__main__":
    main()
