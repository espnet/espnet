#!/usr/bin/env python3
"""Voice activity detection with a one-speaker diarization model.

A VAD here is a diar1 model trained with ``num_spk 1`` (egs2/frontia_vad_v1/diar1):
each frame gets the probability that someone is speaking. This module turns
those probabilities into what the front ends show - speech segments - in the
three ways a VAD is used:

- online: :meth:`VoiceActivityDetector.stream` takes audio chunk by chunk and
  returns each frame's probability as soon as its samples have arrived;
- offline: :meth:`VoiceActivityDetector.detect` reads a whole recording, with
  the largest look-ahead the encoder offers (a dual-mode encoder reads future
  frames; a causal one reads none, and offline equals online);
- batched: :meth:`VoiceActivityDetector.detect_files` runs many recordings
  through ``DiarizeSpeech.diarize_batch``, for cleaning a corpus.

`espnet detect`, the MCP tool ``detect``, the Space in
egs2/frontia_vad_v1/diar1/demo and the notebook Demos/diar_vad_demo.ipynb all call
this module, so the four give the same segments for the same file.

    vad = VoiceActivityDetector.for_mode("offline")  # or a .zip: model_tag=
    for seg in vad.detect_file("meeting.wav"):
        print(f"{seg.start:.2f}\t{seg.end:.2f}")

Two models are published (MODELS): a causal one for online detection and a
dual-mode one for offline detection, each with its own decision rule (RULES).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence, Union

# numpy and torch are imported where they are used, so that the command line,
# which reads MODELS and RULES for its defaults, starts without them
if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

# The two published models, one place to edit when they are named:
# - online: the causal TCN (v1), which reads no future frame and so streams;
# - offline: the dual-mode TCN, run with its largest look-ahead (124 frames,
#   0.99 s), which places the boundaries better when the whole file is there.
# Each of `espnet detect`, the MCP tool, the Space and the notebook takes these
# as defaults and lets a local .zip from stage 8 stand in for either.
MODELS = {
    "online": "espnet/frontia_vad_v1_online",
    "offline": "espnet/frontia_vad_v1_offline",
}

# Each model's decision rule, make_rttm.py's as stage 7 scores it: a frame is
# speech when its probability exceeds the threshold, then a median filter of
# that many frames smooths the 0/1 sequence. The values are each model's
# setting with the lowest detection error on AMI MH dev
# (egs2/frontia_vad_v1/diar1/local/show_vad_result.py), as (threshold, median).
RULES = {
    # v1, scored on 21 sets with a UEM: RESULTS_vad.md of the recipe's run
    "online": (0.7, 11),
    # the dual-mode model at look-ahead 124, scored on the same 21 sets
    "offline": (0.8, 11),
}
THRESHOLD, MEDIAN = RULES["offline"]


@dataclass
class Segment:
    """A stretch of speech, in seconds from the start of the recording."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def frames_to_segments(
    probs: np.ndarray,
    frame_shift: float,
    threshold: float = THRESHOLD,
    median: int = MEDIAN,
) -> List[Segment]:
    """Threshold and median-filter frame probabilities into speech segments.

    This is make_rttm.py's rule: frame i is speech when probs[i] > threshold,
    the 0/1 sequence is median-filtered over ``median`` frames (odd; 1 turns
    the filter off), and a run of speech frames s..e-1 becomes the segment
    [s * frame_shift, e * frame_shift).
    """
    import numpy as np

    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if median < 1 or median % 2 == 0:
        raise ValueError(f"median must be a positive odd number: {median}")
    active = (probs > threshold).astype(np.int64)
    if median > 1 and len(active) > 0:
        from scipy.signal import medfilt

        active = medfilt(active, median).astype(np.int64)
    edges = np.flatnonzero(np.diff(np.pad(active, (1, 1))))
    return [
        Segment(float(s * frame_shift), float(e * frame_shift))
        for s, e in zip(edges[::2], edges[1::2])
    ]


def read_audio(path: Union[str, Path], fs: int) -> np.ndarray:
    """Read an audio file as mono float32 at ``fs``.

    The first channel of a multi-channel file is used, as the model was
    trained on one channel.
    """
    import soundfile

    speech, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    return resample(speech[:, 0], rate, fs)


def resample(speech: np.ndarray, rate: int, fs: int) -> np.ndarray:
    """Return ``speech`` (1-D) at ``fs``."""
    import numpy as np

    speech = np.asarray(speech, dtype=np.float32).reshape(-1)
    if rate == fs:
        return speech
    import librosa

    return librosa.resample(speech, orig_sr=rate, target_sr=fs).astype(np.float32)


class VoiceActivityDetector:
    """Speech probabilities and segments from a one-speaker diar1 model.

    Args:
        diarize: A loaded ``DiarizeSpeech`` with ``num_spk=1``.
        threshold: Probability above which a frame is speech.
        median: Median filter length in frames (odd; 1 for none).
    """

    def __init__(
        self,
        diarize,
        threshold: float = THRESHOLD,
        median: int = MEDIAN,
    ):
        import humanfriendly

        self.diarize = diarize
        self.threshold = threshold
        self.median = median
        args = diarize.diar_train_args
        conf = args.frontend_conf or {}
        self.fs = int(humanfriendly.parse_size(str(conf.get("fs", 16000))))
        self.n_fft = int(conf.get("n_fft", 512))
        self.hop = int(conf.get("hop_length", 128))
        self.center = bool(conf.get("center", True))
        encoder = diarize.diar_model.encoder
        # A dual-mode encoder (AbsLookAheadEncoder) reads up to max_look_ahead
        # future frames; any other encoder reads what it was built to read.
        self.max_look_ahead = int(getattr(encoder, "max_look_ahead", 0))
        # Input frames each output frame depends on; None when the encoder
        # does not say, and then streaming keeps the whole history.
        self.receptive_field = getattr(encoder, "receptive_field", None)

    @classmethod
    def load(
        cls,
        model_tag: str,
        device: str = "cpu",
        threshold: float = THRESHOLD,
        median: int = MEDIAN,
    ) -> "VoiceActivityDetector":
        """Load a packed model: a Hugging Face tag or a local .zip from stage 8.

        DiarizeSpeech.from_pretrained fetches it; this wraps the result.
        """
        from espnet2.bin.diar_inference import DiarizeSpeech

        diarize = DiarizeSpeech.from_pretrained(model_tag, device=device, num_spk=1)
        return cls(diarize, threshold=threshold, median=median)

    @classmethod
    def for_mode(
        cls, mode: str, model_tag: Optional[str] = None, device: str = "cpu"
    ) -> "VoiceActivityDetector":
        """The published model of a mode ("online" or "offline"), with its rule.

        ``model_tag`` replaces the model (a tag or a local .zip) and keeps the
        mode's rule.
        """
        if mode not in MODELS:
            raise ValueError(f'mode is "online" or "offline", not {mode!r}')
        threshold, median = RULES[mode]
        return cls.load(
            model_tag or MODELS[mode], device=device, threshold=threshold, median=median
        )

    @property
    def frame_shift(self) -> float:
        """Seconds between frames."""
        return self.hop / self.fs

    @property
    def offline_look_ahead(self) -> int:
        """The look-ahead offline detection uses: the largest the encoder offers."""
        return self.max_look_ahead

    def _set_look_ahead(self, look_ahead: int):
        encoder = self.diarize.diar_model.encoder
        if look_ahead > self.max_look_ahead:
            raise ValueError(
                f"look_ahead {look_ahead} exceeds this encoder's largest, "
                f"{self.max_look_ahead}"
            )
        if hasattr(encoder, "set_look_ahead"):
            encoder.set_look_ahead(look_ahead)

    def segments(self, probs: np.ndarray) -> List[Segment]:
        """Speech segments of frame probabilities, by this detector's rule."""
        return frames_to_segments(probs, self.frame_shift, self.threshold, self.median)

    def probabilities(
        self, speech: np.ndarray, fs: Optional[int] = None, look_ahead: int = 0
    ) -> np.ndarray:
        """Speech probability of each frame of one recording (1-D at ``fs``)."""
        import numpy as np
        import torch

        speech = resample(speech, fs or self.fs, self.fs)
        self._set_look_ahead(look_ahead)
        if len(speech) < self.n_fft:
            # too short for one frame
            return np.zeros(0, dtype=np.float32)
        with torch.no_grad():
            _, probs = self.diarize(torch.from_numpy(speech)[None], fs=self.fs)
        return np.asarray(probs, dtype=np.float32)[0, :, 0]

    def probabilities_batch(
        self, speeches: Sequence[np.ndarray], look_ahead: int = 0
    ) -> List[np.ndarray]:
        """Frame probabilities of recordings at ``self.fs``, in one batch.

        Each result equals :meth:`probabilities` of that recording alone
        (DiarizeSpeech.diarize_batch).
        """
        import numpy as np
        import torch

        self._set_look_ahead(look_ahead)
        lengths = [len(s) for s in speeches]
        batch = np.zeros((len(speeches), max(lengths)), dtype=np.float32)
        for b, s in enumerate(speeches):
            batch[b, : len(s)] = s
        with torch.no_grad():
            out = self.diarize.diarize_batch(batch, np.asarray(lengths), fs=self.fs)
        return [np.asarray(p, dtype=np.float32)[:, 0] for p in out]

    def detect(
        self, speech: np.ndarray, fs: Optional[int] = None, offline: bool = True
    ) -> List[Segment]:
        """Speech segments of one recording.

        offline=True reads the largest look-ahead the encoder offers; False
        reads none, as the online stream does.
        """
        look_ahead = self.offline_look_ahead if offline else 0
        return self.segments(self.probabilities(speech, fs, look_ahead))

    def detect_file(self, path: Union[str, Path], offline: bool = True):
        """Speech segments of an audio file."""
        return self.detect(read_audio(path, self.fs), self.fs, offline)

    def detect_files(
        self,
        paths: Sequence[Union[str, Path]],
        offline: bool = True,
        batch_size: int = 8,
    ) -> List[List[Segment]]:
        """Speech segments of many files, ``batch_size`` recordings at a time.

        The files are batched in order of length, so a batch holds recordings
        of similar length and little padding; the results come back in the
        order of ``paths``.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive: {batch_size}")
        look_ahead = self.offline_look_ahead if offline else 0
        speeches = [read_audio(p, self.fs) for p in paths]
        order = sorted(range(len(speeches)), key=lambda i: len(speeches[i]))
        results: List[Optional[List[Segment]]] = [None] * len(speeches)
        for k in range(0, len(order), batch_size):
            group = order[k : k + batch_size]
            # recordings shorter than one frame have no frames to detect
            ready = [i for i in group if len(speeches[i]) >= self.n_fft]
            for i in group:
                results[i] = []
            if not ready:
                continue
            probs = self.probabilities_batch([speeches[i] for i in ready], look_ahead)
            for i, p in zip(ready, probs):
                results[i] = self.segments(p)
        return results

    def stream(self) -> "StreamingVAD":
        """Start an online stream: feed chunks, get frame probabilities back."""
        return StreamingVAD(self)


class _CachedEncoder:
    """Runs a causal encoder on new frames only, keeping what it needs of the past.

    Two kinds of encoder qualify:
    - one with ``forward_streaming(feats, state) -> (out, state)`` (state None
      at the start), which owns its cache;
    - the causal Conv-TasNet TCNEncoder of #6995 (v1), cached here: each
      block keeps the last (kernel - 1) * dilation inputs of its dilated
      depthwise convolution, which starts as zeros, as the causal left
      padding does. Every other layer of it works frame by frame.
    """

    def __init__(self, encoder):
        self.encoder = encoder
        self.state = None
        self.blocks = None
        if not hasattr(encoder, "forward_streaming"):
            net = encoder.tcn.network
            self.head = net[:2]  # input cLN, bottleneck 1x1
            self.tail = net[3:]  # (PReLU,) output 1x1
            self.blocks = []
            for repeat in net[2]:
                for block in repeat:
                    conv1x1, prelu, norm, dsconv = block.net
                    depthwise = dsconv.net[0]
                    # dsconv.net is depthwise, (chomp,) PReLU, norm, pointwise;
                    # the chomp drops the right padding, which here is never added
                    rest = dsconv.net[2:] if len(dsconv.net) == 5 else dsconv.net[1:]
                    context = (depthwise.kernel_size[0] - 1) * depthwise.dilation[0]
                    self.blocks.append((conv1x1, prelu, norm, depthwise, rest, context))
            self.caches = None

    @staticmethod
    def supports(encoder) -> bool:
        if hasattr(encoder, "forward_streaming"):
            return True
        tcn = getattr(encoder, "tcn", None)
        return (
            getattr(encoder, "causal", False)
            and tcn is not None
            and not getattr(tcn, "skip_connection", False)
            and hasattr(tcn, "network")
        )

    def __call__(self, feats):
        """Encode new frames (1, k, D) given everything before them."""
        import torch
        import torch.nn.functional as F

        if self.blocks is None:
            out, self.state = self.encoder.forward_streaming(feats, self.state)
            return out
        out = self.head(feats.transpose(1, 2))  # (1, B, k)
        if self.caches is None:
            self.caches = [
                out.new_zeros(out.size(0), dw.in_channels, ctx)
                for _, _, _, dw, _, ctx in self.blocks
            ]
        for i, (conv1x1, prelu, norm, dw, rest, ctx) in enumerate(self.blocks):
            h = torch.cat([self.caches[i], norm(prelu(conv1x1(out)))], dim=2)
            self.caches[i] = h[:, :, h.size(2) - ctx :]
            y = F.conv1d(h, dw.weight, dw.bias, dilation=dw.dilation, groups=dw.groups)
            out = out + rest(y)
        return self.tail(out).transpose(1, 2)  # (1, k, D)


class StreamingVAD:
    """Online detection: each frame's probability as soon as its audio is in.

    The model runs with no look-ahead, and frame i of the STFT covers samples
    [i * hop, i * hop + n_fft), so a frame needs no future samples; this needs
    an STFT without centring (``frontend_conf: center: false``). The
    probabilities equal those of the whole recording at once.

    How each call runs the model:
    - a causal encoder that can be cached (``_CachedEncoder``): only the new
      frames go through the frontend, the normalisation, the encoder and the
      decoder; the buffer keeps the samples from the next frame's start on;
    - any other encoder: the new frames and the ``receptive_field - 1``
      frames before them are recomputed (the whole history when the encoder
      does not state its receptive field).
    """

    def __init__(self, vad: VoiceActivityDetector):
        if vad.center:
            raise ValueError(
                "streaming needs a model whose STFT does not centre its frames "
                "(frontend_conf center: false): a centred frame reads n_fft/2 "
                "samples ahead"
            )
        import numpy as np

        self.vad = vad
        self.buffer = np.zeros(0, dtype=np.float32)
        self.offset = 0  # sample index of buffer[0] in the stream
        self.emitted = 0  # frames returned so far
        rf = vad.receptive_field
        self.context = None if rf is None else int(rf) - 1
        self.probs: List[np.ndarray] = []
        self.resampler = None  # soxr.ResampleStream for input at another rate
        self.resampler_rate = None
        model = vad.diarize.diar_model
        # Only the encoder's past is cached, so everything around it has to
        # work frame by frame: the global (not the utterance) MVN, and a linear
        # decoder without attractors.
        frame_wise = (
            getattr(model, "attractor", None) is None
            and type(getattr(model, "normalize", None)).__name__
            in ("GlobalMVN", "NoneType")
            and type(getattr(model, "decoder", None)).__name__ == "LinearDecoder"
        )
        encoder = model.encoder
        cacheable = hasattr(encoder, "forward_streaming") or (
            vad.max_look_ahead == 0 and _CachedEncoder.supports(encoder)
        )
        vad._set_look_ahead(0)
        self.cached = _CachedEncoder(encoder) if frame_wise and cacheable else None

    @property
    def num_samples(self) -> int:
        return self.offset + len(self.buffer)

    def accept(self, chunk: np.ndarray, fs: Optional[int] = None) -> np.ndarray:
        """Add a chunk of audio; return the probabilities of the new frames.

        A chunk at another rate goes through one soxr stream resampler kept
        across chunks, so the stream is resampled as one signal rather than
        chunk by chunk. The resampler holds back a few milliseconds of audio
        until the next chunk.
        """
        import numpy as np

        vad = self.vad
        rate = fs or vad.fs
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if rate != vad.fs:
            import soxr

            if self.resampler is None or self.resampler_rate != rate:
                self.resampler = soxr.ResampleStream(rate, vad.fs, 1, dtype="float32")
                self.resampler_rate = rate
            chunk = self.resampler.resample_chunk(chunk)
        self.buffer = np.concatenate([self.buffer, chunk])
        total = self.num_samples
        if total < vad.n_fft:
            return np.zeros(0, dtype=np.float32)
        available = (total - vad.n_fft) // vad.hop + 1
        new = available - self.emitted
        if new <= 0:
            return np.zeros(0, dtype=np.float32)
        if self.cached is not None:
            out = self._accept_cached(new)
        else:
            out = self._accept_recomputed(available, new)
        self.emitted = available
        self.probs.append(out)
        return out

    def _accept_cached(self, new: int) -> np.ndarray:
        """Run only the new frames; keep the samples from the next frame on."""
        import numpy as np
        import torch

        vad = self.vad
        model = vad.diarize.diar_model
        device = next(model.parameters()).device
        # buffer[0] is the first sample of the first new frame
        end = (new - 1) * vad.hop + vad.n_fft
        x = torch.from_numpy(self.buffer[:end]).to(device)[None]
        with torch.no_grad():
            lengths = torch.tensor([x.size(1)], device=device)
            feats, flens = model._extract_feats(x, lengths)
            if model.normalize is not None:
                feats, flens = model.normalize(feats, flens)
            out = self.cached(feats)
            logits = model.decoder(out, flens)
        keep = new * vad.hop
        self.buffer = self.buffer[keep:]
        self.offset += keep
        return torch.sigmoid(logits)[0, :, 0].float().cpu().numpy().astype(np.float32)

    def _accept_recomputed(self, available: int, new: int) -> np.ndarray:
        """Rerun the new frames with the receptive field before them."""
        vad = self.vad
        first = 0 if self.context is None else max(0, self.emitted - self.context)
        start = first * vad.hop - self.offset
        end = (available - 1) * vad.hop + vad.n_fft - self.offset
        probs = vad.probabilities(self.buffer[start:end], vad.fs, look_ahead=0)
        if self.context is not None:
            keep = max(0, available - self.context) * vad.hop
            self.buffer = self.buffer[keep - self.offset :]
            self.offset = keep
        return probs[-new:]

    def all_probabilities(self) -> np.ndarray:
        """The probabilities of every frame returned so far."""
        import numpy as np

        if not self.probs:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.probs)

    def segments(self) -> List[Segment]:
        """Speech segments of the frames so far."""
        return self.vad.segments(self.all_probabilities())


def format_segments(segments: Iterable[Segment]) -> str:
    """One line a segment: start and end seconds, tab-separated."""
    return "\n".join(f"{s.start:.2f}\t{s.end:.2f}" for s in segments)
