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


def read_audio(
    path: Union[str, Path], fs: int, max_seconds: Optional[float] = None
) -> np.ndarray:
    """Read an audio file as mono float32 at ``fs``.

    The first channel of a multi-channel file is used, as the model was
    trained on one channel. With ``max_seconds``, only that much of the file
    is decoded and resampled.
    """
    import soundfile

    frames = -1
    if max_seconds is not None:
        frames = int(max_seconds * soundfile.info(str(path)).samplerate)
    speech, rate = soundfile.read(
        str(path), frames=frames, dtype="float32", always_2d=True
    )
    return resample(speech[:, 0], rate, fs)


def audio_seconds(path: Union[str, Path]) -> float:
    """The length of an audio file in seconds, from its header."""
    import soundfile

    info = soundfile.info(str(path))
    return info.frames / info.samplerate


def as_mono_float(speech) -> np.ndarray:
    """Return audio as a 1-D float32 array in [-1, 1].

    Integer PCM (int16, int32, ...) is scaled by its largest value; a
    (samples, 1) array is squeezed. Several channels are refused rather than
    flattened, which would interleave them into one signal, and so are NaN
    and infinite values, which would make every frame non-speech.
    """
    import numpy as np

    speech = np.asarray(speech)
    if speech.ndim == 2 and 1 in speech.shape:
        speech = speech.reshape(-1)
    if speech.ndim != 1:
        raise ValueError(
            f"audio must be one channel, (samples,), not {speech.shape}; "
            "read_audio takes the first channel of a file"
        )
    if speech.dtype.kind in "iu":
        info = np.iinfo(speech.dtype)
        speech = (speech.astype(np.float64) - (info.max + info.min + 1) / 2) / (
            (info.max - info.min + 1) / 2
        )
    speech = speech.astype(np.float32)
    if not np.isfinite(speech).all():
        raise ValueError("audio contains NaN or infinite values")
    return speech


def resample(speech: np.ndarray, rate: int, fs: int) -> np.ndarray:
    """Return ``speech`` (1-D) at ``fs``, as :func:`as_mono_float` reads it."""
    import numpy as np

    speech = as_mono_float(speech)
    if rate <= 0:
        raise ValueError(f"the sample rate must be positive, not {rate}")
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
    def min_samples(self) -> int:
        """The shortest input with one frame: n_fft, or n_fft / 2 + 1 centred.

        A centred STFT pads each end by reflection, which needs more than
        n_fft / 2 samples.
        """
        return self.n_fft // 2 + 1 if self.center else self.n_fft

    def num_frames(self, num_samples: int) -> int:
        """Frames the frontend makes of ``num_samples`` samples."""
        if num_samples < self.min_samples:
            return 0
        if self.center:
            return num_samples // self.hop + 1
        return (num_samples - self.n_fft) // self.hop + 1

    def _check_frames(self, probs, num_samples: int):
        """Refuse an encoder that changes the frame rate.

        Times are frame index * hop / fs, so a subsampling encoder would put
        every segment at the wrong time.
        """
        if len(probs) != self.num_frames(num_samples):
            raise ValueError(
                f"{num_samples} samples gave {len(probs)} frames, not the "
                f"{self.num_frames(num_samples)} of the frontend: an encoder that "
                "subsamples is not supported"
            )

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

        speech = resample(speech, self.fs if fs is None else fs, self.fs)
        self._set_look_ahead(look_ahead)
        if len(speech) < self.min_samples:
            # too short for one frame
            return np.zeros(0, dtype=np.float32)
        with torch.no_grad():
            _, probs = self.diarize(torch.from_numpy(speech)[None], fs=self.fs)
        probs = np.asarray(probs, dtype=np.float32)[0, :, 0]
        self._check_frames(probs, len(speech))
        return probs

    def probabilities_batch(
        self, speeches: Sequence[np.ndarray], look_ahead: int = 0
    ) -> List[np.ndarray]:
        """Frame probabilities of recordings at ``self.fs``, in one batch.

        Each result equals :meth:`probabilities` of that recording alone
        (DiarizeSpeech.diarize_batch); a recording shorter than one frame gets
        an empty array.
        """
        import numpy as np
        import torch

        self._set_look_ahead(look_ahead)
        speeches = [as_mono_float(s) for s in speeches]
        results = [np.zeros(0, dtype=np.float32) for _ in speeches]
        ready = [i for i, s in enumerate(speeches) if len(s) >= self.min_samples]
        if not ready:
            return results
        lengths = [len(speeches[i]) for i in ready]
        batch = np.zeros((len(ready), max(lengths)), dtype=np.float32)
        for b, i in enumerate(ready):
            batch[b, : lengths[b]] = speeches[i]
        with torch.no_grad():
            out = self.diarize.diarize_batch(batch, np.asarray(lengths), fs=self.fs)
        for i, n, p in zip(ready, lengths, out):
            results[i] = np.asarray(p, dtype=np.float32)[:, 0]
            self._check_frames(results[i], n)
        return results

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

        The files are batched in order of length (read from their headers), so
        a batch holds recordings of similar length and little padding; each
        batch is decoded when it runs, so only one batch of audio is in memory.
        The results come back in the order of ``paths``.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive: {batch_size}")
        look_ahead = self.offline_look_ahead if offline else 0
        order = sorted(range(len(paths)), key=lambda i: audio_seconds(paths[i]))
        results: List[List[Segment]] = [[] for _ in paths]
        for k in range(0, len(order), batch_size):
            group = order[k : k + batch_size]
            speeches = [read_audio(paths[i], self.fs) for i in group]
            for i, p in zip(group, self.probabilities_batch(speeches, look_ahead)):
                results[i] = self.segments(p)
        return results

    def stream(self) -> "StreamingVAD":
        """Start an online stream: feed chunks, get frame probabilities back."""
        return StreamingVAD(self)


class _CachedEncoder:
    """Runs an encoder on new frames only, through its ``forward_streaming``.

    ``forward_streaming(feats, state) -> (out, state)`` (state None at the
    start) keeps what the encoder needs of the past, e.g. the TCNEncoder's
    last (kernel - 1) * dilation inputs of each dilated convolution.
    """

    def __init__(self, encoder):
        self.encoder = encoder
        self.state = None

    @staticmethod
    def supports(encoder) -> bool:
        return hasattr(encoder, "forward_streaming")

    def __call__(self, feats):
        """Encode new frames (1, k, D) given everything before them."""
        out, self.state = self.encoder.forward_streaming(feats, self.state)
        return out


class StreamingVAD:
    """Online detection: each frame's probability as soon as its audio is in.

    The model runs with no look-ahead, and frame i of the STFT covers samples
    [i * hop, i * hop + n_fft), so a frame needs no future samples; this needs
    an STFT without centring (``frontend_conf: center: false``). The
    probabilities equal those of the whole recording at once.

    How each call runs the model:
    - an encoder with ``forward_streaming`` (the TCNEncoder): only the new
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
        model = vad.diarize.diar_model
        normalize = type(getattr(model, "normalize", None)).__name__
        if normalize not in ("GlobalMVN", "NoneType"):
            # the statistics of an utterance include its future
            raise ValueError(
                f"streaming needs a normalisation that reads no future frame, "
                f"such as global_mvn, not {normalize}"
            )
        encoder = model.encoder
        # Online means look-ahead 0: a dual-mode encoder can be set to it, a
        # causal one is there already; any other encoder reads future frames.
        if not (
            hasattr(encoder, "forward_streaming")
            or hasattr(encoder, "set_look_ahead")
            or getattr(encoder, "causal", False)
        ):
            raise ValueError(
                f"streaming needs a causal encoder; {type(encoder).__name__} "
                "does not say it is one"
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
        # Only the encoder's past is cached, so the decoder has to work frame
        # by frame: a linear decoder without attractors.
        frame_wise = (
            getattr(model, "attractor", None) is None
            and type(getattr(model, "decoder", None)).__name__ == "LinearDecoder"
        )
        cacheable = _CachedEncoder.supports(encoder)
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
        until the next chunk; :meth:`finish` releases it.
        """
        import numpy as np

        vad = self.vad
        rate = vad.fs if fs is None else fs
        if rate <= 0:
            raise ValueError(f"the sample rate must be positive, not {rate}")
        chunk = as_mono_float(chunk)
        out = []
        if self.resampler is not None and self.resampler_rate != rate:
            # the audio of the old rate goes in before the new rate's
            out.append(self._push(self._flush_resampler()))
        if rate != vad.fs:
            import soxr

            if self.resampler is None:
                self.resampler = soxr.ResampleStream(rate, vad.fs, 1, dtype="float32")
                self.resampler_rate = rate
            chunk = self.resampler.resample_chunk(chunk)
        out.append(self._push(chunk))
        return np.concatenate(out)

    def finish(self) -> np.ndarray:
        """End the stream: run the audio the resampler still holds.

        Returns the probabilities of the frames it completes.
        """
        import numpy as np

        if self.resampler is None:
            return np.zeros(0, dtype=np.float32)
        return self._push(self._flush_resampler())

    def _flush_resampler(self) -> np.ndarray:
        import numpy as np

        tail = self.resampler.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
        self.resampler = None
        self.resampler_rate = None
        return np.asarray(tail, dtype=np.float32)

    def _push(self, chunk: np.ndarray) -> np.ndarray:
        """Add audio at the model's rate; return the new frames' probabilities."""
        import numpy as np

        vad = self.vad
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
        dtype = next(model.parameters()).dtype
        x = torch.from_numpy(self.buffer[:end]).to(device=device, dtype=dtype)[None]
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


def get_parser():
    """The parser of `espnet detect`, which this module implements."""
    import argparse

    from espnet2.bin import cli

    for action in cli.build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices["detect"]
    raise RuntimeError("espnet has no detect command")


def main(cmd=None):
    """Run `espnet detect` with these arguments."""
    import sys

    from espnet2.bin import cli

    return cli.main(["detect", *(sys.argv[1:] if cmd is None else cmd)])


if __name__ == "__main__":
    raise SystemExit(main())
