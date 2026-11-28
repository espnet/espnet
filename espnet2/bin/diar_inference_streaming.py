#!/usr/bin/env python3
"""Online voice activity detection: frame decisions as the audio arrives.

StreamingVAD feeds a VoiceActivityDetector (espnet2.bin.diar_inference) chunk
by chunk and returns each frame's speech probability as soon as its samples
are in, at look-ahead 0. With an encoder that has ``forward_streaming`` (the
TCNEncoder) only the new frames are computed, and the probabilities equal
those of the whole recording. `espnet detect --stream/--live` and the Space's
live tab use it.

    stream = VoiceActivityDetector.for_mode("online").stream()
    for chunk in microphone:
        probs = stream.accept(chunk, rate)
    stream.finish()
"""

from typing import List, Optional

import numpy as np
import torch

from espnet2.bin.diar_inference import Segment, VoiceActivityDetector, as_mono_float


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
    """Streaming detection: each frame's probability as soon as its audio is in.

    By default the model runs with no look-ahead (online), and frame i of the
    STFT covers samples [i * hop, i * hop + n_fft), so a frame needs no future
    samples; this needs an STFT without centring (``frontend_conf: center:
    false``). With look_ahead > 0 (a dual-mode encoder) a frame is returned
    once the look_ahead frames after it are in. Either way the probabilities
    equal those of the whole recording at once at the same look-ahead.

    How each call runs the model:
    - an encoder with ``forward_streaming`` (the TCNEncoder): only the new
      frames go through the frontend, the normalisation, the encoder and the
      decoder; the buffer keeps the samples from the next frame's start on;
    - any other encoder: the new frames and the ``receptive_field - 1``
      frames before them are recomputed (the whole history when the encoder
      does not state its receptive field).
    """

    def __init__(
        self,
        vad: VoiceActivityDetector,
        keep_probabilities: bool = True,
        look_ahead: int = 0,
    ):
        """Start a stream on a detector.

        Args:
            vad: The detector whose model and rule the stream uses.
            keep_probabilities: Keep every frame's probability for
                all_probabilities() and segments(). A caller that reads only
                what accept() returns can pass False, so that a long stream
                keeps no history.
            look_ahead: Frames each frame reads ahead, up to the encoder's
                max_look_ahead (a dual-mode encoder). A frame's probability
                is returned once the look_ahead frames after it are in, and
                equals the whole recording's at that look-ahead. Above 0 the
                stream reruns the encoder on each new frame's receptive field
                and look-ahead instead of using forward_streaming.
        """
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
        if not 0 <= look_ahead <= vad.max_look_ahead:
            raise ValueError(
                f"look_ahead must be in [0, {vad.max_look_ahead}] for this "
                f"encoder, not {look_ahead}"
            )
        self.look_ahead = look_ahead
        self.vad = vad
        self.buffer = np.zeros(0, dtype=np.float32)
        self.offset = 0  # sample index of buffer[0] in the stream
        self.emitted = 0  # frames returned so far
        rf = vad.receptive_field
        self.context = None if rf is None else int(rf) - 1
        self.keep_probabilities = keep_probabilities
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
        # forward_streaming runs at look-ahead 0 only
        self.cached = (
            _CachedEncoder(encoder)
            if frame_wise and cacheable and look_ahead == 0
            else None
        )

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
        if self.resampler is None:
            return np.zeros(0, dtype=np.float32)
        return self._push(self._flush_resampler())

    def _flush_resampler(self) -> np.ndarray:
        tail = self.resampler.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
        self.resampler = None
        self.resampler_rate = None
        return np.asarray(tail, dtype=np.float32)

    def _push(self, chunk: np.ndarray) -> np.ndarray:
        """Add audio at the model's rate; return the new frames' probabilities."""
        vad = self.vad
        self.buffer = np.concatenate([self.buffer, chunk])
        total = self.num_samples
        if total < vad.n_fft:
            return np.zeros(0, dtype=np.float32)
        available = (total - vad.n_fft) // vad.hop + 1
        # frames whose look_ahead frames after them are in
        ready = available - self.look_ahead
        new = ready - self.emitted
        if new <= 0:
            return np.zeros(0, dtype=np.float32)
        if self.cached is not None:
            out = self._accept_cached(new)
        else:
            out = self._accept_recomputed(available, ready)
        self.emitted = ready
        if self.keep_probabilities:
            self.probs.append(out)
        return out

    def _accept_cached(self, new: int) -> np.ndarray:
        """Run only the new frames; keep the samples from the next frame on."""
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

    def _accept_recomputed(self, available: int, ready: int) -> np.ndarray:
        """Rerun the new frames with the receptive field before them.

        Frames emitted..ready-1 are returned. The run covers the
        receptive_field - 1 frames before them and the frames after them up
        to the last one in (look_ahead of them), so each returned frame reads
        exactly what it reads in the whole recording.
        """
        vad = self.vad
        first = 0 if self.context is None else max(0, self.emitted - self.context)
        start = first * vad.hop - self.offset
        end = (available - 1) * vad.hop + vad.n_fft - self.offset
        probs = vad.probabilities(
            self.buffer[start:end], vad.fs, look_ahead=self.look_ahead
        )
        if self.context is not None:
            # the next run starts receptive_field - 1 frames before frame ready
            keep = max(0, ready - self.context) * vad.hop
            self.buffer = self.buffer[keep - self.offset :]
            self.offset = keep
        return probs[self.emitted - first : ready - first]

    def all_probabilities(self) -> np.ndarray:
        """The probabilities of every frame returned so far."""
        if not self.keep_probabilities:
            raise ValueError(
                "this stream keeps no probabilities (keep_probabilities=False)"
            )
        if not self.probs:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.probs)

    def segments(self) -> List[Segment]:
        """Speech segments of the frames so far."""
        return self.vad.segments(self.all_probabilities())


def get_parser():
    """The parser of `espnet detect`, whose --stream and --live use this module."""
    import argparse

    from espnet2.bin import cli

    for action in cli.build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices["detect"]
    raise RuntimeError("espnet has no detect command")


def main(cmd=None):
    """Run `espnet detect` with these arguments (add --stream or --live)."""
    import sys

    from espnet2.bin import cli

    return cli.main(["detect", *(sys.argv[1:] if cmd is None else cmd)])


if __name__ == "__main__":
    raise SystemExit(main())
