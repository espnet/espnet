from typing import Optional

import numpy as np
from typeguard import typechecked

from espnet2.sds.vad.abs_frame_vad import AbsFrameVAD
from espnet2.sds.vad.turn_detector import TurnDetector


class FrontiaFrameVAD(AbsFrameVAD):
    """frontia-vad frame scores, computed causally by StreamingVAD.

    StreamingVAD (espnet2.bin.diar_inference_streaming) runs the model at
    look-ahead 0 and keeps no score history here (keep_probabilities=False).
    """

    def __init__(self, detector):
        """Score frames with a loaded VoiceActivityDetector.

        Args:
            detector: A VoiceActivityDetector, or an object with its fs,
                n_fft, frame_shift and stream(keep_probabilities=...).
        """
        self.detector = detector
        self.reset()

    @property
    def frame_shift(self) -> float:
        return self.detector.frame_shift

    @property
    def window_s(self) -> float:
        """A frame's probability needs its whole STFT window, n_fft samples."""
        return self.detector.n_fft / self.detector.fs

    def accept(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return self.stream.accept(audio, sample_rate)

    def reset(self):
        self.stream = self.detector.stream(keep_probabilities=False)
        if (
            getattr(self.stream, "cached", True) is None
            and getattr(self.stream, "context", 0) is None
        ):
            # An encoder without forward_streaming or a receptive field makes
            # StreamingVAD rerun, and so keep, the whole audio of the stream.
            raise ValueError(
                "FrontiaVADModel needs an encoder with forward_streaming or a "
                "receptive_field, so that the stream keeps bounded audio"
            )

    def warmup(self):
        """Run one second of silence through the model, then reset."""
        fs = self.detector.fs
        self.stream.accept(np.zeros(fs, dtype=np.float32), fs)
        self.reset()


class FrontiaVADModel(TurnDetector):
    """frontia-vad, a neural VAD, behind the SDS VAD interface.

    FrontiaFrameVAD scores the frames and TurnDetector finds the turns; see
    TurnDetector for the rule. The model, threshold and median default to
    ESPnet's online VAD, VAD_MODELS["online"] and VAD_RULES["online"] in
    espnet2.bin.diar_inference. With the default median of 25 every decision
    waits 96 ms after its frame, so the detection is not strictly causal
    although the model reads no future audio; median=1 makes it causal.
    """

    @typechecked
    def __init__(
        self,
        tag: Optional[str] = None,
        threshold: Optional[float] = None,
        median: Optional[int] = None,
        hangover_ms: float = 500.0,
        pad_start_ms: float = 300.0,
        pad_end_ms: float = 300.0,
        min_speech_ms: float = 200.0,
        max_turn_ms: float = 30000.0,
        target_sr: int = 16000,
        device: str = "cuda",
    ):
        """Load the model and set the end-of-turn rule.

        Args:
            tag (str, optional):
                The model tag on Hugging Face, or a local .zip.
                Defaults to VAD_MODELS["online"].
            threshold (float, optional):
                Defaults to the threshold of VAD_RULES["online"].
            median (int, optional):
                Defaults to the median of VAD_RULES["online"].
            device (str, optional):
                The device the model runs on. Defaults to "cuda", as the
                other SDS modules do.

        The other arguments are TurnDetector's.
        """
        from espnet2.bin.diar_inference import (
            VAD_MODELS,
            VAD_RULES,
            VoiceActivityDetector,
        )

        default_threshold, default_median = VAD_RULES["online"]
        tag = VAD_MODELS["online"] if tag is None else tag
        threshold = default_threshold if threshold is None else threshold
        median = default_median if median is None else median
        detector = VoiceActivityDetector.load(
            tag, device=device, threshold=threshold, median=median
        )
        super().__init__(
            FrontiaFrameVAD(detector),
            threshold=threshold,
            median=median,
            hangover_ms=hangover_ms,
            pad_start_ms=pad_start_ms,
            pad_end_ms=pad_end_ms,
            min_speech_ms=min_speech_ms,
            max_turn_ms=max_turn_ms,
            target_sr=target_sr,
        )

    @classmethod
    def from_detector(cls, detector, **kwargs) -> "FrontiaVADModel":
        """Wrap a loaded VoiceActivityDetector instead of loading a tag.

        threshold and median default to the detector's; the other keyword
        arguments are TurnDetector's.
        """
        kwargs.setdefault("threshold", detector.threshold)
        kwargs.setdefault("median", detector.median)
        self = cls.__new__(cls)
        TurnDetector.__init__(self, FrontiaFrameVAD(detector), **kwargs)
        return self
