from typing import Optional, Union

import numpy as np
from typeguard import typechecked

from espnet2.sds.vad.abs_frame_vad import AbsFrameVAD
from espnet2.sds.vad.turn_detector import TurnDetector


class FrontiaFrameVAD(AbsFrameVAD):
    """frontia-vad frame scores, computed by StreamingVAD.

    StreamingVAD (espnet2.bin.diar_inference_streaming) runs the model at
    look_ahead and keeps no score history here (keep_probabilities=False).
    """

    def __init__(self, detector, look_ahead: int = 0):
        """Score frames with a loaded VoiceActivityDetector.

        Args:
            detector: A VoiceActivityDetector, or an object with its fs,
                n_fft, frame_shift and stream(keep_probabilities=...,
                look_ahead=...).
            look_ahead: Frames each frame reads ahead: 0 is online (causal);
                a dual-mode model such as frontia_vad_v2_3 takes up to its
                max_look_ahead (124 frames, 0.99 s), its offline mode.
        """
        self.detector = detector
        self.look_ahead = look_ahead
        self.reset()

    @property
    def frame_shift(self) -> float:
        return self.detector.frame_shift

    @property
    def window_s(self) -> float:
        """A frame's probability needs its STFT window and look_ahead frames."""
        d = self.detector
        return d.n_fft / d.fs + self.look_ahead * d.frame_shift

    def accept(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return self.stream.accept(audio, sample_rate)

    def reset(self):
        self.stream = self.detector.stream(
            keep_probabilities=False, look_ahead=self.look_ahead
        )
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
    TurnDetector for the rule. look_ahead=0 (the default) runs the model
    online; look_ahead="max" runs a dual-mode model such as frontia_vad_v2_3
    in its offline mode (124 frames, 0.99 s ahead), which delays every
    decision by that much. The model, threshold and median default to
    ESPnet's VAD of that mode: VAD_MODELS and VAD_RULES["online"] or
    ["offline"] in espnet2.bin.diar_inference. With the default median of 25
    every decision waits a further 96 ms; online, median=1 makes the
    detection strictly causal.
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
        look_ahead: Union[int, str] = 0,
    ):
        """Load the model and set the end-of-turn rule.

        Args:
            tag (str, optional):
                The model tag on Hugging Face, or a local .zip.
                Defaults to VAD_MODELS of the mode.
            threshold (float, optional):
                Defaults to the threshold of VAD_RULES of the mode.
            median (int, optional):
                Defaults to the median of VAD_RULES of the mode.
            device (str, optional):
                The device the model runs on. Defaults to "cuda", as the
                other SDS modules do.
            look_ahead (int or "max", optional):
                Frames each frame reads ahead. 0 (online, the default) or
                up to the model's max_look_ahead; "max" takes that (offline).
                The mode is "offline" when it is above 0.

        The other arguments are TurnDetector's.
        """
        from espnet2.bin.diar_inference import (
            VAD_MODELS,
            VAD_RULES,
            VoiceActivityDetector,
        )

        if isinstance(look_ahead, str) and look_ahead != "max":
            raise ValueError(f'look_ahead is an int or "max", not {look_ahead!r}')
        mode = "online" if look_ahead == 0 else "offline"
        default_threshold, default_median = VAD_RULES[mode]
        tag = VAD_MODELS[mode] if tag is None else tag
        threshold = default_threshold if threshold is None else threshold
        median = default_median if median is None else median
        detector = VoiceActivityDetector.load(
            tag, device=device, threshold=threshold, median=median
        )
        if look_ahead == "max":
            look_ahead = detector.max_look_ahead
        super().__init__(
            FrontiaFrameVAD(detector, look_ahead),
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
    def from_detector(
        cls, detector, look_ahead: int = 0, **kwargs
    ) -> "FrontiaVADModel":
        """Wrap a loaded VoiceActivityDetector instead of loading a tag.

        threshold and median default to the detector's; the other keyword
        arguments are TurnDetector's.
        """
        kwargs.setdefault("threshold", detector.threshold)
        kwargs.setdefault("median", detector.median)
        self = cls.__new__(cls)
        TurnDetector.__init__(self, FrontiaFrameVAD(detector, look_ahead), **kwargs)
        return self
