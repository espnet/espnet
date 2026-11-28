import warnings
from typing import Optional

import numpy as np
from typeguard import typechecked

from espnet2.sds.vad.abs_frame_vad import AbsFrameVAD
from espnet2.sds.vad.turn_detector import TurnDetector

try:
    import webrtcvad

    is_webrtcvad_available = True
except ImportError:
    is_webrtcvad_available = False

# The rates webrtcvad accepts; other audio is resampled to 16 kHz.
WEBRTC_RATES = (8000, 16000, 32000, 48000)


class WebrtcFrameVAD(AbsFrameVAD):
    """WebRTC VAD frame decisions: 1.0 for speech, 0.0 otherwise."""

    def __init__(self, mode: int = 3, frame_ms: int = 20):
        """Score frames with webrtcvad.

        Args:
            mode (int, optional): webrtcvad's aggressiveness, 0 (least) to 3
                (most aggressive in calling audio non-speech). Defaults to 3.
            frame_ms (int, optional): The frame length, 10, 20 or 30 ms.
                Defaults to 20.
        """
        if not is_webrtcvad_available:
            raise ImportError("Error: webrtcvad is not properly installed.")
        if frame_ms not in (10, 20, 30):
            raise ValueError(f"frame_ms must be 10, 20 or 30, not {frame_ms}")
        self.mode = mode
        self.frame_ms = frame_ms
        self.reset()

    @property
    def frame_shift(self) -> float:
        return self.frame_ms / 1000

    @property
    def window_s(self) -> float:
        """Frames do not overlap: a frame's decision needs the frame alone."""
        return self.frame_ms / 1000

    def reset(self):
        self.vad = webrtcvad.Vad(self.mode)
        # int16 samples not yet in a whole frame
        self.buffer = np.zeros(0, dtype=np.int16)
        self.resampler = None

    def accept(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate not in WEBRTC_RATES:
            # One stream resampler across chunks, so that the stream is
            # resampled as one signal.
            import soxr

            if self.resampler is None:
                self.resampler = soxr.ResampleStream(
                    sample_rate, 16000, 1, dtype="float32"
                )
            audio = self.resampler.resample_chunk(audio.astype(np.float32))
            sample_rate = 16000
        pcm = np.clip(np.round(audio * 32768), -32768, 32767).astype(np.int16)
        self.buffer = np.concatenate([self.buffer, pcm])
        size = sample_rate * self.frame_ms // 1000
        n = len(self.buffer) // size
        scores = np.array(
            [
                float(
                    self.vad.is_speech(
                        self.buffer[i * size : (i + 1) * size].tobytes(), sample_rate
                    )
                )
                for i in range(n)
            ],
            dtype=np.float32,
        )
        self.buffer = self.buffer[n * size :]
        return scores


class WebrtcVADModel(TurnDetector):
    """WebRTC VAD behind the SDS VAD interface.

    WebrtcFrameVAD decides each 20 ms frame and TurnDetector finds the turns;
    see TurnDetector for the rule. The default centred median of 11 frames
    (220 ms) is a majority vote over the frame and the 5 on each side, so
    every decision waits 100 ms after its frame.
    """

    @typechecked
    def __init__(
        self,
        # The five positional parameters of the old per-chunk class, in their
        # order, so that existing positional calls still mean the same.
        speakup_threshold: Optional[int] = None,
        continue_threshold: Optional[int] = None,
        min_speech_ms: float = 500.0,
        max_speech_ms: Optional[float] = None,
        target_sr: int = 16000,
        *,
        mode: int = 3,
        median: int = 11,
        hangover_ms: float = 500.0,
        pad_start_ms: float = 300.0,
        pad_end_ms: float = 300.0,
        max_turn_ms: float = 30000.0,
    ):
        """Set up WebRTC VAD and the end-of-turn rule.

        The first five arguments keep the old class's positions; the others
        are keyword-only.

        Args:
            mode (int, optional): webrtcvad's aggressiveness, 0 to 3.
                Defaults to 3.
            median (int, optional): The centred median filter over the 0/1
                frame decisions, in frames. Defaults to 11 (220 ms).
            min_speech_ms (float, optional): A turn with less speech than
                this is dropped. Defaults to 500 ms.
            speakup_threshold, continue_threshold (int, optional): Ignored,
                with a warning. They counted speech frames per chunk, so they
                depended on the chunk length; the median filter replaces
                them.
            max_speech_ms (float, optional): Deprecated: passed on as
                max_turn_ms, with a warning. A turn that reaches it is now
                split, not dropped.

        The other arguments are TurnDetector's.
        """
        if speakup_threshold is not None or continue_threshold is not None:
            warnings.warn(
                "speakup_threshold and continue_threshold are ignored: "
                "WebrtcVADModel now decides per 20 ms frame with a median "
                "filter (median=)",
                DeprecationWarning,
                stacklevel=2,
            )
        if max_speech_ms is not None:
            warnings.warn(
                "max_speech_ms is deprecated; use max_turn_ms. A turn that "
                "reaches it is split, not dropped",
                DeprecationWarning,
                stacklevel=2,
            )
            if max_speech_ms != float("inf"):
                max_turn_ms = max_speech_ms
        super().__init__(
            WebrtcFrameVAD(mode),
            threshold=0.5,
            median=median,
            hangover_ms=hangover_ms,
            pad_start_ms=pad_start_ms,
            pad_end_ms=pad_end_ms,
            min_speech_ms=min_speech_ms,
            max_turn_ms=max_turn_ms,
            target_sr=target_sr,
        )
