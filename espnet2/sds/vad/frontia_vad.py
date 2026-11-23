from typing import List, Optional

import librosa
import numpy as np
from typeguard import typechecked

from espnet2.sds.vad.abs_vad import AbsVAD


class FrontiaVADModel(AbsVAD):
    """frontia-vad, a neural VAD, behind the SDS VAD interface.

    The model scores each frame causally through StreamingVAD
    (espnet2.bin.diar_inference_streaming), so forward() takes the audio
    chunk by chunk, as WebrtcVADModel does.
    """

    @typechecked
    def __init__(
        self,
        tag: str = "espnet/frontia_vad_v2_3",
        threshold: float = 0.6,
        median: int = 25,
        hangover_ms: float = 500.0,
        pad_start_ms: float = 300.0,
        pad_end_ms: float = 300.0,
        min_speech_ms: float = 200.0,
        target_sr: int = 16000,
        device: str = "cpu",
    ):
        """Load the model and set the end-of-turn rule.

        Args:
            tag (str, optional):
                The model tag on Hugging Face, or a local .zip.
                Defaults to "espnet/frontia_vad_v2_3".
            threshold (float, optional):
                A frame is speech when its probability exceeds this value.
                Defaults to 0.6, the online rule on the model card.
            median (int, optional):
                The length in frames of the centred median filter applied to
                the 0/1 decisions. Defaults to 25, the online rule on the
                model card.
            hangover_ms (float, optional):
                The silence after speech that ends the user's turn.
                Defaults to 500 ms.
            pad_start_ms (float, optional):
                The audio kept before the first speech frame.
                Defaults to 300 ms.
            pad_end_ms (float, optional):
                The audio kept after the end of speech. It must not exceed
                hangover_ms, so that this audio is already in when the turn
                ends. Defaults to 300 ms.
            min_speech_ms (float, optional):
                A turn with less speech than this is dropped.
                Defaults to 200 ms.
            target_sr (int, optional):
                The sampling rate of the float audio forward() returns.
                Defaults to 16000 Hz.
            device (str, optional):
                The device the model runs on. Defaults to "cpu".
        """
        from espnet2.bin.diar_inference import VoiceActivityDetector

        detector = VoiceActivityDetector.load(
            tag, device=device, threshold=threshold, median=median
        )
        self._setup(
            detector,
            hangover_ms,
            pad_start_ms,
            pad_end_ms,
            min_speech_ms,
            target_sr,
        )

    @classmethod
    def from_detector(
        cls,
        detector,
        hangover_ms: float = 500.0,
        pad_start_ms: float = 300.0,
        pad_end_ms: float = 300.0,
        min_speech_ms: float = 200.0,
        target_sr: int = 16000,
    ) -> "FrontiaVADModel":
        """Wrap a loaded VoiceActivityDetector instead of loading a tag."""
        self = cls.__new__(cls)
        self._setup(
            detector, hangover_ms, pad_start_ms, pad_end_ms, min_speech_ms, target_sr
        )
        return self

    def _setup(
        self, detector, hangover_ms, pad_start_ms, pad_end_ms, min_speech_ms, target_sr
    ):
        if pad_end_ms > hangover_ms:
            raise ValueError(
                f"pad_end_ms ({pad_end_ms}) must not exceed hangover_ms "
                f"({hangover_ms}): the turn ends before that audio is in"
            )
        if detector.median < 1 or detector.median % 2 == 0:
            raise ValueError(f"median must be a positive odd number: {detector.median}")
        super().__init__()
        self.detector = detector
        self.hangover_frames = max(1, round(hangover_ms / 1000 / detector.frame_shift))
        self.pad_start_s = pad_start_ms / 1000
        self.pad_end_s = pad_end_ms / 1000
        self.min_speech_s = min_speech_ms / 1000
        self.target_sr = target_sr
        # Hold the last turn's frame scores, for a caller that plots them.
        self.last_turn: Optional[dict] = None
        self.reset()

    def reset(self):
        """Forget the current turn and start a new stream."""
        self.stream = self.detector.stream()
        self.chunks: List[np.ndarray] = []
        self.sample_rate: Optional[int] = None
        self.probs: List[np.ndarray] = []

    def warmup(self):
        """Run one second of silence through the model, then reset."""
        fs = self.detector.fs
        self.stream.accept(np.zeros(fs, dtype=np.float32), fs)
        self.reset()

    @property
    def algorithmic_delay_s(self) -> float:
        """The delay of the frame decision, without compute time.

        A frame's probability needs its whole STFT window (n_fft samples),
        and the centred median filter needs median // 2 frames after it.
        """
        d = self.detector
        return d.n_fft / d.fs + (d.median // 2) * d.frame_shift

    @property
    def endpoint_delay_s(self) -> float:
        """The time from the end of speech to the end of the turn."""
        return (
            self.algorithmic_delay_s + self.hangover_frames * self.detector.frame_shift
        )

    def decisions(self, probs: np.ndarray) -> np.ndarray:
        """Return the 0/1 decision of each frame whose median window is complete.

        The centred median filter of frame i reads frames up to
        i + median // 2, so the last median // 2 frames have no final decision
        yet and are not returned.
        """
        from scipy.signal import medfilt

        d = self.detector
        active = (probs > d.threshold).astype(np.float64)
        if d.median > 1 and len(active) > 0:
            active = medfilt(active, d.median)
        return active[: max(0, len(active) - d.median // 2)].astype(np.int64)

    @staticmethod
    def find_endpoint(decisions: np.ndarray, hangover_frames: int) -> Optional[int]:
        """Return the first silent frame after speech that starts a long silence.

        The silence must last hangover_frames frames. This rule decides from
        the frames received so far, so it can be used while audio arrives.
        A pause inside a sentence that lasts hangover_frames also ends the
        turn. Returns None when no such silence has been seen.
        """
        started, run = False, 0
        for i, v in enumerate(decisions):
            if v > 0:
                started, run = True, 0
            else:
                run += 1
                if started and run >= hangover_frames:
                    return i - run + 1
        return None

    def forward(
        self,
        speech: np.ndarray,
        sample_rate: int,
        binary: bool = False,
    ) -> Optional[np.ndarray]:
        """Add a chunk of audio; return the user's turn when it has ended.

        Args:
            speech (np.ndarray):
                The next chunk of audio, int16 PCM or float in [-1, 1].
            sample_rate (int):
                The sampling rate of the chunk.
            binary (bool, optional):
                If True, return the turn as the chunks were given, at
                sample_rate. Otherwise return float audio at target_sr.
                Defaults to False.

        Returns:
            Optional[np.ndarray]:
                The turn, from pad_start_ms before the first speech frame to
                pad_end_ms after the endpoint, or None while the turn goes on.
        """
        from espnet2.bin.diar_inference import as_mono_float

        if self.sample_rate is not None and sample_rate != self.sample_rate:
            raise ValueError(
                f"the sample rate changed within a turn: "
                f"{self.sample_rate} then {sample_rate}"
            )
        self.sample_rate = sample_rate
        speech = np.asarray(speech)
        self.chunks.append(speech)
        self.probs.append(self.stream.accept(as_mono_float(speech), sample_rate))
        probs = np.concatenate(self.probs)
        decisions = self.decisions(probs)
        end = self.find_endpoint(decisions, self.hangover_frames)
        if end is None:
            return None

        shift = self.detector.frame_shift
        start = int(np.flatnonzero(decisions)[0])
        if (end - start) * shift < self.min_speech_s:
            self.reset()
            return None
        audio = np.concatenate(self.chunks)
        start_s = max(0.0, start * shift - self.pad_start_s)
        end_s = min(len(audio) / sample_rate, end * shift + self.pad_end_s)
        self.last_turn = dict(
            probs=probs,
            decisions=decisions,
            frame_shift=shift,
            speech_start_s=start * shift,
            endpoint_s=end * shift,
            start_s=start_s,
            end_s=end_s,
        )
        turn = audio[int(start_s * sample_rate) : int(end_s * sample_rate)]
        self.reset()
        if binary:
            return turn
        return librosa.resample(
            as_mono_float(turn), orig_sr=sample_rate, target_sr=self.target_sr
        )
