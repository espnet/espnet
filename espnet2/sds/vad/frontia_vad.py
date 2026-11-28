from typing import List, Optional, Tuple

import librosa
import numpy as np
from typeguard import typechecked

from espnet2.sds.vad.abs_vad import AbsVAD


class FrontiaVADModel(AbsVAD):
    """frontia-vad, a neural VAD, behind the SDS VAD interface.

    The model scores each frame causally through StreamingVAD
    (espnet2.bin.diar_inference_streaming), so forward() takes the audio
    chunk by chunk, as WebrtcVADModel does.

    The model, threshold and median default to ESPnet's online VAD,
    VAD_MODELS["online"] and VAD_RULES["online"] in espnet2.bin.diar_inference.
    With a median above 1 the decision is not strictly causal: the centred
    median filter waits median // 2 frames after each frame (96 ms for 25).
    A turn ends endpoint_delay_s after its speech, the median's delay plus
    hangover_ms. Pass median=1 for a decision that reads no future frame.
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
                A frame is speech when its probability exceeds this value.
                Defaults to the threshold of VAD_RULES["online"].
            median (int, optional):
                The length in frames of the centred median filter applied to
                the 0/1 decisions. Defaults to the median of
                VAD_RULES["online"]. The filter reads (median - 1) / 2 frames
                after each frame, so with a median above 1 every decision
                comes that much after the audio and the detection is not
                strictly causal, although the model reads no future audio.
                1 makes it strictly causal. algorithmic_delay_s reports the
                delay.
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
            max_turn_ms (float, optional):
                A turn whose speech reaches this length is split there: it is
                returned, and the speech that goes on starts the next turn.
                (WebrtcVADModel's max_speech_ms instead drops a longer
                segment.) Without the cap, speech or noise that never pauses for
                hangover_ms would keep one turn, and its audio, growing for as
                long as the stream runs. Defaults to 30000 ms, the window OWSM
                recognizes at once.
            target_sr (int, optional):
                The sampling rate of the float audio forward() returns.
                Defaults to 16000 Hz.
            device (str, optional):
                The device the model runs on. Defaults to "cuda", as the
                other SDS modules do.
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
        self._setup(
            detector,
            hangover_ms,
            pad_start_ms,
            pad_end_ms,
            min_speech_ms,
            max_turn_ms,
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
        max_turn_ms: float = 30000.0,
        target_sr: int = 16000,
    ) -> "FrontiaVADModel":
        """Wrap a loaded VoiceActivityDetector instead of loading a tag."""
        self = cls.__new__(cls)
        self._setup(
            detector,
            hangover_ms,
            pad_start_ms,
            pad_end_ms,
            min_speech_ms,
            max_turn_ms,
            target_sr,
        )
        return self

    def _setup(
        self,
        detector,
        hangover_ms,
        pad_start_ms,
        pad_end_ms,
        min_speech_ms,
        max_turn_ms,
        target_sr,
    ):
        if max_turn_ms < min_speech_ms:
            raise ValueError(
                f"max_turn_ms ({max_turn_ms}) is below min_speech_ms "
                f"({min_speech_ms})"
            )
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
        # Both bounds in frames, rounded the same way, so that a forced turn of
        # max_turn_frames always passes the minimum (max_turn_ms >=
        # min_speech_ms is checked above).
        self.min_speech_frames = max(
            1, round(min_speech_ms / 1000 / detector.frame_shift)
        )
        self.max_turn_frames = max(1, round(max_turn_ms / 1000 / detector.frame_shift))
        self.target_sr = target_sr
        # Hold the last turn's frame scores, for a caller that plots them.
        self.last_turn: Optional[dict] = None
        self.reset()

    def reset(self):
        """Forget all audio and start a new stream."""
        # This class keeps the probabilities it needs itself, so the stream
        # keeps none.
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
        self.sample_rate: Optional[int] = None
        # The audio kept, as given (raw) and as float in [-1, 1] (audio), with
        # the stream's sample index of their first sample.
        self.raw: Optional[np.ndarray] = None
        self.audio = np.zeros(0, dtype=np.float32)
        self.audio_start = 0
        # The frame probabilities kept, with the stream's frame index of the
        # first one.
        self.probs = np.zeros(0, dtype=np.float32)
        self.probs_start = 0
        # Frames before this one belong to a returned or dropped turn, or are
        # silence before any speech.
        self.consumed = 0

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

    def _decisions(self, probs: np.ndarray) -> np.ndarray:
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
    def _find_endpoint(decisions: np.ndarray, hangover_frames: int) -> Optional[int]:
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
        """Add a chunk of audio; return a user's turn once one has ended.

        One stream runs across turns, so audio after a turn's endpoint starts
        the next turn. When a chunk ends more than one turn, this call returns
        them as one: the audio from the first one's start to the last one's
        end, pauses included, since the dialogue system could answer none of
        them before this call returns. last_turn["turns"] lists them. So a
        returned turn can be longer than max_turn_ms when the chunk is; feed
        long recordings to VoiceActivityDetector.detect() instead.

        Args:
            speech (np.ndarray):
                The next chunk of audio, int16 PCM or float in [-1, 1]. It is
                copied, so the caller may reuse its buffer.
            sample_rate (int):
                The sampling rate of the chunk. It must not change until
                reset().
            binary (bool, optional):
                If True, return the turn as the chunks were given, at
                sample_rate. Otherwise return float audio at target_sr.
                Defaults to False.

        Returns:
            Optional[np.ndarray]:
                The turn, from pad_start_ms before its first speech frame to
                pad_end_ms after its endpoint, or None while no turn has ended.
        """
        from espnet2.bin.diar_inference import as_mono_float

        if self.sample_rate is not None and sample_rate != self.sample_rate:
            raise ValueError(
                f"the sample rate changed from {self.sample_rate} to "
                f"{sample_rate}; call reset() first"
            )
        self.sample_rate = sample_rate
        # Copy the chunk: a caller may overwrite its buffer after this call.
        raw = np.array(speech, copy=True)
        audio = as_mono_float(raw)
        raw = raw.reshape(-1)
        self.raw = raw if self.raw is None else np.concatenate([self.raw, raw])
        # Normalize each chunk before joining: joining int16 and float chunks
        # first would turn the PCM values into floats that as_mono_float no
        # longer scales.
        self.audio = np.concatenate([self.audio, audio])
        self.probs = np.concatenate(
            [self.probs, self.stream.accept(audio, sample_rate)]
        )
        turns = self._find_turns()
        result = self._join(turns) if turns else None
        # Trim after joining: the joined audio starts before consumed.
        self._trim()
        if result is None:
            return None
        raw_turn, turn = result
        if binary:
            return raw_turn
        return librosa.resample(turn, orig_sr=sample_rate, target_sr=self.target_sr)

    def _join(self, turns: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Return the audio of the turns as one turn, and set last_turn.

        The audio runs from the first turn's start to the last one's end,
        pauses included.
        """
        shift = self.detector.frame_shift
        rate = self.sample_rate
        first, last = turns[0], turns[-1]
        a = int(round(first["start_s"] * rate)) - self.audio_start
        b = int(round(last["end_s"] * rate)) - self.audio_start
        # Frame scores up to the last turn's end of hangover, for plotting.
        upto = last["end"] + self.hangover_frames
        self.last_turn = dict(
            probs=self.probs[:upto].copy(),
            decisions=self._decisions(self.probs)[:upto].copy(),
            probs_start_s=self.probs_start * shift,
            frame_shift=shift,
            speech_start_s=first["speech_start_s"],
            endpoint_s=last["endpoint_s"],
            # True when max_turn_ms, not a silence, ended the (last) turn
            forced=last["forced"],
            start_s=first["start_s"],
            end_s=last["end_s"],
            # each turn that ended in this call, without its frame scores
            turns=[{k: v for k, v in t.items() if k != "end"} for t in turns],
        )
        return self.raw[a:b].copy(), self.audio[a:b].copy()

    def _find_turns(self) -> List[dict]:
        """Return every turn that has ended in the frames received so far.

        Each turn is a dict of its times in seconds (speech_start_s,
        endpoint_s, start_s, end_s), forced, and end, the index of its
        endpoint in self.probs.
        """
        shift = self.detector.frame_shift
        rate = self.sample_rate
        turns: List[dict] = []
        while True:
            decisions = self._decisions(self.probs)
            first = self.consumed - self.probs_start
            d = decisions[first:]
            speech = np.flatnonzero(d)
            if len(speech) == 0:
                return turns
            start = int(speech[0])
            end = self._find_endpoint(d, self.hangover_frames)
            # A turn that reaches max_turn_ms of speech ends there.
            forced = (end is None or end - start > self.max_turn_frames) and (
                len(d) - start >= self.max_turn_frames
            )
            if forced:
                end = start + self.max_turn_frames
            elif end is None:
                return turns
            start += first
            end += first
            # stream frame indices of the first speech frame and the endpoint
            start_f, end_f = self.probs_start + start, self.probs_start + end
            last_s = (self.audio_start + len(self.audio)) / rate
            if forced and last_s < end_f * shift + self.pad_end_s:
                # A silence ends a turn hangover_ms after the endpoint, so its
                # end padding is already in; a forced turn ends while speech
                # goes on. Wait for its padding: the endpoint stays at the
                # first speech frame plus max_turn_frames.
                return turns
            self.consumed = end_f
            if end_f - start_f < self.min_speech_frames:
                continue
            turns.append(
                dict(
                    speech_start_s=start_f * shift,
                    endpoint_s=end_f * shift,
                    forced=forced,
                    start_s=max(
                        self.audio_start / rate, start_f * shift - self.pad_start_s
                    ),
                    end_s=min(last_s, end_f * shift + self.pad_end_s),
                    end=end,
                )
            )

    def _trim(self):
        """Drop the audio and frames that no later turn can use.

        Without this, the buffers grow for as long as the stream runs while
        nobody speaks. A later turn needs the frames from consumed on, the
        median // 2 frames before them for its median filter, and
        pad_start_ms of audio before its first speech frame.
        """
        shift = self.detector.frame_shift
        rate = self.sample_rate
        d = self._decisions(self.probs)
        # Silence after the last turn cannot start one: move past it, up to the
        # first speech frame or the last decided frame.
        speech = np.flatnonzero(d[self.consumed - self.probs_start :])
        if len(speech):
            self.consumed += int(speech[0])
        else:
            self.consumed = self.probs_start + len(d)
        keep_f = max(self.probs_start, self.consumed - self.detector.median // 2)
        self.probs = self.probs[keep_f - self.probs_start :]
        self.probs_start = keep_f
        keep_s = max(
            self.audio_start, int((self.consumed * shift - self.pad_start_s) * rate)
        )
        self.raw = self.raw[keep_s - self.audio_start :]
        self.audio = self.audio[keep_s - self.audio_start :]
        self.audio_start = keep_s
