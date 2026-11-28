from typing import List, Optional, Tuple

import librosa
import numpy as np

from espnet2.sds.vad.abs_frame_vad import AbsFrameVAD
from espnet2.sds.vad.abs_vad import AbsVAD


class TurnDetector(AbsVAD):
    """The user's turns from the frame scores of any AbsFrameVAD.

    Every SDS VAD built on this class shares one end-of-turn rule:

    1. A frame is speech when its score exceeds threshold.
    2. A centred median filter of median frames smooths the 0/1 decisions.
       It reads median // 2 frames after each frame, so with a median above
       1 the decision is not strictly causal; 1 makes it causal.
    3. A turn ends at the first silence after speech that lasts hangover_ms
       (_find_endpoint), or when its speech reaches max_turn_ms.
    4. A turn with less than min_speech_ms of speech is dropped.
    5. The returned audio runs from pad_start_ms before the turn's first
       speech frame to pad_end_ms after its endpoint. When one chunk ends
       several turns, they are returned as one (AbsVAD).

    algorithmic_delay_s is the delay of a frame's decision, and
    endpoint_delay_s the time from a turn's endpoint to its return,
    both without compute time. last_turn holds the frame scores and times of
    the turn last returned, for a caller that plots them.
    """

    def __init__(
        self,
        frame_vad: AbsFrameVAD,
        threshold: float = 0.5,
        median: int = 1,
        hangover_ms: float = 500.0,
        pad_start_ms: float = 300.0,
        pad_end_ms: float = 300.0,
        min_speech_ms: float = 200.0,
        max_turn_ms: float = 30000.0,
        target_sr: int = 16000,
    ):
        """Set the end-of-turn rule on a frame scorer.

        Args:
            frame_vad (AbsFrameVAD):
                The frame scorer.
            threshold (float, optional):
                A frame is speech when its score exceeds this value.
                Defaults to 0.5.
            median (int, optional):
                The length in frames of the centred median filter, a positive
                odd number. Defaults to 1, no filter.
            hangover_ms (float, optional):
                The silence after speech that ends the user's turn.
                Defaults to 500 ms.
            pad_start_ms (float, optional):
                The audio kept before the first speech frame.
                Defaults to 300 ms.
            pad_end_ms (float, optional):
                The audio kept after the end of speech. It must not exceed
                endpoint_delay_s, so that this audio is already in when the
                turn is returned. Defaults to 300 ms.
            min_speech_ms (float, optional):
                A turn with less speech than this is dropped.
                Defaults to 200 ms.
            max_turn_ms (float, optional):
                A turn whose speech reaches this length is split there: it is
                returned, and the speech that goes on starts the next turn.
                Without the cap, speech or noise that never pauses for
                hangover_ms would keep one turn, and its audio, growing for as
                long as the stream runs. Defaults to 30000 ms, the window OWSM
                recognizes at once.
            target_sr (int, optional):
                The sampling rate of the float audio forward() returns.
                Defaults to 16000 Hz.
        """
        if max_turn_ms < min_speech_ms:
            raise ValueError(
                f"max_turn_ms ({max_turn_ms}) is below min_speech_ms "
                f"({min_speech_ms})"
            )
        if median < 1 or median % 2 == 0:
            raise ValueError(f"median must be a positive odd number: {median}")
        super().__init__()
        self.frame_vad = frame_vad
        self.threshold = threshold
        self.median = median
        shift = frame_vad.frame_shift
        self.hangover_frames = max(1, round(hangover_ms / 1000 / shift))
        self.pad_start_s = pad_start_ms / 1000
        self.pad_end_s = pad_end_ms / 1000
        # Both bounds in frames, rounded the same way, so that a forced turn of
        # max_turn_frames always passes the minimum (max_turn_ms >=
        # min_speech_ms is checked above).
        self.min_speech_frames = max(1, round(min_speech_ms / 1000 / shift))
        self.max_turn_frames = max(1, round(max_turn_ms / 1000 / shift))
        self.target_sr = target_sr
        # The audio after the endpoint that is in when the turn is returned;
        # hangover_ms rounds to whole frames, so compare with the frames.
        if self.pad_end_s > self.endpoint_delay_s + 1e-9:
            raise ValueError(
                f"pad_end_ms ({pad_end_ms}) must not exceed the "
                f"{self.endpoint_delay_s * 1000:.0f} ms of audio after the "
                "endpoint that is in when the turn is returned "
                "(endpoint_delay_s)"
            )
        self.last_turn: Optional[dict] = None
        self.reset()

    def reset(self):
        """Forget all audio and start a new stream."""
        self.frame_vad.reset()
        self.sample_rate: Optional[int] = None
        # The audio kept, as given (raw) and as float in [-1, 1] (audio), with
        # the stream's sample index of their first sample.
        self.raw: Optional[np.ndarray] = None
        self.audio = np.zeros(0, dtype=np.float32)
        self.audio_start = 0
        # The frame scores kept, with the stream's frame index of the first
        # one.
        self.probs = np.zeros(0, dtype=np.float32)
        self.probs_start = 0
        # Frames before this one belong to a returned or dropped turn, or are
        # silence before any speech.
        self.consumed = 0

    def warmup(self):
        """Warm the frame scorer up, then reset."""
        self.frame_vad.warmup()
        self.reset()

    @property
    def algorithmic_delay_s(self) -> float:
        """The delay of the frame decision, without compute time.

        A frame's score needs window_s of audio from its start, and the
        centred median filter needs median // 2 frames after it.
        """
        f = self.frame_vad
        return f.window_s + (self.median // 2) * f.frame_shift

    @property
    def endpoint_delay_s(self) -> float:
        """The time from a turn's endpoint to forward() returning the turn.

        The endpoint is the start of the first silent frame. The turn is
        returned once the last of its hangover_frames silent frames, the
        (hangover_frames - 1)-th after it, has its decision.
        """
        return (
            self.algorithmic_delay_s
            + (self.hangover_frames - 1) * self.frame_vad.frame_shift
        )

    def _decisions(self, probs: np.ndarray) -> np.ndarray:
        """Return the 0/1 decision of each frame whose median window is complete.

        The centred median filter of frame i reads frames up to
        i + median // 2, so the last median // 2 frames have no final decision
        yet and are not returned.
        """
        from scipy.signal import medfilt

        active = (probs > self.threshold).astype(np.float64)
        if self.median > 1 and len(active) > 0:
            # Zero-pad both ends ourselves, as medfilt does, so that a stream
            # shorter than the filter gives no "kernel_size exceeds volume
            # extent" warning.
            half = self.median // 2
            padded = np.pad(active, half)
            active = medfilt(padded, self.median)[half : half + len(active)]
        return active[: max(0, len(active) - self.median // 2)].astype(np.int64)

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
        long recordings to an offline VAD (e.g. espnet detect) instead.

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
            [self.probs, self.frame_vad.accept(audio, sample_rate)]
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
        shift = self.frame_vad.frame_shift
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
        shift = self.frame_vad.frame_shift
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
        shift = self.frame_vad.frame_shift
        rate = self.sample_rate
        d = self._decisions(self.probs)
        # Silence after the last turn cannot start one: move past it, up to the
        # first speech frame or the last decided frame.
        speech = np.flatnonzero(d[self.consumed - self.probs_start :])
        if len(speech):
            self.consumed += int(speech[0])
        else:
            self.consumed = self.probs_start + len(d)
        keep_f = max(self.probs_start, self.consumed - self.median // 2)
        self.probs = self.probs[keep_f - self.probs_start :]
        self.probs_start = keep_f
        keep_s = max(
            self.audio_start, int((self.consumed * shift - self.pad_start_s) * rate)
        )
        self.raw = self.raw[keep_s - self.audio_start :]
        self.audio = self.audio[keep_s - self.audio_start :]
        self.audio_start = keep_s
