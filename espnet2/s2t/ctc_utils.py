"""Frame counts for buffered long-form CTC inference."""

import math
from typing import Tuple


def buffered_frame_counts(
    frames_per_sec: float, speech_length: float, context_len_in_secs: float
) -> Tuple[int, int]:
    """Return buffer and context lengths on the encoder's frame grid.

    Both durations must cover whole frames. Truncating a fractional frame
    would make the audio step differ from the number of retained posteriors,
    accumulating a timing error at every buffer join.
    """
    counts = []
    for name, seconds in (
        ("speech_length", speech_length),
        ("context_len_in_secs", context_len_in_secs),
    ):
        frames = frames_per_sec * seconds
        if not math.isfinite(frames) or not math.isclose(
            frames, round(frames), rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                f"{name} must correspond to a whole number of encoder frames "
                f"at {frames_per_sec} frames per second; got {seconds} seconds"
            )
        counts.append(round(frames))
    buffer_frames, context_frames = counts
    if context_frames < 0 or buffer_frames <= 2 * context_frames:
        raise ValueError(
            "context_len_in_secs must be non-negative and less than "
            "half of speech_length"
        )
    return buffer_frames, context_frames
