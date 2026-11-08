"""Check how far ahead a sequence model reads its input.

A model with look-ahead L computes output frame i from input positions up to
``last_input(i)`` (for example ``i + L`` when input and output have the same
frame rate, or the last sample of frame i + L for a model on waveforms). This
checks that bound empirically, for any model given as a function from an input
tensor to an output tensor or array, both with time on ``time_dim``.
"""

from typing import Callable, Iterable, NamedTuple

import numpy as np
import torch


class LookAheadCheck(NamedTuple):
    """Largest output changes seen by check_look_ahead.

    unread: largest change of an output frame whose last_input precedes the
        changed input (0 when the model reads no further than last_input).
    read: largest change of the first output frames that may read the changed
        input (above 0 when the model does read up to last_input).
    """

    unread: float
    read: float


def check_look_ahead(
    forward: Callable[[torch.Tensor], object],
    inputs: torch.Tensor,
    last_input: Callable[[int], int],
    cut_points: Iterable[int],
    time_dim: int = 1,
    atol: float = 0.0,
    tight: bool = True,
    tight_window: int = 1,
    seed: int = 0,
) -> LookAheadCheck:
    """Check that output frame i of forward reads no input after last_input(i).

    For each cut point c, the inputs at positions c, c + 1, ... on time_dim are
    replaced with random values, ten times the inputs' standard deviation. Every
    output frame i with last_input(i) < c must change by at most atol. With
    tight=True, one of the first tight_window output frames with
    last_input(i) >= c must change by more than atol, which shows that the bound
    is reached and not only respected (use tight_window > 1 when the first such
    frame reads input c with a tiny weight, e.g. at the edge of a window).

    Args:
        forward: Maps an input like ``inputs`` to the output (tensor or array).
        inputs: Input with time on time_dim.
        last_input: Output frame index -> last input position it may read.
        cut_points: Input positions from which the input is changed.
        time_dim: The time dimension of the input and the output.
        atol: Largest change allowed for the frames that must not change.
        tight: Also require a change of the frames that may read the cut.
        tight_window: Number of such frames to look at for that change.
        seed: Seed of the replacement values.

    Returns:
        The largest changes seen over all cut points.

    Raises:
        AssertionError: An output frame changes although last_input(i) < c, or
            (tight) none of the frames that may read the cut changes.
    """
    generator = torch.Generator().manual_seed(seed)
    scale = 10.0 * float(inputs.float().std()) + 1e-3

    def run(x):
        with torch.no_grad():
            y = forward(x)
        y = (
            y.detach().cpu()
            if isinstance(y, torch.Tensor)
            else torch.as_tensor(np.asarray(y))
        )
        return y.double().movedim(time_dim, 0)

    reference = run(inputs)
    num_frames = reference.size(0)
    unread_max, read_max = 0.0, 0.0
    for cut in cut_points:
        if not 0 < cut < inputs.size(time_dim):
            raise ValueError(f"cut point {cut} is outside the input")
        changed = inputs.clone()
        tail = changed.narrow(time_dim, cut, inputs.size(time_dim) - cut)
        noise = torch.randn(tail.shape, generator=generator, dtype=torch.float32)
        tail.copy_(noise.to(tail) * scale)
        output = run(changed)
        assert output.shape == reference.shape, (output.shape, reference.shape)
        diff = (output - reference).abs().reshape(num_frames, -1).amax(dim=1)
        unread = [i for i in range(num_frames) if last_input(i) < cut]
        first_read = len(unread)
        if unread:
            worst = float(diff[unread].max())
            if worst > atol:
                frame = unread[int(diff[unread].argmax())]
                raise AssertionError(
                    f"output frame {frame} changes by {worst:.3g} when the input "
                    f"from position {cut} on changes, but may read only up to "
                    f"position {last_input(frame)}"
                )
            unread_max = max(unread_max, worst)
        window = diff[first_read : first_read + tight_window]
        if len(window):
            changed_max = float(window.max())
            read_max = max(read_max, changed_max)
            if tight and changed_max <= atol:
                raise AssertionError(
                    f"output frames {first_read}..{first_read + len(window) - 1}, "
                    f"which may read input position {cut}, do not change: the "
                    "model reads less far ahead than last_input"
                )
    return LookAheadCheck(unread=unread_max, read=read_max)


def frame_last_sample(
    hop_length: int, win_length: int, look_ahead: int = 0, center: bool = False
) -> Callable[[int], int]:
    """last_input for a model on waveforms whose frames come from an STFT.

    Frame i covers samples [i * hop, i * hop + win) (shifted left by win // 2
    with center=True); with look-ahead L frames it may read the last sample of
    frame i + L.
    """
    shift = win_length // 2 if center else 0

    def last(i: int) -> int:
        return (i + look_ahead) * hop_length + win_length - 1 - shift

    return last


__all__ = ["LookAheadCheck", "check_look_ahead", "frame_last_sample"]
