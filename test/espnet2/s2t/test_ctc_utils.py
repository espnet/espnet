import pytest

from espnet2.s2t.ctc_utils import buffered_frame_counts


@pytest.mark.parametrize(
    "fps,window,context,expected",
    [
        (12.5, 30, 2, (375, 25)),
        (12.5, 30, 0.56, (375, 7)),
        (25, 30, 2, (750, 50)),
        (100 / 6, 30, 1.92, (500, 32)),
    ],
)
def test_frame_counts_allow_float_roundoff(fps, window, context, expected):
    """Integral frame durations need not multiply to exact binary integers."""
    assert buffered_frame_counts(fps, window, context) == expected


@pytest.mark.parametrize(
    "window,context", [(30, 0.5), (3, 0.8), (30, float("nan")), (30, float("inf"))]
)
def test_frame_counts_reject_fractional_or_nonfinite_durations(window, context):
    """Never silently quantize the audio stride and frame stride differently."""
    with pytest.raises(ValueError, match="whole number"):
        buffered_frame_counts(12.5, window, context)


@pytest.mark.parametrize("window,context", [(30, -0.8), (32, 16), (30, 16), (0, 0)])
def test_frame_counts_require_a_positive_chunk(window, context):
    """A nonpositive audio step cannot advance through the recording."""
    with pytest.raises(ValueError, match="half of speech_length"):
        buffered_frame_counts(12.5, window, context)
