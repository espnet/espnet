"""Word spans of CTC segmentation: segments vs the frames of the tokens."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

cs = pytest.importorskip("ctc_segmentation")

from ctc_batch import token_word_spans  # noqa: E402

FRAME = 0.08


def _align(spikes, words, frames=40):
    """CTC segmentation of synthetic posteriors.

    Blank everywhere except one spike frame per token; spikes = [(frame, token id)].
    """
    config = cs.CtcSegmentationParameters(char_list=["<blank>", "a", "b", "c", "d"])
    config.index_duration = FRAME
    config.blank = 0
    gt, ubi = cs.prepare_token_list(config, [np.array(w) for w in words])
    lpz = np.full((frames, 5), -20.0)
    lpz[:, 0] = -0.01
    for t, k in spikes:
        lpz[t, :] = -20.0
        lpz[t, k] = -0.01
    timings, probs, _ = cs.ctc_segmentation(config, lpz, gt)
    segments = cs.determine_utterance_segments(
        config, ubi, probs, timings, ["w"] * len(words)
    )
    return timings, ubi, segments, config


def test_token_spans_cover_the_frames_of_each_word():
    # word 0 = tokens a b at frames 5 and 7; word 1 = c at 20; word 2 = d a at 30, 32
    spikes = [(5, 1), (7, 2), (20, 3), (30, 4), (32, 1)]
    timings, ubi, _, config = _align(spikes, [[1, 2], [3], [4, 1]])
    spans = token_word_spans(timings, ubi, config.index_duration_in_seconds)
    expected = [(5, 8), (20, 21), (30, 33)]  # first token frame .. last token frame + 1
    for (s, e), (fs, fe) in zip(spans, expected):
        assert s == pytest.approx(fs * FRAME)
        assert e == pytest.approx(fe * FRAME)


def test_segmentation_starts_words_early_and_closes_pauses():
    """Segments start up to 0.5 s before the first token.

    determine_utterance_segments places them so; this made the word labels of
    v2's stage 4 start early.
    """
    spikes = [(5, 1), (7, 2), (20, 3), (30, 4), (32, 1)]
    timings, ubi, segments, config = _align(spikes, [[1, 2], [3], [4, 1]])
    spans = token_word_spans(timings, ubi, config.index_duration_in_seconds)
    word1_start = segments[1][0]
    assert word1_start == pytest.approx(20 * FRAME - 0.5)
    assert spans[1][0] - word1_start == pytest.approx(0.5)
    # the token spans keep the pause between words 0 and 1 (0.64 s .. 1.60 s)
    assert spans[1][0] - spans[0][1] == pytest.approx(12 * FRAME)


def test_a_word_without_tokens_is_an_error():
    with pytest.raises(ValueError):
        token_word_spans(np.zeros(4), [1, 2], FRAME)


def _emission(frames, segments, vocab=5):
    """Log posteriors: blank everywhere except token k on frames [s, e)."""
    import torch

    lpz = torch.full((frames, vocab), -20.0)
    lpz[:, 0] = -0.01
    for s, e, k in segments:
        lpz[s:e, :] = -20.0
        lpz[s:e, k] = -0.01
    return torch.log_softmax(lpz, dim=-1)


def test_forced_align_word_spans_cover_the_token_frames():
    pytest.importorskip("torchaudio")
    from ctc_batch import forced_align_word_spans

    # word 0 = a (frames 3-5) b (6-7); word 1 = c (15-18); word 2 = d (25) a (27-28)
    lpz = _emission(40, [(3, 6, 1), (6, 8, 2), (15, 19, 3), (25, 26, 4), (27, 29, 1)])
    spans = forced_align_word_spans(lpz, [[1, 2], [3], [4, 1]], 0, FRAME)
    expected = [(3, 8), (15, 19), (25, 29)]
    for (s, e, score), (fs, fe) in zip(spans, expected):
        assert s == pytest.approx(fs * FRAME)
        assert e == pytest.approx(fe * FRAME)
        assert -0.1 < score <= 0.0  # mean log posterior of the token frames


def test_forced_align_needs_frames_for_tokens_and_repeats():
    pytest.importorskip("torchaudio")
    from ctc_batch import TooFewFrames, forced_align_word_spans

    lpz = _emission(3, [(0, 1, 1), (2, 3, 1)])
    # a a needs 2 tokens + 1 repeat = 3 frames: fits
    forced_align_word_spans(lpz, [[1, 1]], 0, FRAME)
    with pytest.raises(TooFewFrames):
        forced_align_word_spans(lpz, [[1, 1], [2]], 0, FRAME)


def test_word_token_ids_skip_separators():
    from types import SimpleNamespace

    from ctc_batch import word_token_ids

    gt = np.array([[-1], [0], [1], [1], [2], [0], [2], [0]])
    task = SimpleNamespace(
        ground_truth_mat=gt,
        utt_begin_indices=[1, 5, 7],
        config=SimpleNamespace(blank=0),
    )
    assert word_token_ids(task) == [[1, 1, 2], [2]]
