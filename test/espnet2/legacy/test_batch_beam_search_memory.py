"""How much memory the beam search holds, and how that scales.

Beam search is easy to make quietly expensive: a state that should be dropped
gets carried, or a slice of a batch-sized tensor keeps the whole batch alive.
Neither shows up as a wrong hypothesis, so nothing else in the test suite
notices until a real decode runs out of memory.

These tests measure the bytes the search actually holds -- the running
hypotheses plus the finished ones -- and assert the shape of the growth rather
than an absolute number, so they mean the same thing on any machine and need
no accelerator.
"""

import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.legacy.nets.batch_beam_search import BatchBeamSearch
from espnet2.legacy.nets.scorers.ctc import CTCPrefixScorer
from espnet2.legacy.nets.scorers.length_bonus import LengthBonus

VOCAB, DIM, FRAMES = 50, 16, 80


def _held_bytes(obj, seen):
    """Sum the distinct storages reachable from `obj`.

    Storages rather than tensors, because a view costs whatever it is a view
    of, and deduplicated, because hypotheses legitimately share one.
    """
    total, stack = 0, [obj]
    while stack:
        item = stack.pop()
        if torch.is_tensor(item):
            storage = item.untyped_storage()
            if storage.data_ptr() not in seen:
                seen.add(storage.data_ptr())
                total += storage.nbytes()
        elif isinstance(item, (tuple, list)):
            stack.extend(item)
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif hasattr(item, "_fields"):  # BatchHypothesis / Hypothesis
            stack.extend(item)
    return total


class _MemoryProbe(BatchBeamSearch):
    """Record the high-water mark of what the search holds."""

    def post_process(self, i, maxlen, minlen, maxlenratio, running_hyps, ended_hyps):
        out = super().post_process(
            i, maxlen, minlen, maxlenratio, running_hyps, ended_hyps
        )
        seen = set()
        held = _held_bytes(out, seen) + _held_bytes(ended_hyps, seen)
        self.peak = max(getattr(self, "peak", 0), held)
        return out


def _build(beam):
    torch.manual_seed(0)
    decoder = TransformerDecoder(
        vocab_size=VOCAB,
        encoder_output_size=DIM,
        num_blocks=2,
        attention_heads=2,
        linear_units=16,
    ).eval()
    ctc = CTC(odim=VOCAB, encoder_output_size=DIM).eval()
    return _MemoryProbe(
        vocab_size=VOCAB,
        weights={"decoder": 0.7, "ctc": 0.3, "length_bonus": 0.1},
        scorers={
            "decoder": decoder,
            "ctc": CTCPrefixScorer(ctc=ctc, eos=VOCAB - 1),
            "length_bonus": LengthBonus(VOCAB),
        },
        sos=VOCAB - 1,
        eos=VOCAB - 1,
        beam_size=beam,
        pre_beam_score_key="full",
    ).eval()


def _peak(n_utt, steps, beam=3):
    """Decode `n_utt` utterances for exactly `steps` tokens and report the peak."""
    search = _build(beam)
    x = torch.randn(n_utt, FRAMES, DIM)
    x_lengths = torch.full((n_utt,), FRAMES, dtype=torch.long)
    with torch.no_grad():
        # a constant max output length, so every run takes the same steps
        search(x=x, x_lengths=x_lengths, maxlenratio=-steps, minlenratio=0.0)
    return search.peak


def test_memory_is_linear_in_decoding_steps():
    """Twice the output length must not cost much more than twice the memory.

    The decoder's key/value cache grows with the output length, so linear
    growth is expected. Anything that accumulates per-step state instead of
    replacing it turns this quadratic.
    """
    peaks = {steps: _peak(4, steps) for steps in (8, 16, 32)}
    for small, large in ((8, 16), (16, 32)):
        ratio = peaks[large] / peaks[small]
        assert ratio < 2.5, (
            f"doubling the output length from {small} to {large} steps "
            f"multiplied the held memory by {ratio:.2f} "
            f"({peaks[small]} -> {peaks[large]} bytes); "
            "quadratic growth means per-step state is being accumulated"
        )


def test_memory_is_linear_in_batch_size():
    """Twice the utterances must not cost much more than twice the memory."""
    peaks = {n: _peak(n, 8) for n in (2, 4, 8)}
    for small, large in ((2, 4), (4, 8)):
        ratio = peaks[large] / peaks[small]
        assert ratio < 2.5, (
            f"going from {small} to {large} utterances multiplied the held "
            f"memory by {ratio:.2f} ({peaks[small]} -> {peaks[large]} bytes)"
        )


def test_finished_hypotheses_do_not_pin_the_running_batch():
    """An ended hypothesis must not keep a batch-sized storage alive.

    Scorer states are slices of `(n_utt * beam, ...)` tensors. A hypothesis
    that ended is held until the whole batch finishes, so if its states stay
    views, each one pins a storage `n_utt * beam` times larger than it needs.
    """
    search = _build(beam=3)
    n_utt = 6
    x = torch.randn(n_utt, FRAMES, DIM)
    x_lengths = torch.full((n_utt,), FRAMES, dtype=torch.long)
    with torch.no_grad():
        results = search(x=x, x_lengths=x_lengths, maxlenratio=-8, minlenratio=0.0)

    checked = 0
    for nbest in results:
        for hyp in nbest:
            seen = set()
            for item in (hyp.yseq, hyp.states, hyp.hs):
                stack = [item]
                while stack:
                    obj = stack.pop()
                    if torch.is_tensor(obj):
                        needed = obj.numel() * obj.element_size()
                        alive = obj.untyped_storage().nbytes()
                        assert alive == needed, (
                            f"a finished hypothesis keeps {alive} bytes alive "
                            f"for a {needed}-byte tensor "
                            f"({alive / max(needed, 1):.0f}x)"
                        )
                        checked += 1
                    elif isinstance(obj, (tuple, list)):
                        stack.extend(obj)
                    elif isinstance(obj, dict):
                        stack.extend(obj.values())
            del seen
    assert checked > 0, "no tensors were inspected"


def test_scorers_are_called_once_per_step_over_the_whole_grid():
    """Every scorer must see all `n_utt * beam` hypotheses in one call.

    This is the property the whole design rests on (Seki et al., "Vectorized
    Beam Search for CTC-Attention-Based Speech Recognition", Interspeech 2019):
    a decoding step is one decoder forward over the full grid, not a loop over
    utterances or over the beam. Losing it would keep the results correct while
    quietly destroying the speedup, which no other test would notice.
    """
    from espnet2.asr.decoder.transformer_decoder import BaseTransformerDecoder
    from espnet2.legacy.nets.ctc_prefix_score import CTCPrefixScoreTH

    n_utt, beam, steps = 4, 3, 6
    seen = {"decoder": [], "ctc": []}

    original_decoder = BaseTransformerDecoder.forward_one_step
    original_ctc = CTCPrefixScoreTH.__call__

    def spy_decoder(self, tgt, tgt_mask, memory, memory_mask=None, **kwargs):
        seen["decoder"].append((tgt.shape[0], memory.shape[0]))
        return original_decoder(self, tgt, tgt_mask, memory, memory_mask, **kwargs)

    def spy_ctc(self, y, state, scoring_ids=None, att_w=None):
        seen["ctc"].append((len(y), self.batch))
        return original_ctc(self, y, state, scoring_ids, att_w)

    BaseTransformerDecoder.forward_one_step = spy_decoder
    CTCPrefixScoreTH.__call__ = spy_ctc
    try:
        search = _build(beam=beam)
        x = torch.randn(n_utt, FRAMES, DIM)
        with torch.no_grad():
            search(
                x=x,
                x_lengths=torch.full((n_utt,), FRAMES, dtype=torch.long),
                maxlenratio=-steps,
                minlenratio=0.0,
            )
    finally:
        BaseTransformerDecoder.forward_one_step = original_decoder
        CTCPrefixScoreTH.__call__ = original_ctc

    grid = n_utt * beam
    assert len(seen["decoder"]) == steps, seen["decoder"]
    assert {t for t, _ in seen["decoder"]} == {grid}, seen["decoder"]
    # the decoder attends over the beam itself, so it is handed one encoder
    # row per utterance rather than one per hypothesis
    assert {m for _, m in seen["decoder"]} == {n_utt}, seen["decoder"]

    assert len(seen["ctc"]) == steps, seen["ctc"]
    # `CTCPrefixScoreTH` scores the whole grid at once and recovers the
    # utterance of a hypothesis as `flat_index // (n_bh // self.batch)`
    assert {n for n, _ in seen["ctc"]} == {grid}, seen["ctc"]
    assert {b for _, b in seen["ctc"]} == {n_utt}, seen["ctc"]
