import pytest
import torch

from espnet2.legacy.nets.batch_beam_search_online import BatchBeamSearchOnline
from espnet2.legacy.nets.scorers.length_bonus import LengthBonus

VOCAB_SIZE = 5
SOS = EOS = VOCAB_SIZE - 1


def build_beam_search(cls=BatchBeamSearchOnline, **kwargs):
    """Build a scorer-light BatchBeamSearchOnline.

    `length_bonus` is the only scorer so that the search runs without a decoder,
    which keeps the test focused on `forward()`'s control flow.
    """
    return cls(
        beam_size=2,
        vocab_size=VOCAB_SIZE,
        weights={"length_bonus": 1.0},
        scorers={"length_bonus": LengthBonus(VOCAB_SIZE)},
        sos=SOS,
        eos=EOS,
        token_list=[str(i) for i in range(VOCAB_SIZE)],
        # the block_size == 0 branch builds its initial hypothesis straight from
        # hyp_primer and hard-codes its length to 2, so the primer needs 2 tokens
        hyp_primer=[SOS, SOS],
        time_sync=False,
        **kwargs,
    )


def test_batch_beam_search_online_block_size_zero():
    """Decoding with block_size == 0 and time_sync off must reach the end.

    Regression test for the `process_one_block()` call in that branch being one
    argument short, which made every call raise TypeError before any search ran.
    """
    beam = build_beam_search(block_size=0)
    x = torch.randn(6, 4)
    nbest = beam(x, maxlenratio=0.0, minlenratio=0.0, is_final=True)
    assert len(nbest) > 0
    for hyp in nbest:
        assert hyp.yseq[:2].tolist() == [SOS, SOS]


@pytest.mark.parametrize("block_size", [0, 4])
def test_batch_beam_search_online_process_one_block_args(block_size):
    """`minlen` and `maxlenratio` must reach the slots they are named for.

    Both `if`/`else` pairs in `forward()` are covered: passing `maxlenratio` in
    the `minlen` slot raises TypeError, but swapping two adjacent arguments
    would silently change the length constraints instead, so the bound values
    are checked and not just the arity.
    """
    calls = []

    class _RecordingBeamSearch(BatchBeamSearchOnline):
        def process_one_block(self, h, is_final, maxlen, minlen, maxlenratio):
            calls.append(dict(minlen=minlen, maxlenratio=maxlenratio))
            return []

    beam = build_beam_search(
        _RecordingBeamSearch, block_size=block_size, hop_size=2, look_ahead=1
    )
    x = torch.randn(10, 4)
    beam(x, maxlenratio=0.5, minlenratio=0.2, is_final=True)

    assert len(calls) > 0
    for call in calls:
        assert call["minlen"] == int(0.2 * x.shape[0])
        assert call["maxlenratio"] == 0.5
