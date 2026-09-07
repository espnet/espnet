"""Assert that single-utterance decoding still matches the pre-batching code.

`BatchBeamSearch` was rewritten to decode a batch of utterances, with a single
utterance as the `n_utt == 1` case. These tests pin that case against
`reference_batch_beam_search.py`, a frozen copy of the implementation it
replaced, so that the rewrite -- and any later change to it -- cannot quietly
alter what ESPnet decodes today.

The comparison is exact: same tokens, same scores, same n-best ordering.
"""

from test.espnet2.legacy.reference_batch_beam_search import (
    BatchBeamSearch as ReferenceBatchBeamSearch,
)
from test.espnet2.legacy.test_beam_search import prepare, transformer_args

import numpy
import pytest
import torch

from espnet2.legacy.nets.batch_beam_search import BatchBeamSearch
from espnet2.legacy.nets.scorers.ctc import CTCPrefixScorer
from espnet2.legacy.nets.scorers.length_bonus import LengthBonus
from espnet2.lm.seq_rnn_lm import SequentialRNNLM
from espnet2.lm.transformer_lm import TransformerLM


def _setup(args, ctc_weight, lm, bonus, dtype, normalize_length, return_hs=False):
    """Build one model and the scorer set shared by both implementations."""
    torch.manual_seed(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model, x, ilens, y, data, train_args = prepare(args, mtlalpha=ctc_weight)
    model.to(dtype=dtype)
    model.eval()
    token_list = train_args.token_list

    scorers = {"decoder": model.decoder, "length_bonus": LengthBonus(len(token_list))}
    weights = {
        "decoder": 1.0 - ctc_weight,
        "length_bonus": bonus,
        "ctc": ctc_weight,
        "lm": 0.0,
    }
    if ctc_weight != 0:
        scorers["ctc"] = CTCPrefixScorer(ctc=model.ctc, eos=model.eos)
    if lm == "transformer":
        scorers["lm"] = TransformerLM(
            len(token_list), unit=2, layer=1, embed_unit=2, dropout_rate=0.0
        )
        weights["lm"] = 0.5
    elif lm == "rnn":
        scorers["lm"] = SequentialRNNLM(len(token_list), unit=2, nlayers=1)
        weights["lm"] = 0.5
    for scorer in scorers.values():
        if isinstance(scorer, torch.nn.Module):
            scorer.to(dtype=dtype)
            scorer.eval()

    with torch.no_grad():
        enc, enc_lens = model.encode(
            x[0, : ilens[0]].unsqueeze(0).to(dtype=dtype),
            ilens[:1].to(dtype=torch.int32),
        )

    kwargs = dict(
        beam_size=3,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        token_list=token_list,
        sos=model.sos,
        eos=model.eos,
        pre_beam_score_key=None if ctc_weight == 1.0 else "full",
        normalize_length=normalize_length,
        return_hs=return_hs,
    )
    return enc[0, : enc_lens[0]], kwargs, dtype


def _assert_identical(reference, actual):
    """The two implementations must agree exactly, not just approximately."""
    assert len(actual) == len(reference), (len(actual), len(reference))
    for ref, act in zip(reference, actual):
        assert ref.yseq.tolist() == act.yseq.tolist()
        numpy.testing.assert_allclose(
            ref.score.cpu().float().numpy(),
            act.score.cpu().float().numpy(),
            rtol=0,
            atol=0,
        )
        assert sorted(ref.scores) == sorted(act.scores)
        for k in ref.scores:
            numpy.testing.assert_allclose(
                numpy.asarray(ref.scores[k], dtype=numpy.float64),
                numpy.asarray(act.scores[k], dtype=numpy.float64),
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize(
    "args, ctc_weight, lm, bonus, maxlenratio, minlenratio, normalize_length, dtype",
    [
        (args, ctc, lm, 0.1, maxr, minr, norm, dtype)
        for args in (transformer_args,)
        for ctc in (0.0, 0.5, 1.0)
        for lm in (None, "transformer", "rnn")
        # the length bounds matter most: they drive the <eos> forced at maxlen,
        # the minlen cutoff and the empty-n-best retry
        for maxr, minr in (
            (0.0, 0.0),
            (-5.0, 0.0),
            (0.5, 0.0),
            (0.0, 0.3),
            (0.0, -1.0),
            (0.0, -3.0),
        )
        for norm in (False, True)
        for dtype in (torch.float32, torch.float64)
    ],
)
def test_matches_reference_single_utterance(
    args, ctc_weight, lm, bonus, maxlenratio, minlenratio, normalize_length, dtype
):
    """Decoding one utterance must reproduce the pre-batching implementation."""
    enc, kwargs, dtype = _setup(args, ctc_weight, lm, bonus, dtype, normalize_length)

    reference = ReferenceBatchBeamSearch(**kwargs)
    reference.to(dtype=dtype)
    reference.eval()
    current = BatchBeamSearch(**kwargs)
    current.to(dtype=dtype)
    current.eval()

    with torch.no_grad():
        expected = reference(x=enc, maxlenratio=maxlenratio, minlenratio=minlenratio)
        actual = current(x=enc, maxlenratio=maxlenratio, minlenratio=minlenratio)

    _assert_identical(expected, actual)


@pytest.mark.parametrize("ctc_weight", [0.0, 0.5])
def test_matches_reference_with_hyp_primer(ctc_weight):
    """A shared `hyp_primer` must be honoured the same way as before."""
    enc, kwargs, dtype = _setup(
        transformer_args, ctc_weight, "transformer", 0.1, torch.float64, False
    )
    # NOTE: kept short on purpose. `CTCPrefixScoreTH` cannot score a prefix
    # longer than the encoder output, and the primer counts towards it, so a
    # longer primer makes *both* implementations raise IndexError on this toy
    # model. That is pre-existing behaviour, not something this test is for.
    primer = [kwargs["sos"], 1]

    reference = ReferenceBatchBeamSearch(**kwargs)
    reference.eval()
    reference.set_hyp_primer(primer)
    current = BatchBeamSearch(**kwargs)
    current.eval()
    current.set_hyp_primer(primer)

    with torch.no_grad():
        expected = reference(x=enc, maxlenratio=0.0, minlenratio=0.0)
        actual = current(x=enc, maxlenratio=0.0, minlenratio=0.0)

    _assert_identical(expected, actual)
    assert actual[0].yseq[: len(primer)].tolist() == primer


@pytest.mark.parametrize("ctc_weight", [0.0, 0.5])
def test_matches_reference_return_hs(ctc_weight):
    """`return_hs` collects the same decoder hidden states as before."""
    enc, kwargs, dtype = _setup(
        transformer_args,
        ctc_weight,
        None,
        0.1,
        torch.float64,
        False,
        return_hs=True,
    )

    reference = ReferenceBatchBeamSearch(**kwargs)
    reference.eval()
    current = BatchBeamSearch(**kwargs)
    current.eval()

    with torch.no_grad():
        expected = reference(x=enc, maxlenratio=0.0, minlenratio=0.0)
        actual = current(x=enc, maxlenratio=0.0, minlenratio=0.0)

    _assert_identical(expected, actual)
    for ref, act in zip(expected, actual):
        assert len(ref.hs) == len(act.hs)
        for a, b in zip(ref.hs, act.hs):
            numpy.testing.assert_allclose(a.numpy(), b.numpy(), rtol=0, atol=0)


def test_reference_is_a_frozen_copy_of_master():
    """Guard the reference against being edited to track the implementation."""
    import inspect
    from test.espnet2.legacy import reference_batch_beam_search

    source = inspect.getsource(reference_batch_beam_search)
    assert "Frozen copy of the pre-batching" in source
    # the batch axis of the frozen copy is the beam, never the utterance
    assert "n_utt" not in source
