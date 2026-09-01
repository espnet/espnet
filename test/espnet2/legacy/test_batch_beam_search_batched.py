from test.espnet2.legacy.test_beam_search import prepare, transformer_args

import numpy
import pytest
import torch

from espnet2.legacy.nets.batch_beam_search import BatchBeamSearch
from espnet2.legacy.nets.scorers.ctc import CTCPrefixScorer
from espnet2.legacy.nets.scorers.length_bonus import LengthBonus
from espnet2.lm.transformer_lm import TransformerLM


def _build(args, ctc_weight, lm_weight, bonus, device, dtype, normalize_length=False):
    """Build a model, a set of scorers and per-utterance encoder outputs."""
    torch.manual_seed(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model, x, ilens, y, data, train_args = prepare(args, mtlalpha=ctc_weight)
    model.to(device, dtype=dtype)
    model.eval()
    token_list = train_args.token_list

    lm = TransformerLM(len(token_list), unit=2, layer=1, embed_unit=2, dropout_rate=0.0)
    lm.to(device, dtype=dtype)
    lm.eval()

    scorers = {"decoder": model.decoder, "length_bonus": LengthBonus(len(token_list))}
    if lm_weight != 0:
        scorers["lm"] = lm
    if ctc_weight != 0:
        scorers["ctc"] = CTCPrefixScorer(ctc=model.ctc, eos=model.eos)
    weights = dict(
        decoder=1.0 - ctc_weight,
        ctc=ctc_weight,
        lm=lm_weight,
        length_bonus=bonus,
    )

    # Encode every utterance on its own, so that the reference decoding is not
    # affected by encoder-side padding. The batched beam search is then fed the
    # padded stack of exactly these encoder outputs.
    encs = []
    with torch.no_grad():
        for b in range(x.size(0)):
            feat = x[b, : ilens[b]].unsqueeze(0).to(device, dtype=dtype)
            feat_lengths = ilens[b : b + 1].to(device, dtype=torch.int32)
            enc, enc_lens = model.encode(feat, feat_lengths)
            encs.append(enc[0, : enc_lens[0]])

    common = dict(
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        token_list=token_list,
        sos=model.sos,
        eos=model.eos,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
        normalize_length=normalize_length,
    )
    return encs, common, dtype, device


def _pad(encs, device, dtype):
    """Stack per-utterance encoder outputs into a padded batch."""
    lengths = torch.tensor([e.size(0) for e in encs], dtype=torch.long, device=device)
    padded = torch.zeros(
        len(encs), int(lengths.max()), encs[0].size(1), device=device, dtype=dtype
    )
    for b, e in enumerate(encs):
        padded[b, : e.size(0)] = e
    return padded, lengths


def _assert_same_nbest(expected, actual, nbest, rtol):
    assert len(actual) >= min(nbest, len(expected))
    for exp, act in zip(expected[:nbest], actual[:nbest]):
        assert exp.yseq.tolist() == act.yseq.tolist()
        numpy.testing.assert_allclose(
            exp.score.cpu().float(), act.score.cpu().float(), rtol=rtol
        )


@pytest.mark.parametrize(
    "ctc_weight, lm_weight, bonus, beam_size, normalize_length, device, dtype",
    [
        (ctc, lm, bonus, beam, norm, device, dtype)
        for device in ("cpu", "cuda")
        for ctc in (0.0, 0.5, 1.0)
        for lm in (0.0, 0.5)
        for bonus in (0.1,)
        for beam in (1, 3)
        for norm in (False, True)
        for dtype in ("float32", "float64")
    ],
)
def test_utt_batch_beam_search_equal(
    ctc_weight, lm_weight, bonus, beam_size, normalize_length, device, dtype
):
    """Batched decoding must match per-utterance decoding token for token."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no cuda device is available")

    dtype = getattr(torch, dtype)
    encs, common, dtype, device = _build(
        transformer_args,
        ctc_weight,
        lm_weight,
        bonus,
        device,
        dtype,
        normalize_length=normalize_length,
    )

    ref = BatchBeamSearch(beam_size=beam_size, **common)
    ref.to(device, dtype=dtype)
    ref.eval()
    with torch.no_grad():
        expected = [ref(x=e, maxlenratio=0.0, minlenratio=0.0) for e in encs]

    batched = BatchBeamSearch(beam_size=beam_size, **common)
    batched.to(device, dtype=dtype)
    batched.eval()
    padded, lengths = _pad(encs, device, dtype)
    with torch.no_grad():
        actual = batched(x=padded, x_lengths=lengths, maxlenratio=0.0, minlenratio=0.0)

    assert len(actual) == len(encs)
    for b in range(len(encs)):
        _assert_same_nbest(expected[b], actual[b], nbest=len(expected[b]), rtol=1e-5)


@pytest.mark.parametrize("ctc_weight", [0.0, 0.5])
def test_utt_batch_beam_search_equal_uniform_length(ctc_weight):
    """A batch of equally long utterances takes the no-mask code path."""
    encs, common, dtype, device = _build(
        transformer_args, ctc_weight, 0.5, 0.1, "cpu", torch.float64
    )
    # trim to a common length so that `_expand_over_beam` returns no mask
    n = min(e.size(0) for e in encs)
    encs = [e[:n] for e in encs]

    ref = BatchBeamSearch(beam_size=3, **common)
    ref.eval()
    with torch.no_grad():
        expected = [ref(x=e, maxlenratio=0.0, minlenratio=0.0) for e in encs]

    batched = BatchBeamSearch(beam_size=3, **common)
    batched.eval()
    padded, lengths = _pad(encs, device, dtype)
    with torch.no_grad():
        # x_lengths omitted on purpose: every utterance fills the whole tensor
        actual = batched(x=padded, maxlenratio=0.0, minlenratio=0.0)

    for b in range(len(encs)):
        _assert_same_nbest(expected[b], actual[b], nbest=len(expected[b]), rtol=1e-6)


@pytest.mark.parametrize("maxlenratio", [-4.0, 0.5])
def test_utt_batch_beam_search_equal_maxlenratio(maxlenratio):
    """Non-zero maxlenratio disables end detection and caps the output length."""
    encs, common, dtype, device = _build(
        transformer_args, 0.5, 0.5, 0.1, "cpu", torch.float64
    )

    ref = BatchBeamSearch(beam_size=3, **common)
    ref.eval()
    with torch.no_grad():
        expected = [ref(x=e, maxlenratio=maxlenratio, minlenratio=0.0) for e in encs]

    batched = BatchBeamSearch(beam_size=3, **common)
    batched.eval()
    padded, lengths = _pad(encs, device, dtype)
    with torch.no_grad():
        actual = batched(
            x=padded, x_lengths=lengths, maxlenratio=maxlenratio, minlenratio=0.0
        )

    for b in range(len(encs)):
        _assert_same_nbest(expected[b], actual[b], nbest=len(expected[b]), rtol=1e-6)


def test_utt_batch_beam_search_repeated_utterance():
    """Identical utterances in one batch must give identical n-best lists."""
    encs, common, dtype, device = _build(
        transformer_args, 0.5, 0.5, 0.1, "cpu", torch.float64
    )
    encs = [encs[0], encs[0], encs[0]]

    batched = BatchBeamSearch(beam_size=3, **common)
    batched.eval()
    padded, lengths = _pad(encs, device, dtype)
    with torch.no_grad():
        actual = batched(x=padded, x_lengths=lengths, maxlenratio=0.0, minlenratio=0.0)

    for b in (1, 2):
        _assert_same_nbest(actual[0], actual[b], nbest=len(actual[0]), rtol=1e-10)


def test_utt_batch_beam_search_hyp_primer():
    """A per-utterance primer conditions each utterance independently."""
    encs, common, dtype, device = _build(
        transformer_args, 0.0, 0.0, 0.1, "cpu", torch.float64
    )
    sos = common["sos"]

    batched = BatchBeamSearch(beam_size=2, **common)
    batched.eval()
    padded, lengths = _pad(encs, device, dtype)

    batched.set_hyp_primer([[sos, 1], [sos, 2]])
    with torch.no_grad():
        actual = batched(x=padded, x_lengths=lengths, maxlenratio=-5.0)
    assert actual[0][0].yseq[:2].tolist() == [sos, 1]
    assert actual[1][0].yseq[:2].tolist() == [sos, 2]

    # primers of different lengths cannot be advanced in lock step
    batched.set_hyp_primer([[sos, 1], [sos, 1, 2]])
    with pytest.raises(ValueError):
        with torch.no_grad():
            batched(x=padded, x_lengths=lengths, maxlenratio=-5.0)


def test_utt_batch_beam_search_rejects_bad_shape():
    encs, common, dtype, device = _build(
        transformer_args, 0.0, 0.0, 0.1, "cpu", torch.float64
    )
    batched = BatchBeamSearch(beam_size=2, **common)
    batched.eval()
    with pytest.raises(ValueError):
        batched(x=encs[0][:, 0], maxlenratio=-5.0)
