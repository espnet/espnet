import os
from argparse import Namespace
from test.test_beam_search import prepare, transformer_args

import numpy
import pytest
import torch

from espnet.nets.batch_beam_search import BatchBeamSearch, BeamSearch
from espnet.nets.beam_search import Hypothesis
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.scorers.ngram import NgramFullScorer


def test_batchfy_hyp():
    vocab_size = 5
    eos = -1
    # simplest beam search
    beam = BatchBeamSearch(
        beam_size=3,
        vocab_size=vocab_size,
        weights={"a": 0.5, "b": 0.5},
        scorers={"a": LengthBonus(vocab_size), "b": LengthBonus(vocab_size)},
        pre_beam_score_key="a",
        sos=eos,
        eos=eos,
    )
    hs = [
        Hypothesis(
            yseq=torch.tensor([0, 1, 2]),
            score=torch.tensor(0.15),
            scores={"a": torch.tensor(0.1), "b": torch.tensor(0.2)},
            states={"a": 1, "b": 2},
        ),
        Hypothesis(
            yseq=torch.tensor([0, 1]),
            score=torch.tensor(0.1),
            scores={"a": torch.tensor(0.0), "b": torch.tensor(0.2)},
            states={"a": 3, "b": 4},
        ),
    ]
    bs = beam.batchfy(hs)
    assert torch.all(bs.yseq == torch.tensor([[0, 1, 2], [0, 1, eos]]))
    assert torch.all(bs.score == torch.tensor([0.15, 0.1]))
    assert torch.all(bs.scores["a"] == torch.tensor([0.1, 0.0]))
    assert torch.all(bs.scores["b"] == torch.tensor([0.2, 0.2]))
    assert bs.states["a"] == [1, 3]
    assert bs.states["b"] == [2, 4]

    us = beam.unbatchfy(bs)
    for i in range(len(hs)):
        assert us[i].yseq.tolist() == hs[i].yseq.tolist()
        assert us[i].score == hs[i].score
        assert us[i].scores == hs[i].scores
        assert us[i].states == hs[i].states


lstm_lm = Namespace(type="lstm", layer=1, unit=2, dropout_rate=0.0)
gru_lm = Namespace(type="gru", layer=1, unit=2, dropout_rate=0.0)
transformer_lm = Namespace(
    layer=1, unit=2, att_unit=2, embed_unit=2, head=1, pos_enc="none", dropout_rate=0.0
)


@pytest.mark.parametrize(
    "model_class, args, ctc_weight, lm_nn, lm_args, lm_weight, ngram_weight, \
        bonus, device, dtype",
    [
        (nn, args, ctc, lm_nn, lm_args, lm, ngram, bonus, device, dtype)
        for device in ("cpu", "cuda")
        # (("rnn", rnn_args),)
        for nn, args in (("transformer", transformer_args),)
        for ctc in (0.0, 0.5, 1.0)
        for lm_nn, lm_args in (
            ("default", lstm_lm),
            ("default", gru_lm),
            ("transformer", transformer_lm),
        )
        for lm in (0.5,)
        for ngram in (0.5,)
        for bonus in (0.1,)
        for dtype in ("float32", "float64")  # TODO(karita): float16
    ],
)
def test_batch_beam_search_equal(
    model_class,
    args,
    ctc_weight,
    lm_nn,
    lm_args,
    lm_weight,
    ngram_weight,
    bonus,
    device,
    dtype,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no cuda device is available")
    if device == "cpu" and dtype == "float16":
        pytest.skip("cpu float16 implementation is not available in pytorch yet")

    # seed setting
    torch.manual_seed(123)
    torch.backends.cudnn.deterministic = True
    # https://github.com/pytorch/pytorch/issues/6351
    torch.backends.cudnn.benchmark = False

    dtype = getattr(torch, dtype)
    model, x, ilens, y, data, train_args = prepare(
        model_class, args, mtlalpha=ctc_weight
    )
    model.eval()
    char_list = train_args.char_list
    lm = dynamic_import_lm(lm_nn, backend="pytorch")(len(char_list), lm_args)
    lm.eval()
    root = os.path.dirname(os.path.abspath(__file__))
    ngram = NgramFullScorer(os.path.join(root, "beam_search_test.arpa"), args.char_list)

    # test previous beam search
    args = Namespace(
        beam_size=3,
        penalty=bonus,
        ctc_weight=ctc_weight,
        maxlenratio=0,
        lm_weight=lm_weight,
        ngram_weight=ngram_weight,
        minlenratio=0,
        nbest=5,
    )

    # new beam search
    scorers = model.scorers()
    if lm_weight != 0:
        scorers["lm"] = lm
    if ngram_weight != 0:
        scorers["ngram"] = ngram
    scorers["length_bonus"] = LengthBonus(len(char_list))
    weights = dict(
        decoder=1.0 - ctc_weight,
        ctc=ctc_weight,
        lm=args.lm_weight,
        ngram=args.ngram_weight,
        length_bonus=args.penalty,
    )
    model.to(device, dtype=dtype)
    model.eval()
    with torch.no_grad():
        enc = model.encode(x[0, : ilens[0]].to(device, dtype=dtype))

    legacy_beam = BeamSearch(
        beam_size=args.beam_size,
        vocab_size=len(char_list),
        weights=weights,
        scorers=scorers,
        token_list=train_args.char_list,
        sos=model.sos,
        eos=model.eos,
        pre_beam_score_key=None if ctc_weight == 1.0 else "full",
    )
    legacy_beam.to(device, dtype=dtype)
    legacy_beam.eval()

    beam = BatchBeamSearch(
        beam_size=args.beam_size,
        vocab_size=len(char_list),
        weights=weights,
        scorers=scorers,
        token_list=train_args.char_list,
        sos=model.sos,
        eos=model.eos,
        pre_beam_score_key=None if ctc_weight == 1.0 else "full",
    )
    beam.to(device, dtype=dtype)
    beam.eval()
    with torch.no_grad():
        legacy_nbest_bs = legacy_beam(
            x=enc, maxlenratio=args.maxlenratio, minlenratio=args.minlenratio
        )
        nbest_bs = beam(
            x=enc, maxlenratio=args.maxlenratio, minlenratio=args.minlenratio
        )

    for i, (expected, actual) in enumerate(zip(legacy_nbest_bs, nbest_bs)):
        assert expected.yseq.tolist() == actual.yseq.tolist()
        numpy.testing.assert_allclose(
            expected.score.cpu(), actual.score.cpu(), rtol=1e-6
        )
