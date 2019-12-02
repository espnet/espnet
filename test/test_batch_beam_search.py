from argparse import Namespace

import numpy
import pytest
import torch

from espnet.nets.asr_interface import dynamic_import_asr
from espnet.nets.beam_search import BeamSearch
from espnet.nets.beam_search import Hypothesis
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.batch_beam_search import BatchHypothesis
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.scorers.length_bonus import LengthBonus

from test.test_beam_search import prepare
from test.test_beam_search import rnn_args


def test_batchfy_hyp():
    vocab_size = 5
    eos = -1
    beam = BatchBeamSearch(
        beam_size=3,
        vocab_size=vocab_size,
        weights={"a": 0.5,
                 "b": 0.5},
        scorers={"a": LengthBonus(vocab_size),
                 "b": LengthBonus(vocab_size)},
        sos=eos,
        eos=eos,
    )
    hs = [
        Hypothesis(yseq=torch.tensor([0, 1, 2]), score=0.15,
                   scores={"a": 0.1, "b": 0.2},
                   states={"a": None, "b": None}
                   ),
        Hypothesis(yseq=torch.tensor([0, 1]), score=0.1,
                   scores={"a": 0.0, "b": 0.2},
                   states={"a": None, "b": None}
                   ),
    ]
    bs = beam.batchfy(hs)
    assert torch.all(bs.yseq == torch.tensor([[0, 1, 2], [0, 1, eos]]))
    assert torch.all(bs.score == torch.tensor([0.15, 0.1]))
    # assert beam.unbatchfy(bs) == hs


@pytest.mark.parametrize(
    "model_class, args, ctc_weight, lm_weight, bonus, device, dtype",
    [(nn, args, ctc, lm, bonus, device, dtype)
     for device in ("cpu",)               # "cuda")
     for nn, args in (("rnn", rnn_args),)  # ("transformer", transformer_args))
     for ctc in (0.0,)                    # 0.5, 1.0)
     for lm in (0.0,)                     # 0.5)
     for bonus in (0.0,)                  # 0.1)
     for dtype in ("float32",)            # "float16", "float64")
     ]
)
def test_batch_beam_search_equal(model_class, args, ctc_weight, lm_weight, bonus, device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no cuda device is available")
    if device == "cpu" and dtype == "float16":
        pytest.skip(
            "cpu float16 implementation is not available in pytorch yet")

    # seed setting
    torch.manual_seed(123)
    torch.backends.cudnn.deterministic = True
    # https://github.com/pytorch/pytorch/issues/6351
    torch.backends.cudnn.benchmark = False

    dtype = getattr(torch, dtype)
    model, x, ilens, y, data, train_args = prepare(
        model_class, args, mtlalpha=ctc_weight)
    model.eval()
    char_list = train_args.char_list
    lm_args = Namespace(type="lstm", layer=1, unit=2, dropout_rate=0.0)
    lm = dynamic_import_lm("default", backend="pytorch")(
        len(char_list), lm_args)
    lm.eval()

    # test previous beam search
    args = Namespace(
        beam_size=3,
        penalty=bonus,
        ctc_weight=ctc_weight,
        maxlenratio=0,
        lm_weight=lm_weight,
        minlenratio=0,
        nbest=5
    )

    feat = x[0, :ilens[0]].numpy()
    # legacy beam search
    with torch.no_grad():
        nbest = model.recognize(feat, args, char_list, lm.model)

    # new beam search
    scorers = model.scorers()
    if lm_weight != 0:
        scorers["lm"] = lm
    scorers["length_bonus"] = LengthBonus(len(char_list))
    weights = dict(decoder=1.0 - ctc_weight, ctc=ctc_weight,
                   lm=args.lm_weight, length_bonus=args.penalty)
    model.to(device, dtype=dtype)
    model.eval()
    beam = BatchBeamSearch(
        beam_size=args.beam_size,
        vocab_size=len(char_list),
        weights=weights,
        scorers=scorers,
        token_list=train_args.char_list,
        sos=model.sos,
        eos=model.eos,
    )
    beam.to(device, dtype=dtype)
    beam.eval()
    with torch.no_grad():
        enc = model.encode(torch.as_tensor(feat).to(device, dtype=dtype))
        nbest_bs = beam(x=enc, maxlenratio=args.maxlenratio,
                        minlenratio=args.minlenratio)
    if dtype == torch.float16:
        # skip because results are different. just checking it is decodable
        return

    for i, (expected, actual) in enumerate(zip(nbest, nbest_bs)):
        actual = actual.asdict()
        assert expected["yseq"] == actual["yseq"]
        numpy.testing.assert_allclose(
            expected["score"], actual["score"], rtol=1e-6)
