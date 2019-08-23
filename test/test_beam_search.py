from argparse import Namespace

import numpy
import pytest
import torch

from espnet.nets.asr_interface import dynamic_import_asr
from espnet.nets.beam_search import BeamSearch
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.scorers.length_bonus import LengthBonus

rnn_args = Namespace(
    elayers=1,
    subsample="1_2_2_1_1",
    etype="vggblstm",
    eunits=16,
    eprojs=8,
    dtype="lstm",
    dlayers=1,
    dunits=16,
    atype="location",
    aheads=2,
    awin=5,
    aconv_chans=4,
    aconv_filts=10,
    lsm_type="",
    lsm_weight=0.0,
    sampling_probability=0.0,
    adim=16,
    dropout_rate=0.0,
    dropout_rate_decoder=0.0,
    nbest=5,
    beam_size=2,
    penalty=0.5,
    maxlenratio=1.0,
    minlenratio=0.0,
    ctc_weight=0.2,
    lm_weight=0.0,
    rnnlm=None,
    streaming_min_blank_dur=10,
    streaming_onset_margin=2,
    streaming_offset_margin=2,
    verbose=2,
    outdir=None,
    ctc_type="warpctc",
    report_cer=False,
    report_wer=False,
    sym_space="<space>",
    sym_blank="<blank>",
    sortagrad=0,
    grad_noise=False,
    context_residual=False,
    use_frontend=False,
    replace_sos=False,
    tgt_lang=False
)

transformer_args = Namespace(
    adim=16,
    aheads=2,
    dropout_rate=0.0,
    transformer_attn_dropout_rate=None,
    elayers=2,
    eunits=16,
    dlayers=2,
    dunits=16,
    sym_space="<space>",
    sym_blank="<blank>",
    transformer_init="pytorch",
    transformer_input_layer="conv2d",
    transformer_length_normalized_loss=True,
    report_cer=False,
    report_wer=False,
    ctc_type="warpctc",
    lsm_weight=0.001,
)


# from test.test_e2e_asr_transformer import prepare
def prepare(E2E, args, mtlalpha=0.0):
    args.mtlalpha = mtlalpha
    args.char_list = ['a', 'e', 'i', 'o', 'u']
    idim = 40
    odim = 5
    model = dynamic_import_asr(E2E, "pytorch")(idim, odim, args)
    batchsize = 5
    x = torch.randn(batchsize, 40, idim)
    ilens = [40, 30, 20, 15, 10]
    n_token = odim - 1
    # avoid 0 for eps in ctc
    y = (torch.rand(batchsize, 10) * n_token % (n_token - 1)).long() + 1
    olens = [3, 9, 10, 2, 3]
    for i in range(batchsize):
        x[i, ilens[i]:] = -1
        y[i, olens[i]:] = -1

    data = []
    for i in range(batchsize):
        data.append(("utt%d" % i, {
            "input": [{"shape": [ilens[i], idim]}],
            "output": [{"shape": [olens[i]]}]
        }))
    return model, x, torch.tensor(ilens), y, data, args


@pytest.mark.parametrize(
    "model_class, args, ctc_weight, lm_weight, bonus, device, dtype",
    [(nn, args, ctc, lm, bonus, device, dtype)
     for device in ("cpu", "cuda")
     for nn, args in (("transformer", transformer_args), ("rnn", rnn_args))
     for ctc in (0.0, 0.5, 1.0)
     for lm in (0.0, 0.5)
     for bonus in (0.0, 0.1)
     for dtype in ("float16", "float32", "float64")]
)
def test_beam_search_equal(model_class, args, ctc_weight, lm_weight, bonus, device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no cuda device is available")
    if device == "cpu" and dtype == "float16":
        pytest.skip("cpu float16 implementation is not available in pytorch yet")

    # seed setting
    torch.manual_seed(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # https://github.com/pytorch/pytorch/issues/6351

    dtype = getattr(torch, dtype)
    model, x, ilens, y, data, train_args = prepare(model_class, args, mtlalpha=ctc_weight)
    model.eval()
    char_list = train_args.char_list
    lm_args = Namespace(type="lstm", layer=1, unit=2, dropout_rate=0.0)
    lm = dynamic_import_lm("default", backend="pytorch")(len(char_list), lm_args)
    lm.eval()

    # test previous beam search
    args = Namespace(
        beam_size=3,
        penalty=bonus,
        ctc_weight=ctc_weight,
        maxlenratio=0,
        lm_weight=lm_weight,
        minlenratio=0,
        nbest=2
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
    weights = dict(decoder=1.0 - ctc_weight, ctc=ctc_weight, lm=args.lm_weight, length_bonus=args.penalty)
    model.to(device, dtype=dtype)
    model.eval()
    beam = BeamSearch(
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
        nbest_bs = beam(x=enc, maxlenratio=args.maxlenratio, minlenratio=args.minlenratio)
    if dtype == torch.float16:
        # skip because results are different. just checking it is decodable
        return

    for i, (expected, actual) in enumerate(zip(nbest, nbest_bs)):
        actual = actual.asdict()
        assert expected["yseq"] == actual["yseq"]
        numpy.testing.assert_allclose(expected["score"], actual["score"], rtol=1e-6)
