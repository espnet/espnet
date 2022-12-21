from argparse import Namespace

import pytest
import torch

from espnet.nets.asr_interface import dynamic_import_asr
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.time_sync_beam_search import TimeSyncBeamSearch

rnn_args = Namespace(
    elayers=1,
    subsample=None,
    etype="vgglstm",
    eunits=2,
    eprojs=2,
    dtype="lstm",
    dlayers=1,
    dunits=2,
    atype="dot",
    aheads=2,
    awin=2,
    aconv_chans=2,
    aconv_filts=2,
    lsm_type="",
    lsm_weight=0.0,
    sampling_probability=0.0,
    adim=2,
    dropout_rate=0.0,
    dropout_rate_decoder=0.0,
    nbest=3,
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
    ctc_type="builtin",
    report_cer=False,
    report_wer=False,
    sym_space="<space>",
    sym_blank="<blank>",
    sortagrad=0,
    grad_noise=False,
    context_residual=False,
    use_frontend=False,
    replace_sos=False,
    tgt_lang=False,
)

transformer_args = Namespace(
    adim=4,
    aheads=2,
    dropout_rate=0.0,
    transformer_attn_dropout_rate=None,
    elayers=1,
    eunits=2,
    dlayers=1,
    dunits=2,
    sym_space="<space>",
    sym_blank="<blank>",
    transformer_init="pytorch",
    transformer_input_layer="conv2d",
    transformer_length_normalized_loss=True,
    report_cer=False,
    report_wer=False,
    ctc_type="builtin",
    lsm_weight=0.001,
)

ldconv_args = Namespace(
    **vars(transformer_args),
    transformer_decoder_selfattn_layer_type="lightconv",
    transformer_encoder_selfattn_layer_type="lightconv",
    wshare=2,
    ldconv_encoder_kernel_length="31_31",
    ldconv_decoder_kernel_length="11_11",
    ldconv_usebias=False,
)


# from test.test_e2e_asr_transformer import prepare
def prepare(E2E, args, mtlalpha=0.0):
    args.mtlalpha = mtlalpha
    args.char_list = ["a", "e", "i", "o", "u"]
    idim = 8
    odim = len(args.char_list)
    model = dynamic_import_asr(E2E, "pytorch")(idim, odim, args)

    batchsize = 1
    x = torch.randn(batchsize, 20, idim)
    ilens = [20, 15]
    n_token = odim - 1
    # avoid 0 for eps in ctc
    y = (torch.rand(batchsize, 10) * n_token % (n_token - 1)).long() + 1
    olens = [10, 2]
    for i in range(batchsize):
        x[i, ilens[i] :] = -1
        y[i, olens[i] :] = -1

    data = []
    for i in range(batchsize):
        data.append(
            (
                "utt%d" % i,
                {
                    "input": [{"shape": [ilens[i], idim]}],
                    "output": [{"shape": [olens[i]]}],
                },
            )
        )
    return model, x, torch.tensor(ilens), y, data, args


@pytest.mark.parametrize(
    "model_class, args, mtlalpha, ctc_weight, lm_weight, bonus, device, dtype",
    [
        (nn, args, ctc_train, ctc_recog, lm, bonus, device, dtype)
        for device in ("cpu", "cuda")
        for nn, args in (
            ("transformer", transformer_args),
            ("transformer", ldconv_args),
            ("rnn", rnn_args),
        )
        for ctc_train in (0.0, 0.5, 1.0)
        for ctc_recog in (0.0, 0.5, 1.0)
        for lm in (0.5,)
        for bonus in (0.1,)
        for dtype in ("float16", "float32", "float64")
    ],
)
def test_beam_search_equal(
    model_class, args, mtlalpha, ctc_weight, lm_weight, bonus, device, dtype
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no cuda device is available")
    if device == "cpu" and dtype == "float16":
        pytest.skip("cpu float16 implementation is not available in pytorch yet")
    if mtlalpha == 0.0 and ctc_weight > 0.0:
        pytest.skip("no CTC + CTC decoding.")
    if mtlalpha == 1.0 and ctc_weight < 1.0:
        pytest.skip("pure CTC + attention decoding")

    # seed setting
    torch.manual_seed(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = (
        False  # https://github.com/pytorch/pytorch/issues/6351
    )

    dtype = getattr(torch, dtype)
    model, x, ilens, y, data, train_args = prepare(model_class, args, mtlalpha=mtlalpha)
    model.eval()
    char_list = train_args.char_list
    lm_args = Namespace(type="lstm", layer=1, unit=2, embed_unit=2, dropout_rate=0.0)
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
        nbest=3,
    )

    feat = x[0, : ilens[0]].numpy()

    # new beam search
    scorers = model.scorers()
    scorers["ctc"] = model.ctc
    if lm_weight != 0:
        scorers["lm"] = lm
    scorers["length_bonus"] = LengthBonus(len(char_list))
    weights = dict(
        decoder=1.0 - ctc_weight,
        ctc=ctc_weight,
        lm=args.lm_weight,
        length_bonus=args.penalty,
    )
    model.to(device, dtype=dtype)
    model.eval()
    beam = TimeSyncBeamSearch(
        beam_size=args.beam_size,
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        token_list=train_args.char_list,
    )
    beam.to(device, dtype=dtype)
    beam.eval()
    with torch.no_grad():
        enc = model.encode(torch.as_tensor(feat).to(device, dtype=dtype))
        beam(x=enc, maxlenratio=args.maxlenratio, minlenratio=args.minlenratio)

    # just checking it is decodable
    return
