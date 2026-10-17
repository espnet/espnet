from argparse import Namespace

import pytest
import torch

from espnet2.legacy.nets.beam_search_timesync import BeamSearchTimeSync
from espnet2.legacy.nets.scorers.length_bonus import LengthBonus
from espnet2.lm.transformer_lm import TransformerLM
from espnet2.tasks.asr import ASRTask

rnn_args = Namespace(
    encoder="rnn",
    decoder="rnn",
    input_size=8,
    encoder_conf=dict(
        num_layers=1,
        hidden_size=2,
    ),
    specaug=None,
    normalize="utterance_mvn",
    normalize_conf={},
    decoder_conf=dict(
        hidden_size=2,
        num_layers=1,
    ),
    token_list=["a", "e", "i", "o", "u"],
    odim=5,
    mtlalpha=0.0,
    ctc_weight=0.2,
    ctc_conf={},
    init=None,
    ignore_id=-1,
    model_conf=dict(
        ctc_weight=0.2,
        ignore_id=-1,
    ),
)
transformer_args = Namespace(
    encoder="transformer",
    decoder="transformer",
    input_size=8,
    encoder_conf=dict(
        attention_heads=2,
        linear_units=2,
        num_blocks=1,
        dropout_rate=0.0,
    ),
    specaug=None,
    normalize="utterance_mvn",
    normalize_conf={},
    decoder_conf=dict(
        attention_heads=2,
        linear_units=2,
        num_blocks=1,
        dropout_rate=0.0,
    ),
    token_list=["a", "e", "i", "o", "u"],
    odim=5,
    mtlalpha=0.0,
    ctc_weight=0.2,
    ctc_conf={},
    init=None,
    ignore_id=-1,
    model_conf=dict(
        ctc_weight=0.2,
        ignore_id=-1,
    ),
)
ldconv_args = Namespace(
    **vars(transformer_args),
)


def prepare(args, mtlalpha=0.0):
    args.mtlalpha = mtlalpha
    args.token_list = ["a", "e", "i", "o", "u"]
    args.odim = len(args.token_list)
    model = ASRTask.build_model(args)

    batchsize = 1
    x = torch.randn(batchsize, 20, 8)
    ilens = [20]
    n_token = args.odim - 1
    y = (torch.rand(batchsize, 10) * n_token % (n_token - 1)).long() + 1
    olens = [10]
    for i in range(batchsize):
        x[i, ilens[i] :] = -1
        y[i, olens[i] :] = -1

    data = []
    for i in range(batchsize):
        data.append(
            (
                "utt%d" % i,
                {
                    "input": [{"shape": [ilens[i], 8]}],
                    "output": [{"shape": [olens[i]]}],
                },
            )
        )
    return model, x, torch.tensor(ilens), y, data, args


@pytest.mark.parametrize(
    "args, mtlalpha, ctc_weight, lm_weight, bonus, device, dtype",
    [
        (args, ctc_train, ctc_recog, lm, bonus, device, dtype)
        for device in ("cpu", "cuda")
        for args in (transformer_args, ldconv_args, rnn_args)
        for ctc_train in (0.0, 0.5, 1.0)
        for ctc_recog in (0.0, 0.5, 1.0)
        for lm in (0.5,)
        for bonus in (0.1,)
        for dtype in ("float16", "float32", "float64")
    ],
)
def test_beam_search_timesync(
    args, mtlalpha, ctc_weight, lm_weight, bonus, device, dtype
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no cuda device is available")
    if device == "cpu" and dtype == "float16":
        pytest.skip("cpu float16 implementation is not available in pytorch yet")
    if mtlalpha == 0.0 or ctc_weight == 0:
        pytest.skip("no CTC.")
    if mtlalpha == 1.0 and ctc_weight < 1.0:
        pytest.skip("pure CTC + attention decoding")
    if ctc_weight == 1.0:
        pytest.skip("pure CTC beam search is not implemented")

    torch.manual_seed(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dtype = getattr(torch, dtype)
    model, x, ilens, y, data, train_args = prepare(args, mtlalpha=mtlalpha)
    model.eval()
    token_list = train_args.token_list

    lm_args = Namespace(
        lm="default",
        lm_conf=dict(
            unit=2,
            layer=1,
            embed_unit=2,
            dropout_rate=0.0,
        ),
        token_list=token_list,
    )
    lm = TransformerLM(len(token_list), **lm_args.lm_conf)
    lm.eval()

    recog_args = Namespace(
        beam_size=3,
        penalty=bonus,
        ctc_weight=ctc_weight,
        maxlenratio=0,
        lm_weight=lm_weight,
        minlenratio=0,
        nbest=3,
    )

    feat = x[0, : ilens[0]].unsqueeze(0)  # (1, T, D)
    feat_lengths = ilens[:1]
    model.to(device, dtype=dtype)
    model.eval()

    scorers = {}
    scorers["decoder"] = model.decoder
    scorers["ctc"] = model.ctc
    if lm_weight != 0:
        scorers["lm"] = lm
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = dict(
        decoder=1.0 - ctc_weight,
        ctc=ctc_weight,
        lm=recog_args.lm_weight,
        length_bonus=recog_args.penalty,
    )
    beam = BeamSearchTimeSync(
        beam_size=recog_args.beam_size,
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        token_list=token_list,
    )
    beam.to(device, dtype=dtype)
    beam.eval()
    with torch.no_grad():
        enc, _ = model.encode(
            torch.as_tensor(feat).to(device, dtype=dtype),
            torch.as_tensor(feat_lengths).to(device, dtype=torch.int32),
        )
        beam(
            x=enc[0],
            maxlenratio=recog_args.maxlenratio,
            minlenratio=recog_args.minlenratio,
        )
    # just checking it is decodable
