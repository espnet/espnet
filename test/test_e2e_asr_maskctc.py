import argparse

import pytest
import torch

from espnet.nets.pytorch_backend.e2e_asr_maskctc import E2E
from espnet.nets.pytorch_backend.maskctc.add_mask_token import mask_uniform
from espnet.nets.pytorch_backend.transformer import plot


def make_arg(**kwargs):
    defaults = dict(
        adim=2,
        aheads=2,
        dropout_rate=0.0,
        transformer_attn_dropout_rate=None,
        elayers=1,
        eunits=2,
        dlayers=1,
        dunits=2,
        sym_space="<space>",
        sym_blank="<blank>",
        transformer_decoder_selfattn_layer_type="selfattn",
        transformer_encoder_selfattn_layer_type="selfattn",
        transformer_init="pytorch",
        transformer_input_layer="conv2d",
        transformer_length_normalized_loss=False,
        report_cer=False,
        report_wer=False,
        mtlalpha=0.3,
        lsm_weight=0.001,
        wshare=4,
        char_list=["<blank>", "a", "e", "<eos>"],
        ctc_type="builtin",
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def prepare(args):
    idim = 10
    odim = len(args.char_list)
    model = E2E(idim, odim, args)
    batchsize = 2

    x = torch.randn(batchsize, 15, idim)
    ilens = [15, 10]

    n_token = model.odim - 2  # w/o <eos>/<sos>, <mask>
    y = (torch.rand(batchsize, 10) * n_token % n_token).long()
    olens = [7, 6]
    for i in range(batchsize):
        x[i, ilens[i] :] = -1
        y[i, olens[i] :] = model.ignore_id

    data = {}
    uttid_list = []
    for i in range(batchsize):
        data["utt%d" % i] = {
            "input": [{"shape": [ilens[i], idim]}],
            "output": [{"shape": [olens[i]]}],
        }
        uttid_list.append("utt%d" % i)

    return model, x, torch.tensor(ilens), y, data, uttid_list


def test_mask():
    args = make_arg()
    model, x, ilens, y, data, uttid_list = prepare(args)

    # check <sos>/<eos>, <mask> position
    n_char = len(args.char_list) + 1
    assert model.sos == n_char - 2
    assert model.eos == n_char - 2
    assert model.mask_token == n_char - 1
    yi, yo = mask_uniform(y, model.mask_token, model.eos, model.ignore_id)
    assert (
        (yi == model.mask_token).detach().numpy()
        == (yo != model.ignore_id).detach().numpy()
    ).all()


def _savefn(*args, **kwargs):
    return


maskctc_interctc = {
    "maskctc_n_iterations": 0,
    "maskctc_probability_threshold": 0.5,
    "elayers": 2,
    "intermediate_ctc_weight": 0.3,
    "intermediate_ctc_layer": "1",
}


@pytest.mark.parametrize(
    "model_dict",
    [
        ({"maskctc_n_iterations": 1, "maskctc_probability_threshold": 0.0}),
        ({"maskctc_n_iterations": 1, "maskctc_probability_threshold": 0.5}),
        ({"maskctc_n_iterations": 2, "maskctc_probability_threshold": 0.5}),
        ({"maskctc_n_iterations": 0, "maskctc_probability_threshold": 0.5}),
        maskctc_interctc,
    ],
)
def test_transformer_trainable_and_decodable(model_dict):
    args = make_arg(**model_dict)
    model, x, ilens, y, data, uttid_list = prepare(args)

    # decoding params
    recog_args = argparse.Namespace(
        maskctc_n_iterations=args.maskctc_n_iterations,
        maskctc_probability_threshold=args.maskctc_probability_threshold,
    )
    # test training
    optim = torch.optim.Adam(model.parameters(), 0.01)
    loss = model(x, ilens, y)
    optim.zero_grad()
    loss.backward()
    optim.step()

    # test attention plot
    attn_dict = model.calculate_all_attentions(x[0:1], ilens[0:1], y[0:1])
    plot.plot_multi_head_attention(data, uttid_list, attn_dict, "", savefn=_savefn)

    # test decoding
    with torch.no_grad():
        model.recognize(x[0, : ilens[0]].numpy(), recog_args, args.char_list)
