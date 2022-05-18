import argparse

import chainer
import numpy
import pytest
import torch

import espnet.nets.chainer_backend.e2e_asr_transformer as ch
import espnet.nets.pytorch_backend.e2e_asr_transformer as th
from espnet.nets.pytorch_backend.nets_utils import rename_state_dict
from espnet.nets.pytorch_backend.transformer import plot
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.mask import (subsequent_mask,
                                                          target_mask)


def test_sequential():
    class Masked(torch.nn.Module):
        def forward(self, x, m):
            return x, m

    from espnet.nets.pytorch_backend.transformer.repeat import MultiSequential

    f = MultiSequential(Masked(), Masked())
    x = torch.randn(2, 3)
    m = torch.randn(2, 3) > 0
    assert len(f(x, m)) == 2
    if torch.cuda.is_available():
        f = torch.nn.DataParallel(f)
        f.cuda()
        assert len(f(x.cuda(), m.cuda())) == 2


def ref_subsequent_mask(size):
    # http://nlp.seas.harvard.edu/2018/04/03/attention.html
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    mask = numpy.triu(numpy.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0


def test_mask():
    m = subsequent_mask(3)
    assert (m.unsqueeze(0) == ref_subsequent_mask(3)).all()


def make_arg(**kwargs):
    defaults = dict(
        adim=2,
        aheads=1,
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
        transformer_length_normalized_loss=True,
        report_cer=False,
        report_wer=False,
        mtlalpha=0.0,
        lsm_weight=0.001,
        char_list=["<blank>", "a", "e", "i", "o", "u"],
        ctc_type="warpctc",
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def prepare(backend, args):
    idim = 10
    odim = 3
    batchsize = 2
    ilens = [30, 20]
    olens = [5, 4]
    n_token = odim - 1
    if backend == "pytorch":
        model = th.E2E(idim, odim, args)
        x = torch.randn(batchsize, max(ilens), idim)
        y = (torch.rand(batchsize, max(olens)) * n_token % n_token).long()
    else:
        model = ch.E2E(idim, odim, args)
        x = numpy.random.randn(batchsize, max(ilens), idim).astype(numpy.float32)
        y = numpy.random.rand(batchsize, max(olens)) * n_token % n_token
        y = y.astype(numpy.int32)
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

    if backend == "pytorch":
        return model, x, torch.tensor(ilens), y, data, uttid_list
    else:
        return model, x, ilens, y, data, uttid_list


def test_transformer_mask():
    args = make_arg()
    model, x, ilens, y, data, uttid_list = prepare("pytorch", args)
    yi, yo = add_sos_eos(y, model.sos, model.eos, model.ignore_id)
    y_mask = target_mask(yi, model.ignore_id)
    y = model.decoder.embed(yi)
    y[0, 3:] = float("nan")
    a = model.decoder.decoders[0].self_attn
    a(y, y, y, y_mask)
    assert not numpy.isnan(a.attn[0, :, :3, :3].detach().numpy()).any()


ldconv_lconv_args = dict(
    transformer_decoder_selfattn_layer_type="lightconv",
    transformer_encoder_selfattn_layer_type="lightconv",
    wshare=2,
    ldconv_encoder_kernel_length="5_7_11",
    ldconv_decoder_kernel_length="3_7",
    ldconv_usebias=False,
)

ldconv_dconv_args = dict(
    transformer_decoder_selfattn_layer_type="dynamicconv",
    transformer_encoder_selfattn_layer_type="dynamicconv",
    wshare=2,
    ldconv_encoder_kernel_length="5_7_11",
    ldconv_decoder_kernel_length="3_7",
    ldconv_usebias=False,
)

ldconv_lconv2d_args = dict(
    transformer_decoder_selfattn_layer_type="lightconv2d",
    transformer_encoder_selfattn_layer_type="lightconv2d",
    wshare=2,
    ldconv_encoder_kernel_length="5_7_11",
    ldconv_decoder_kernel_length="3_7",
    ldconv_usebias=False,
)

ldconv_dconv2d_args = dict(
    transformer_decoder_selfattn_layer_type="dynamicconv2d",
    transformer_encoder_selfattn_layer_type="dynamicconv2d",
    wshare=2,
    ldconv_encoder_kernel_length="5_7_11",
    ldconv_decoder_kernel_length="3_7",
    ldconv_usebias=False,
)

interctc_args = dict(
    mtlalpha=1.0,
    elayers=2,
    intermediate_ctc_weight=0.3,
    intermediate_ctc_layer="1",
    stochastic_depth_rate=0.3,
)

selfconditionedctc_args = dict(
    mtlalpha=1.0,
    elayers=2,
    intermediate_ctc_weight=0.3,
    intermediate_ctc_layer="1",
    stochastic_depth_rate=0.0,
    self_conditioning=True,
)


def _savefn(*args, **kwargs):
    return


@pytest.mark.parametrize(
    "module, model_dict",
    [
        ("pytorch", {}),
        ("pytorch", ldconv_lconv_args),
        ("pytorch", ldconv_dconv_args),
        ("pytorch", ldconv_lconv2d_args),
        ("pytorch", ldconv_dconv2d_args),
        ("pytorch", {"report_cer": True}),
        ("pytorch", {"report_wer": True}),
        ("pytorch", {"report_cer": True, "report_wer": True}),
        ("pytorch", {"report_cer": True, "report_wer": True, "mtlalpha": 0.0}),
        ("pytorch", {"report_cer": True, "report_wer": True, "mtlalpha": 1.0}),
        ("pytorch", interctc_args),
        ("pytorch", selfconditionedctc_args),
        ("chainer", {}),
    ],
)
def test_transformer_trainable_and_decodable(module, model_dict):
    args = make_arg(**model_dict)
    model, x, ilens, y, data, uttid_list = prepare(module, args)

    # check for pure CTC and pure Attention
    if args.mtlalpha == 1:
        assert model.decoder is None
    elif args.mtlalpha == 0:
        assert model.ctc is None

    # test beam search
    recog_args = argparse.Namespace(
        beam_size=1,
        penalty=0.0,
        ctc_weight=0.0,
        maxlenratio=1.0,
        lm_weight=0,
        minlenratio=0,
        nbest=1,
    )
    if module == "pytorch":
        # test trainable
        optim = torch.optim.Adam(model.parameters(), 0.01)
        loss = model(x, ilens, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # test attention plot
        attn_dict = model.calculate_all_attentions(x[0:1], ilens[0:1], y[0:1])
        plot.plot_multi_head_attention(data, uttid_list, attn_dict, "", savefn=_savefn)

        # test CTC plot
        ctc_probs = model.calculate_all_ctc_probs(x[0:1], ilens[0:1], y[0:1])
        if args.mtlalpha > 0:
            print(ctc_probs.shape)
        else:
            assert ctc_probs is None

        # test decodable
        with torch.no_grad():
            nbest = model.recognize(x[0, : ilens[0]].numpy(), recog_args)
            print(y[0])
            print(nbest[0]["yseq"][1:-1])
    else:
        # test trainable
        optim = chainer.optimizers.Adam(0.01)
        optim.setup(model)
        loss, loss_ctc, loss_att, acc = model(x, ilens, y)
        model.cleargrads()
        loss.backward()
        optim.update()

        # test attention plot
        attn_dict = model.calculate_all_attentions(x[0:1], ilens[0:1], y[0:1])
        plot.plot_multi_head_attention(data, uttid_list, attn_dict, "", savefn=_savefn)

        # test decodable
        with chainer.no_backprop_mode():
            nbest = model.recognize(x[0, : ilens[0]], recog_args)
            print(y[0])
            print(nbest[0]["yseq"][1:-1])


# https://github.com/espnet/espnet/issues/1750
def test_v0_3_transformer_input_compatibility():
    args = make_arg()
    model, x, ilens, y, data, uttid_list = prepare("pytorch", args)
    # these old names are used in v.0.3.x
    state_dict = model.state_dict()
    prefix = "encoder."
    rename_state_dict(prefix + "embed.", prefix + "input_layer.", state_dict)
    rename_state_dict(prefix + "after_norm.", prefix + "norm.", state_dict)
    prefix = "decoder."
    rename_state_dict(prefix + "after_norm.", prefix + "output_norm.", state_dict)
    model.load_state_dict(state_dict)
