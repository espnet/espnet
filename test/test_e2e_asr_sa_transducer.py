# coding: utf-8

import argparse
import importlib
import logging
import numpy
import pytest
import torch

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')


def make_train_args(**kwargs):
    train_defaults = dict(
        transformer_init="pytorch",
        transformer_input_layer="conv2d",
        transformer_dec_input_layer="embed",
        etype='transformer',
        elayers=2,
        eunits=16,
        dtype='transformer',
        dlayers=2,
        dunits=16,
        adim=16,
        aheads=2,
        dropout_rate=0.0,
        dropout_rate_decoder=0.0,
        transformer_attn_dropout_rate_encoder=0.0,
        transformer_attn_dropout_rate_decoder=0.0,
        joint_dim=8,
        mtlalpha=1.0,
        trans_type='warp-transducer',
        rnnt_mode='rnnt_mode',
        char_list=['a', 'e', 'i', 'o', 'u'],
        sym_space='<space>',
        sym_blank='<blank>',
        report_cer=False,
        report_wer=False,
        score_norm_transducer=True,
        beam_size=1,
        nbest=1,
        verbose=2,
        outdir=None,
        rnnlm=None)
    train_defaults.update(kwargs)

    return argparse.Namespace(**train_defaults)


def make_recog_args(**kwargs):
    recog_defaults = dict(
        batchsize=0,
        beam_size=1,
        nbest=1,
        verbose=2,
        score_norm_transducer=True,
        rnnlm=None)
    recog_defaults.update(kwargs)

    return argparse.Namespace(**recog_defaults)


def get_default_scope_inputs():
    bs = 5
    idim = 40
    odim = 5

    ilens = [40, 30, 20, 15, 10]
    olens = [3, 9, 10, 2, 3]

    return bs, idim, odim, ilens, olens


def test_sequential():
    from espnet.nets.pytorch_backend.transformer.repeat import MultiSequential

    class Masked(torch.nn.Module):
        def forward(self, x, m):
            return x, m

    f = MultiSequential(Masked(), Masked())
    x = torch.randn(2, 3)
    m = torch.randn(2, 3) > 0
    assert len(f(x, m)) == 2

    if torch.cuda.is_available():
        f = torch.nn.DataParallel(f)
        f.cuda()
        assert len(f(x.cuda(), m.cuda())) == 2


def subsequent_mask(size):
    # http://nlp.seas.harvard.edu/2018/04/03/attention.html
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = numpy.triu(numpy.ones(attn_shape), k=1).astype('uint8')

    return torch.from_numpy(subsequent_mask) == 0


@pytest.mark.parametrize("module", ["pytorch"])
def test_mask(module):
    T = importlib.import_module('espnet.nets.{}_backend.transformer.mask'.format(module))
    m = T.subsequent_mask(3)
    assert (m.unsqueeze(0) == subsequent_mask(3)).all()


def prepare(backend, args):
    bs, idim, odim, ilens, olens = get_default_scope_inputs()
    n_token = odim - 1

    T = importlib.import_module('espnet.nets.{}_backend.e2e_asr_transducer'.format(backend))
    model = T.E2E(idim, odim, args)

    x = torch.randn(bs, 40, idim)
    y = (torch.rand(bs, 10) * n_token % n_token).long()

    for i in range(bs):
        x[i, ilens[i]:] = -1
        y[i, olens[i]:] = model.ignore_id

    data = []
    for i in range(bs):
        data.append(("utt%d" % i, {
            "input": [{"shape": [ilens[i], idim]}],
            "output": [{"shape": [olens[i]]}]
        }))

    return model, x, torch.tensor(ilens), y, data


@pytest.mark.parametrize("module", ["pytorch"])
def test_sa_transducer_mask(module):
    from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
    from espnet.nets.pytorch_backend.transducer.utils import prepare_loss_inputs
    from espnet.nets.pytorch_backend.transformer.mask import target_mask

    train_args = make_train_args()
    model, x, ilens, y, data = prepare(module, train_args)

    # dummy mask
    x_mask = (~make_pad_mask(ilens.tolist())).to(x.device).unsqueeze(-2)

    _, target, _, _ = prepare_loss_inputs(y, x_mask)
    y_mask = target_mask(target, model.blank_id)

    y = model.decoder.embed(target.type(torch.long))
    y[0, 3:] = float("nan")

    a = model.decoder.decoders[0].self_attn
    a(y, y, y, y_mask)
    assert not numpy.isnan(a.attn[0, :, :3, :3].detach().numpy()).any()


@pytest.mark.parametrize('train_dic, recog_dic', [
    ({}, {}),
    ({}, {'beam_size': 4}),
    ({}, {'beam_size': 4, 'nbest': 4}),
    ({}, {'beam_size': 5, 'score_norm_transducer': False}),
    ({'num_save_attention': 1}, {}),
    ({'dropout_rate_encoder': 0.1, 'dropout_rate_decoder': 0.1}, {}),
    ({'eunits': 16, 'elayers': 2, 'joint_dim': 2}, {}),
    ({'adim': 16, 'aheads': 2, 'transformer_attn_dropout_rate_encoder': 0.2}, {'beam_size': 3}),
    ({'transformer_attn_dropout_rate_encoder': 0.2, 'transformer_attn_dropout_rate_decoder': 0.3}, {}),
    ({'dlayers': 2, 'dunits': 16, 'joint_dim': 3}, {'score_norm_transducer': False}),
    ({'transformer_input_layer': 'vgg2l'}, {}),
    ({'transformer_input_layer': 'vgg2l', 'eunits': 8, 'adim': 4, 'joint_dim': 2}, {}),
    ({'report_cer': True, 'beam_size': 1}, {}),
    ({'report_wer': True, 'beam_size': 1}, {}),
    ({'report_cer': True, 'beam_size': 2}, {}),
    ({'report_wer': True, 'beam_size': 2}, {}),
    ({'report_cer': True, 'report_wer': True, 'beam_size': 1}, {}),
    ({'report_wer': True, 'report_wer': True, 'score_norm_transducer': False}, {})])
def test_sa_transducer_trainable_and_decodable(train_dic, recog_dic):
    from espnet.nets.pytorch_backend.transformer import plot

    train_args = make_train_args(**train_dic)
    recog_args = make_recog_args(**recog_dic)

    model, x, ilens, y, data = prepare('pytorch', train_args)

    optim = torch.optim.Adam(model.parameters(), 0.01)
    loss = model(x, ilens, y)

    optim.zero_grad()
    loss.backward()
    optim.step()

    attn_dict = model.calculate_all_attentions(x[0:1], ilens[0:1], y[0:1])
    plot.plot_multi_head_attention(data, attn_dict, "/tmp/espnet-test")

    with torch.no_grad():
        nbest = model.recognize(x[0, :ilens[0]].numpy(), recog_args)

        print(y[0])
        print(nbest[0]["yseq"][1:-1])


def test_sa_transducer_parallel():
    if not torch.cuda.is_available():
        return

    train_args = make_train_args()

    model, x, ilens, y, data = prepare("pytorch", train_args)
    model = torch.nn.DataParallel(model).cuda()

    logging.debug(ilens)

    optim = torch.optim.Adam(model.parameters(), 0.02)

    for i in range(10):
        loss = model(x, torch.as_tensor(ilens), y)

        optim.zero_grad()
        loss.mean().backward()
        optim.step()

        print(loss)
