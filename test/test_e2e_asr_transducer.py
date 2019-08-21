# coding: utf-8

import argparse
import importlib
import numpy as np
import pytest
import torch

from espnet.nets.pytorch_backend.nets_utils import pad_list


def get_default_train_args(**kwargs):
    train_defaults = dict(
        etype='vggblstmp',
        elayers=1,
        subsample="1_2_2_1_1",
        eunits=8,
        eprojs=8,
        dtype='lstm',
        dlayers=1,
        dunits=8,
        dec_embed_dim=8,
        atype='location',
        adim=8,
        aheads=2,
        awin=5,
        aconv_chans=4,
        aconv_filts=10,
        dropout_rate=0.0,
        dropout_rate_decoder=0.0,
        dropout_rate_embed_decoder=0.0,
        joint_dim=8,
        mtlalpha=1.0,
        rnnt_mode='rnnt',
        use_frontend=False,
        rnnt_type='warp-transducer',
        char_list=['a', 'b', 'c', 'd'],
        sym_space='<space>',
        sym_blank='<blank>',
        report_cer=False,
        report_wer=False,
        beam_size=1,
        nbest=1,
        verbose=2,
        outdir=None,
        rnnlm=None,
    )
    train_defaults.update(kwargs)

    return argparse.Namespace(**train_defaults)


def get_default_recog_args(**kwargs):
    recog_defaults = dict(
        batchsize=0,
        beam_size=2,
        nbest=1,
        verbose=2,
        score_norm_transducer=True,
        rnnlm=None
    )
    recog_defaults.update(kwargs)

    return argparse.Namespace(**recog_defaults)


def get_default_scope_inputs():
    idim = 40
    odim = 4
    ilens = [20, 15]
    olens = [4, 3]

    return idim, odim, ilens, olens


def prepare_inputs(backend, idim, odim, ilens, olens, is_cuda=False):
    np.random.seed(1)

    xs = [np.random.randn(ilen, idim).astype(np.float32) for ilen in ilens]
    ys = [np.random.randint(1, odim, olen).astype(np.int32) for olen in olens]
    ilens = np.array([x.shape[0] for x in xs], dtype=np.int32)

    if backend == 'pytorch':
        xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0)
        ys_pad = pad_list([torch.from_numpy(y).long() for y in ys], -1)
        ilens = torch.from_numpy(ilens).long()

        if is_cuda:
            xs_pad = xs_pad.cuda()
            ys_pad = ys_pad.cuda()
            ilens = ilens.cuda()

        return xs_pad, ilens, ys_pad
    else:
        raise ValueError('Invalid backend')


@pytest.mark.parametrize('train_dic, recog_dic', [
    ({}, {'beam_size': 1}),
    ({'rnnt_mode': 'rnnt-att'}, {'beam_size': 1}),
    ({}, {'beam_size': 8}),
    ({'rnnt_mode': 'rnnt-att'}, {'beam_size': 8}),
    ({}, {}),
    ({'rnnt_mode': 'rnnt-att'}, {}),
    ({'etype': 'gru'}, {}),
    ({'rnnt_mode': 'rnnt-att', 'etype': 'gru'}, {}),
    ({'etype': 'blstm'}, {}),
    ({'rnnt_mode': 'rnnt-att', 'etype': 'blstm'}, {}),
    ({'etype': 'vgggru'}, {}),
    ({'rnnt_mode': 'rnnt-att', 'etype': 'vgggru'}, {}),
    ({'etype': 'vggbru'}, {}),
    ({'rnnt_mode': 'rnnt-att', 'etype': 'vggbgru'}, {}),
    ({'etype': 'vgggrup'}, {}),
    ({'rnnt_mode': 'rnnt-att', 'etype': 'vgggrup'}, {}),
    ({'etype': 'blstm', 'elayers': 2}, {}),
    ({'rnnt_mode': 'rnnt-att', 'etype': 'blstm', 'elayers': 2}, {}),
    ({'etype': 'blstm', 'eunits': 16}, {}),
    ({'rnnt_mode': 'rnnt-att', 'etype': 'blstm', 'eunits': 16}, {}),
    ({'etype': 'blstm', 'eprojs': 16}, {}),
    ({'rnnt_mode': 'rnnt-att', 'etype': 'blstm', 'eprojs': 16}, {}),
    ({'dtype': 'gru'}, {}),
    ({'rnnt_mode': 'rnnt-att', 'dtype': 'gru'}, {}),
    ({'dtype': 'bgrup'}, {}),
    ({'rnnt_mode': 'rnnt-att', 'dtype': 'bgrup'}, {}),
    ({'dtype': 'gru', 'dlayers': 2}, {}),
    ({'rnnt_mode': 'rnnt-att', 'dtype': 'gru', 'dlayers': 2}, {}),
    ({'dtype': 'lstm', 'dlayers': 3}, {}),
    ({'rnnt_mode': 'rnnt-att', 'dtype': 'lstm', 'dlayers': 3}, {}),
    ({'dtype': 'gru', 'dunits': 16}, {}),
    ({'rnnt_mode': 'rnnt-att', 'dtype': 'gru', 'dunits': 16}, {}),
    ({'dtype': 'lstm', 'dlayers': 2, 'dunits': 16}, {}),
    ({'rnnt_mode': 'rnnt-att', 'dtype': 'lstm', 'dlayers': 3, 'dunits': 16}, {}),
    ({'joint-dim': 16}, {}),
    ({'rnnt_mode': 'rnnt-att', 'joint-dim': 16}, {}),
    ({'dtype': 'lstm', 'dlayers': 2, 'dunits': 16, 'joint-dim': 4}, {}),
    ({'rnnt_mode': 'rnnt-att', 'dtype': 'lstm', 'dlayers': 3, 'dunits': 16, 'joint-dim': 4}, {}),
    ({'dec-embed-dim': 16}, {}),
    ({'dec-embed-dim': 16, 'dropout-rate-embed-decoder': 0.1}, {}),
    ({'dunits': 16}, {'beam_size': 1}),
    ({'rnnt_mode': 'rnnt-att', 'dunits': 2}, {'beam_size': 1}),
    ({'dropout-rate-decoder': 0.2}, {}),
    ({'rnnt-mode': 'rnnt-att', 'dropout-rate-decoder': 0.2}, {}),
    ({'rnnt_mode': 'rnnt-att', 'atype': 'noatt'}, {}),
    ({'rnnt_mode': 'rnnt-att', 'atype': 'dot'}, {}),
    ({'rnnt_mode': 'rnnt-att', 'atype': 'coverage'}, {}),
    ({'rnnt_mode': 'rnnt-att', 'atype': 'coverage'}, {}),
    ({'rnnt_mode': 'rnnt-att', 'atype': 'coverage_location'}, {}),
    ({'rnnt_mode': 'rnnt-att', 'atype': 'location2d'}, {}),
    ({'rnnt_mode': 'rnnt-att', 'atype': 'location_recurrent'}, {}),
    ({'rnnt_mode': 'rnnt-att', 'atype': 'multi_head_dot'}, {}),
    ({'rnnt_mode': 'rnnt-att', 'atype': 'multi_head_add'}, {}),
    ({'rnnt_mode': 'rnnt-att', 'atype': 'multi_head_loc'}, {}),
    ({'rnnt_mode': 'rnnt-att', 'atype': 'multi_head_multi_res_loc'}, {}),
    ({}, {'score_norm_transducer': False}),
    ({'rnnt_mode': 'rnnt-att'}, {'score_norm_transducer': False}),
    ({}, {'nbest': 2}),
    ({'rnnt_mode': 'rnnt-att'}, {'nbest': 2}),
    ({'beam_size': 1, 'report_cer': True, 'report_wer': True}, {}),
    ({'rnnt_mode': 'rnnt-att', 'beam_size': 1, 'report_cer': True, 'report_wer': True}, {}),
    ({'beam_size': 1, 'report_cer': True, 'report_wer': False}, {}),
    ({'rnnt_mode': 'rnnt-att', 'beam_size': 1, 'report_cer': True, 'report_wer': False}, {}),
    ({'beam_size': 1, 'report_cer': False, 'report_wer': True}, {}),
    ({'rnnt_mode': 'rnnt-att', 'beam_size': 1, 'report_cer': False, 'report_wer': True}, {})])
def test_pytorch_transducer_trainable_and_decodable(train_dic, recog_dic, backend='pytorch'):
    idim, odim, ilens, olens = get_default_scope_inputs()
    train_args = get_default_train_args(**train_dic)

    module = importlib.import_module('espnet.nets.{}_backend.e2e_asr_transducer'.format(backend))
    model = module.E2E(idim, odim, train_args)

    batch = prepare_inputs(backend, idim, odim, ilens, olens)

    loss = model(*batch)
    loss.backward()

    with torch.no_grad():
        in_data = np.random.randn(20, idim)
        recog_args = get_default_recog_args(**recog_dic)
        model.recognize(in_data, recog_args, train_args.char_list)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu required")
@pytest.mark.parametrize('backend', ['pytorch'])
def test_pytorch_transducer_gpu_trainable(backend):
    idim, odim, ilens, olens = get_default_scope_inputs()
    train_args = get_default_train_args()

    module = importlib.import_module('espnet.nets.{}_backend.e2e_asr_transducer'.format(backend))
    model = module.E2E(idim, odim, train_args)
    model.cuda()

    batch = prepare_inputs(backend, idim, odim, ilens, olens, is_cuda=True)

    loss = model(*batch)
    loss.backward()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")
@pytest.mark.parametrize('backend', ['pytorch'])
def test_pytorch_multi_gpu_trainable(backend):
    idim, odim, ilens, olens = get_default_scope_inputs()
    train_args = get_default_train_args()

    ngpu = 2
    device_ids = list(range(ngpu))

    module = importlib.import_module('espnet.nets.{}_backend.e2e_asr_transducer'.format(backend))
    model = module.E2E(idim, odim, train_args)
    model = torch.nn.DataParallel(model, device_ids)
    model.cuda()

    batch = prepare_inputs(backend, idim, odim, ilens, olens, is_cuda=True)

    loss = 1. / ngpu * model(*batch)
    loss.backward(loss.new_ones(ngpu))


@pytest.mark.parametrize(
    'atype', [
        'noatt', 'dot', 'location', 'noatt', 'add', 'coverage',
        'coverage_location', 'location2d', 'location_recurrent',
        'multi_head_dot', 'multi_head_add', 'multi_head_loc',
        'multi_head_multi_res_loc'])
def test_pytorch_calculate_all_attentions(atype, backend='pytorch'):
    idim, odim, ilens, olens = get_default_scope_inputs()
    train_args = get_default_train_args(rnnt_mode='rnnt-att', atype=atype)

    module = importlib.import_module('espnet.nets.{}_backend.e2e_asr_transducer'.format(backend))
    model = module.E2E(idim, odim, train_args)

    batch = prepare_inputs(backend, idim, odim, ilens, olens, is_cuda=False)

    att_ws = model.calculate_all_attentions(*batch)[0]
    print(att_ws.shape)
