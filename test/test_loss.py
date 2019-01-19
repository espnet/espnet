# coding: utf-8

# Copyright 2017 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from distutils.version import LooseVersion

import chainer.functions as F
import numpy
import pytest
import torch

from espnet.nets.pytorch_backend.e2e_asr import pad_list
from espnet.nets.pytorch_backend.nets_utils import th_accuracy


@pytest.mark.parametrize('in_length,out_length',
                         [([11, 17, 15], [4, 2, 3]),
                          ([4], [1]),
                          ([30, 18, 10, 8], [10, 9, 6, 1])])
def test_ctc_loss(in_length, out_length):
    pytest.importorskip("torch")
    if LooseVersion(torch.__version__) < LooseVersion('1.0'):
        pytest.importorskip("warpctc_pytorch")
        import warpctc_pytorch
        torch_ctcloss = warpctc_pytorch.CTCLoss(size_average=True)
    else:
        _ctcloss_sum = torch.nn.CTCLoss(reduction='sum')

        def torch_ctcloss(th_pred, th_target, th_ilen, th_olen):
            th_pred = th_pred.log_softmax(2)
            loss = _ctcloss_sum(th_pred, th_target, th_ilen, th_olen)
            loss = loss / th_pred.size(1)
            return loss

    n_out = 7
    input_length = numpy.array(in_length, dtype=numpy.int32)
    label_length = numpy.array(out_length, dtype=numpy.int32)
    np_pred = [numpy.random.rand(il, n_out).astype(
        numpy.float32) for il in input_length]
    np_target = [numpy.random.randint(
        0, n_out, size=ol, dtype=numpy.int32) for ol in label_length]

    # NOTE: np_pred[i] seems to be transposed and used axis=-1 in e2e_asr.py
    ch_pred = F.separate(F.pad_sequence(np_pred), axis=-2)
    ch_target = F.pad_sequence(np_target, padding=-1)
    ch_loss = F.connectionist_temporal_classification(
        ch_pred, ch_target, 0, input_length, label_length).data

    th_pred = pad_list([torch.from_numpy(x)
                        for x in np_pred], 0.0).transpose(0, 1)
    th_target = torch.from_numpy(numpy.concatenate(np_target))
    th_ilen = torch.from_numpy(input_length)
    th_olen = torch.from_numpy(label_length)
    th_loss = torch_ctcloss(th_pred, th_target, th_ilen, th_olen).numpy()
    numpy.testing.assert_allclose(th_loss, ch_loss, 0.05)


def test_attn_loss():
    n_out = 7
    _eos = n_out - 1
    n_batch = 3
    label_length = numpy.array([4, 2, 3], dtype=numpy.int32)
    np_pred = numpy.random.rand(n_batch, max(
        label_length) + 1, n_out).astype(numpy.float32)
    # NOTE: 0 is only used for CTC, never appeared in attn target
    np_target = [numpy.random.randint(
        1, n_out - 1, size=ol, dtype=numpy.int32) for ol in label_length]

    eos = numpy.array([_eos], 'i')
    ys_out = [F.concat([y, eos], axis=0) for y in np_target]

    # padding for ys with -1
    # pys: utt x olen
    # NOTE: -1 is default ignore index for chainer
    pad_ys_out = F.pad_sequence(ys_out, padding=-1)
    y_all = F.reshape(np_pred, (n_batch * (max(label_length) + 1), n_out))
    ch_loss = F.softmax_cross_entropy(y_all, F.concat(pad_ys_out, axis=0))

    # NOTE: this index 0 is only for CTC not attn. so it can be ignored
    # unfortunately, torch cross_entropy does not accept out-of-bound ids
    th_ignore = 0
    th_pred = torch.from_numpy(y_all.data)
    th_target = pad_list([torch.from_numpy(t.data).long()
                          for t in ys_out], th_ignore)
    if LooseVersion(torch.__version__) < LooseVersion('1.0'):
        reduction_str = 'elementwise_mean'
    else:
        reduction_str = 'mean'
    th_loss = torch.nn.functional.cross_entropy(th_pred, th_target.view(-1),
                                                ignore_index=th_ignore,
                                                reduction=reduction_str)
    print(ch_loss)
    print(th_loss)

    # NOTE: warpctc_pytorch.CTCLoss does not normalize itself by batch-size
    # while chainer's default setting does
    loss_data = float(th_loss)
    numpy.testing.assert_allclose(loss_data, ch_loss.data, 0.05)


def test_train_acc():
    n_out = 7
    _eos = n_out - 1
    n_batch = 3
    label_length = numpy.array([4, 2, 3], dtype=numpy.int32)
    np_pred = numpy.random.rand(n_batch, max(
        label_length) + 1, n_out).astype(numpy.float32)
    # NOTE: 0 is only used for CTC, never appeared in attn target
    np_target = [numpy.random.randint(
        1, n_out - 1, size=ol, dtype=numpy.int32) for ol in label_length]

    eos = numpy.array([_eos], 'i')
    ys_out = [F.concat([y, eos], axis=0) for y in np_target]

    # padding for ys with -1
    # pys: utt x olen
    # NOTE: -1 is default ignore index for chainer
    pad_ys_out = F.pad_sequence(ys_out, padding=-1)
    y_all = F.reshape(np_pred, (n_batch * (max(label_length) + 1), n_out))
    ch_acc = F.accuracy(y_all, F.concat(pad_ys_out, axis=0), ignore_label=-1)

    # NOTE: this index 0 is only for CTC not attn. so it can be ignored
    # unfortunately, torch cross_entropy does not accept out-of-bound ids
    th_ignore = 0
    th_pred = torch.from_numpy(y_all.data)
    th_ys = [torch.from_numpy(numpy.append(t, eos)).long()
             for t in np_target]
    th_target = pad_list(th_ys, th_ignore)
    th_acc = th_accuracy(th_pred, th_target, th_ignore)

    numpy.testing.assert_allclose(ch_acc.data, th_acc)
