import chainer
import torch

import lm_train
import lm_train_th


def transfer_lstm(ch_lstm, th_lstm):
    th_lstm.weight_ih.data = torch.from_numpy(ch_lstm.upward.W.data)
    th_lstm.bias_ih.data = torch.from_numpy(ch_lstm.upward.b.data)
    th_lstm.weight_hh.data = torch.from_numpy(ch_lstm.lateral.W.data)
    th_lstm.bias_hh.data.zero_()


def transfer_lm(ch_rnnlm, th_rnnlm):
    assert isinstance(ch_rnnlm, lm_train.RNNLM)
    assert isinstance(th_rnnlm, lm_train_th.RNNLM)
    th_rnnlm.embed.weight.data = torch.from_numpy(ch_rnnlm.embed.W.data)
    transfer_lstm(ch_rnnlm.l1, th_rnnlm.l1)
    transfer_lstm(ch_rnnlm.l2, th_rnnlm.l2)
    th_rnnlm.lo.weight.data = torch.from_numpy(ch_rnnlm.lo.W.data)
    th_rnnlm.lo.bias.data = torch.from_numpy(ch_rnnlm.lo.b.data)


def test_lm():
    n_vocab = 52
    rnnlm_ch = lm_train.ClassifierWithState(lm_train.RNNLM(n_vocab, 10))
    rnnlm_th = lm_train_th.ClassifierWithState(lm_train_th.RNNLM(n_vocab, 10))
    transfer_lm(rnnlm_ch.predictor, rnnlm_th.predictor)
    import numpy
    # test transfer function
    numpy.testing.assert_equal(rnnlm_ch.predictor.embed.W.data, rnnlm_th.predictor.embed.weight.data.numpy())
    numpy.testing.assert_equal(rnnlm_ch.predictor.l1.upward.b.data, rnnlm_th.predictor.l1.bias_ih.data.numpy())
    numpy.testing.assert_equal(rnnlm_ch.predictor.l1.upward.W.data, rnnlm_th.predictor.l1.weight_ih.data.numpy())
    numpy.testing.assert_equal(rnnlm_ch.predictor.l1.lateral.W.data, rnnlm_th.predictor.l1.weight_hh.data.numpy())
    numpy.testing.assert_equal(rnnlm_ch.predictor.l2.upward.b.data, rnnlm_th.predictor.l2.bias_ih.data.numpy())
    numpy.testing.assert_equal(rnnlm_ch.predictor.l2.upward.W.data, rnnlm_th.predictor.l2.weight_ih.data.numpy())
    numpy.testing.assert_equal(rnnlm_ch.predictor.l2.lateral.W.data, rnnlm_th.predictor.l2.weight_hh.data.numpy())
    numpy.testing.assert_equal(rnnlm_ch.predictor.lo.b.data, rnnlm_th.predictor.lo.bias.data.numpy())
    numpy.testing.assert_equal(rnnlm_ch.predictor.lo.W.data, rnnlm_th.predictor.lo.weight.data.numpy())

    # test prediction equality
    x = torch.autograd.Variable(
        torch.from_numpy(numpy.random.randint(n_vocab, size=(5))),
        volatile=True).long()
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        rnnlm_th.predictor.eval()
        state = {
            'c1': rnnlm_th.predictor.zero_state(x.size(0)),
            'h1': rnnlm_th.predictor.zero_state(x.size(0)),
            'c2': rnnlm_th.predictor.zero_state(x.size(0)),
            'h2': rnnlm_th.predictor.zero_state(x.size(0))
        }
        state_th, y_th = rnnlm_th.predictor(state, x)
        state = {
            'c1': numpy.zeros((x.size(0), 10), dtype=numpy.float32),
            'h1': numpy.zeros((x.size(0), 10), dtype=numpy.float32),
            'c2': numpy.zeros((x.size(0), 10), dtype=numpy.float32),
            'h2': numpy.zeros((x.size(0), 10), dtype=numpy.float32)
        }
        state_ch, y_ch = rnnlm_ch.predictor(state, x.data.numpy())
        for k in state_ch.iterkeys():
            numpy.testing.assert_allclose(state_th[k][0, :10].data.numpy(), state_ch[k][0, :10].data)
        numpy.testing.assert_allclose(y_th.data.numpy(), y_ch.data)
