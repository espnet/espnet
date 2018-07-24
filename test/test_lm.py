import chainer
import torch

from e2e_asr_attctc_th import torch_is_old
import lm_chainer
import lm_pytorch


def transfer_lstm(ch_lstm, th_lstm):
    ch_lstm.upward.W.data[:] = 1
    th_lstm.weight_ih.data[:] = torch.from_numpy(ch_lstm.upward.W.data)
    ch_lstm.upward.b.data[:] = 1
    th_lstm.bias_hh.data[:] = torch.from_numpy(ch_lstm.upward.b.data)
    # NOTE: only lateral weight can directly transfer
    # rest of the weights and biases have quite different placements
    th_lstm.weight_hh.data[:] = torch.from_numpy(ch_lstm.lateral.W.data)
    th_lstm.bias_ih.data.zero_()


def transfer_lm(ch_rnnlm, th_rnnlm):
    assert isinstance(ch_rnnlm, lm_chainer.RNNLM)
    assert isinstance(th_rnnlm, lm_pytorch.RNNLM)
    th_rnnlm.embed.weight.data = torch.from_numpy(ch_rnnlm.embed.W.data)
    transfer_lstm(ch_rnnlm.l1, th_rnnlm.l1)
    transfer_lstm(ch_rnnlm.l2, th_rnnlm.l2)
    th_rnnlm.lo.weight.data = torch.from_numpy(ch_rnnlm.lo.W.data)
    th_rnnlm.lo.bias.data = torch.from_numpy(ch_rnnlm.lo.b.data)


def test_lm():
    n_vocab = 3
    n_units = 2
    batchsize = 5
    rnnlm_ch = lm_chainer.ClassifierWithState(lm_chainer.RNNLM(n_vocab, n_units))
    rnnlm_th = lm_pytorch.ClassifierWithState(lm_pytorch.RNNLM(n_vocab, n_units))
    transfer_lm(rnnlm_ch.predictor, rnnlm_th.predictor)
    import numpy
    # TODO(karita) implement weight transfer
    # numpy.testing.assert_equal(rnnlm_ch.predictor.embed.W.data, rnnlm_th.predictor.embed.weight.data.numpy())
    # numpy.testing.assert_equal(rnnlm_ch.predictor.l1.upward.b.data, rnnlm_th.predictor.l1.bias_ih.data.numpy())
    # numpy.testing.assert_equal(rnnlm_ch.predictor.l1.upward.W.data, rnnlm_th.predictor.l1.weight_ih.data.numpy())
    # numpy.testing.assert_equal(rnnlm_ch.predictor.l1.lateral.W.data, rnnlm_th.predictor.l1.weight_hh.data.numpy())
    # numpy.testing.assert_equal(rnnlm_ch.predictor.l2.upward.b.data, rnnlm_th.predictor.l2.bias_ih.data.numpy())
    # numpy.testing.assert_equal(rnnlm_ch.predictor.l2.upward.W.data, rnnlm_th.predictor.l2.weight_ih.data.numpy())
    # numpy.testing.assert_equal(rnnlm_ch.predictor.l2.lateral.W.data, rnnlm_th.predictor.l2.weight_hh.data.numpy())
    # numpy.testing.assert_equal(rnnlm_ch.predictor.lo.b.data, rnnlm_th.predictor.lo.bias.data.numpy())
    # numpy.testing.assert_equal(rnnlm_ch.predictor.lo.W.data, rnnlm_th.predictor.lo.weight.data.numpy())

    # test prediction equality
    if torch_is_old:
        x = torch.autograd.Variable(
            torch.from_numpy(numpy.random.randint(n_vocab, size=(batchsize))),
            volatile=True).long()
    else:
        x = torch.from_numpy(numpy.random.randint(n_vocab, size=(batchsize))).long()
        torch.set_grad_enabled(False)
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        rnnlm_th.predictor.eval()
        state_th, y_th = rnnlm_th.predictor(None, x.long())
        state_ch, y_ch = rnnlm_ch.predictor(None, x.data.numpy())
        for k in state_ch.keys():
            print(k)
            print(state_th[k].data.numpy())
            print(state_ch[k].data)
            numpy.testing.assert_allclose(state_th[k].data.numpy(), state_ch[k].data, 1e-5)
        print("y")
        print(y_th.data.numpy())
        print(y_ch.data)
        numpy.testing.assert_allclose(y_th.data.numpy(), y_ch.data, 1e-5)
