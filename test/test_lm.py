import chainer
import torch

import espnet.lm.chainer_backend.lm as lm_chainer
import espnet.lm.pytorch_backend.lm as lm_pytorch


def transfer_gru(ch_gru, th_gru):
    pass  # TODO(chainer_pytorch_expert)


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
    if th_rnnlm.typ == "lstm":
        for n in range(ch_rnnlm.n_layers):
            transfer_lstm(getattr(ch_rnnlm, ch_rnnlm.typ)[n], getattr(th_rnnlm, th_rnnlm.typ)[n])
    else:
        for n in range(ch_rnnlm.n_layers):
            transfer_gru(getattr(ch_rnnlm, ch_rnnlm.typ)[n], getattr(th_rnnlm, th_rnnlm.typ)[n])
    th_rnnlm.lo.weight.data = torch.from_numpy(ch_rnnlm.lo.W.data)
    th_rnnlm.lo.bias.data = torch.from_numpy(ch_rnnlm.lo.b.data)


def test_lm():
    n_vocab = 3
    n_layers = 2
    n_units = 2
    batchsize = 5
    for typ in ["lstm"]:  # TODO(anyone) gru
        rnnlm_ch = lm_chainer.ClassifierWithState(lm_chainer.RNNLM(n_vocab, n_layers, n_units, typ=typ))
        rnnlm_th = lm_pytorch.ClassifierWithState(lm_pytorch.RNNLM(n_vocab, n_layers, n_units, typ=typ))
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
        x = torch.from_numpy(numpy.random.randint(n_vocab, size=batchsize)).long()
        with torch.no_grad(), chainer.no_backprop_mode(), chainer.using_config('train', False):
            rnnlm_th.predictor.eval()
            state_th, y_th = rnnlm_th.predictor(None, x.long())
            state_ch, y_ch = rnnlm_ch.predictor(None, x.data.numpy())
            for k in state_ch.keys():
                for n in range(len(state_th[k])):
                    print(k, n)
                    print(state_th[k][n].data.numpy())
                    print(state_ch[k][n].data)
                    numpy.testing.assert_allclose(state_th[k][n].data.numpy(), state_ch[k][n].data, 1e-5)
            print("y")
            print(y_th.data.numpy())
            print(y_ch.data)
            numpy.testing.assert_allclose(y_th.data.numpy(), y_ch.data, 1e-5)
