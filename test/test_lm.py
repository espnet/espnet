import chainer
import numpy
import pytest
import torch

import espnet.lm.chainer_backend.lm as lm_chainer
from espnet.nets.beam_search import beam_search
from espnet.nets.lm_interface import dynamic_import_lm
import espnet.nets.pytorch_backend.lm.default as lm_pytorch
from espnet.nets.scorers.length_bonus import LengthBonus

from test.test_beam_search import prepare
from test.test_beam_search import rnn_args


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
            transfer_lstm(ch_rnnlm.rnn[n], th_rnnlm.rnn[n])
    else:
        assert False
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
            numpy.testing.assert_allclose(y_th.data.numpy(), y_ch.data, 1e-5)


@pytest.mark.parametrize(
    "lm_name, lm_args, device, dtype", [
        (nn, args, device, dtype)
        for nn, args in (
            ("default", dict(type="lstm", layer=2, unit=2, dropout_rate=0.5)),
            ("default", dict(type="gru", layer=2, unit=2, dropout_rate=0.5)),
            ("seq_rnn", dict(type="lstm", layer=2, unit=2, dropout_rate=0.5)),
            ("seq_rnn", dict(type="gru", layer=2, unit=2, dropout_rate=0.5)),
            ("transformer", dict(layer=2, unit=2, att_unit=2, head=2, dropout_rate=0.5, embed_unit=3)),
            ("transformer", dict(layer=2, unit=2, att_unit=2, head=2, dropout_rate=0.5, pos_enc="none", embed_unit=3))
        )
        for device in ("cpu", "cuda")
        for dtype in ("float16", "float32", "float64")
    ])
def test_lm_trainable_and_decodable(lm_name, lm_args, device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no cuda device is available")
    if device == "cpu" and dtype == "float16":
        pytest.skip("cpu float16 implementation is not available in pytorch yet")

    dtype = getattr(torch, dtype)
    model, x, ilens, y, data, train_args = prepare("rnn", rnn_args)
    char_list = train_args.char_list
    n_vocab = len(char_list)
    lm = dynamic_import_lm(lm_name, backend="pytorch").build(n_vocab, **lm_args)
    lm.to(device=device, dtype=dtype)

    # test trainable
    a = torch.randint(1, n_vocab, (3, 2), device=device)
    b = torch.randint(1, n_vocab, (3, 2), device=device)
    loss, logp, count = lm(a, b)
    loss.backward()
    for p in lm.parameters():
        assert p.grad is not None

    # test decodable
    model.to(device=device, dtype=dtype).eval()
    lm.eval()

    scorers = model.scorers()
    scorers["lm"] = lm
    scorers["length_bonus"] = LengthBonus(len(char_list))
    weights = dict(decoder=1.0, lm=1.0, length_bonus=1.0)
    with torch.no_grad():
        feat = x[0, :ilens[0]].to(device=device, dtype=dtype)
        enc = model.encode(feat)
        beam_size = 3
        result = beam_search(
            x=enc,
            sos=model.sos,
            eos=model.eos,
            beam_size=beam_size,
            vocab_size=len(train_args.char_list),
            weights=weights,
            scorers=scorers,
            token_list=train_args.char_list
        )
    assert len(result) >= beam_size
