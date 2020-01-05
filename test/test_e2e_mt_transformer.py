# coding: utf-8

# Copyright 2019 Hirofumi Inaguma
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import importlib
import logging
import numpy
import pytest
import torch

from test.test_e2e_asr_transformer import run_transformer_copy
from test.test_e2e_asr_transformer import subsequent_mask


logging.basicConfig(
    level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')


@pytest.mark.parametrize("module", ["pytorch"])
def test_mask(module):
    T = importlib.import_module('espnet.nets.{}_backend.e2e_mt_transformer'.format(module))
    m = T.subsequent_mask(3)
    print(m)
    print(subsequent_mask(3))
    assert (m.unsqueeze(0) == subsequent_mask(3)).all()


def make_arg(**kwargs):
    defaults = dict(
        adim=16,
        aheads=2,
        dropout_rate=0.0,
        transformer_attn_dropout_rate=None,
        elayers=2,
        eunits=16,
        dlayers=2,
        dunits=16,
        sym_space="<space>",
        sym_blank="<blank>",
        transformer_init="pytorch",
        transformer_input_layer="conv2d",
        transformer_length_normalized_loss=True,
        report_bleu=False,
        lsm_weight=0.001,
        char_list=['<blank>', 'a', 'e', 'i', 'o', 'u'],
        tie_src_tgt_embedding=False,
        tie_classifier=False,
        multilingual=False,
        replace_sos=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def prepare(backend, args):
    idim = 5
    odim = 5
    T = importlib.import_module('espnet.nets.{}_backend.e2e_mt_transformer'.format(backend))

    model = T.E2E(idim, odim, args)
    batchsize = 5
    n_token = odim - 1
    if backend == 'pytorch':
        y_src = (torch.randn(batchsize, 10) * n_token % n_token).long() + 1
        y_tgt = (torch.randn(batchsize, 11) * n_token % n_token).long() + 1
        # NOTE: + 1 to avoid to assign idx:0
    else:
        y_src = numpy.random.randn(batchsize, 10, idim).astype(numpy.int64) + 1
        y_tgt = numpy.random.randn(batchsize, 11, idim).astype(numpy.int64) + 1
    ilens = [3, 9, 10, 2, 3]
    olens = [4, 10, 11, 3, 4]
    for i in range(batchsize):
        y_src[i, ilens[i]:] = model.pad
        y_tgt[i, olens[i]:] = model.ignore_id

    data = []
    for i in range(batchsize):
        data.append(("utt%d" % i, {
            "input": [{"shape": [ilens[i]]}],
            "output": [{"shape": [olens[i]]}]
        }))
    if backend == 'pytorch':
        return model, y_src, torch.tensor(ilens), y_tgt, data
    else:
        return model, y_src, ilens, y_tgt, data


@pytest.mark.parametrize("module", ["pytorch"])
def test_transformer_mask(module):
    args = make_arg()
    model, y_src, ilens, y_tgt, data = prepare(module, args)
    from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
    from espnet.nets.pytorch_backend.transformer.mask import target_mask
    yi, yo = add_sos_eos(y_tgt, model.sos, model.eos, model.ignore_id)
    y_mask = target_mask(yi, model.ignore_id)
    y_tgt = model.decoder.embed(yi)
    y_tgt[0, 3:] = float("nan")
    a = model.decoder.decoders[0].self_attn
    a(y_tgt, y_tgt, y_tgt, y_mask)
    assert not numpy.isnan(a.attn[0, :, :3, :3].detach().numpy()).any()


@pytest.mark.parametrize(
    "module, model_dict", [
        ('pytorch', {}),
        ('pytorch', {'report_bleu': True}),
        ('pytorch', {'tie_src_tgt_embedding': True}),
        ('pytorch', {'tie_classifier': True}),
        ('pytorch', {'tie_src_tgt_embedding': True, 'tie_classifier': True}),
    ]
)
def test_transformer_trainable_and_decodable(module, model_dict):
    args = make_arg(**model_dict)
    model, y_src, ilens, y_tgt, data = prepare(module, args)

    # test beam search
    trans_args = argparse.Namespace(
        beam_size=1,
        penalty=0.0,
        ctc_weight=0.0,
        maxlenratio=1.0,
        lm_weight=0,
        minlenratio=0,
        nbest=1,
        tgt_lang=False,
    )
    if module == "pytorch":
        # test trainable
        optim = torch.optim.Adam(model.parameters(), 0.01)
        loss = model(y_src, ilens, y_tgt)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # test attention plot
        attn_dict = model.calculate_all_attentions(y_src[0:1], ilens[0:1], y_tgt[0:1])
        from espnet.nets.pytorch_backend.transformer import plot
        plot.plot_multi_head_attention(data, attn_dict, "/tmp/espnet-test")

        # test decodable
        with torch.no_grad():
            nbest = model.translate([y_src[0, :ilens[0]].numpy()], trans_args, args.char_list)
            print(y_tgt[0])
            print(nbest[0]["yseq"][1:-1])
    else:
        raise NotImplementedError


if __name__ == "__main__":
    run_transformer_copy()
