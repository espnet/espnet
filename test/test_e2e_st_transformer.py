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
    T = importlib.import_module('espnet.nets.{}_backend.e2e_st_transformer'.format(module))
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
        report_cer=False,
        report_wer=False,
        mtlalpha=0.0,  # for CTC-ASR
        lsm_weight=0.001,
        char_list=['<blank>', 'a', 'e', 'i', 'o', 'u'],
        ctc_type="warpctc",
        asr_weight=0.0,
        mt_weight=0.0,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def prepare(backend, args):
    idim = 40
    odim = 5
    T = importlib.import_module('espnet.nets.{}_backend.e2e_st_transformer'.format(backend))

    model = T.E2E(idim, odim, args)
    batchsize = 5
    if backend == 'pytorch':
        x = torch.randn(batchsize, 40, idim)
    else:
        x = numpy.random.randn(batchsize, 40, idim).astype(numpy.float32)
    ilens = [40, 30, 20, 15, 10]
    n_token = odim - 1
    if backend == 'pytorch':
        y_src = (torch.rand(batchsize, 10) * n_token % n_token).long()
        y_tgt = (torch.rand(batchsize, 11) * n_token % n_token).long()
    else:
        y_src = (numpy.random.rand(batchsize, 10) * n_token % n_token).astype(numpy.int32)
        y_tgt = (numpy.random.rand(batchsize, 11) * n_token % n_token).astype(numpy.int32)
    olens = [3, 9, 10, 2, 3]
    for i in range(batchsize):
        x[i, ilens[i]:] = -1
        y_tgt[i, olens[i]:] = model.ignore_id
        y_src[i, olens[i]:] = model.ignore_id

    data = []
    for i in range(batchsize):
        data.append(("utt%d" % i, {
            "input": [{"shape": [ilens[i], idim]}],
            "output": [{"shape": [olens[i]]}]
        }))
    if backend == 'pytorch':
        return model, x, torch.tensor(ilens), y_tgt, y_src, data
    else:
        return model, x, ilens, y_tgt, y_src, data


@pytest.mark.parametrize("module", ["pytorch"])
def test_transformer_mask(module):
    args = make_arg()
    model, x, ilens, y_tgt, y_src, data = prepare(module, args)
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
        ('pytorch', {'asr_weight': 0.0, 'mt_weight': 0.0}),  # pure E2E-ST
        ('pytorch', {'asr_weight': 0.1, 'mtlalpha': 0.0, 'mt_weight': 0.0}),  # MTL w/ attention ASR
        ('pytorch', {'asr_weight': 0.1, 'mtlalpha': 0.0, 'mt_weight': 0.1}),  # MTL w/ attention ASR + MT
        ('pytorch', {'asr_weight': 0.1, 'mtlalpha': 1.0, 'mt_weight': 0.0}),  # MTL w/ CTC ASR
        ('pytorch', {'asr_weight': 0.1, 'mtlalpha': 1.0, 'ctc_type': "builtin"}),
        ('pytorch', {'asr_weight': 0.1, 'mtlalpha': 1.0, 'report_cer': True}),
        ('pytorch', {'asr_weight': 0.1, 'mtlalpha': 1.0, 'report_wer': True}),
        ('pytorch', {'asr_weight': 0.1, 'mtlalpha': 1.0, 'report_cer': True, 'report_wer': True}),
        ('pytorch', {'asr_weight': 0.1, 'mtlalpha': 1.0, 'mt_weight': 0.1}),  # MTL w/ CTC ASR + MT
        ('pytorch', {'asr_weight': 0.1, 'mtlalpha': 0.5, 'mt_weight': 0.0}),  # MTL w/ attention ASR + CTC ASR
        ('pytorch', {'asr_weight': 0.1, 'mtlalpha': 0.5, 'mt_weight': 0.1}),  # MTL w/ attention ASR + CTC ASR + MT
    ]
)
def test_transformer_trainable_and_decodable(module, model_dict):
    args = make_arg(**model_dict)
    model, x, ilens, y_tgt, y_src, data = prepare(module, args)

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
        loss = model(x, ilens, y_tgt, y_src)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # test attention plot
        attn_dict = model.calculate_all_attentions(x[0:1], ilens[0:1], y_tgt[0:1], y_src[0:1])
        from espnet.nets.pytorch_backend.transformer import plot
        plot.plot_multi_head_attention(data, attn_dict, "/tmp/espnet-test")

        # test decodable
        with torch.no_grad():
            nbest = model.translate(x[0, :ilens[0]].numpy(), trans_args, args.char_list)
            print(y_tgt[0])
            print(nbest[0]["yseq"][1:-1])
    else:
        raise NotImplementedError


if __name__ == "__main__":
    run_transformer_copy()
