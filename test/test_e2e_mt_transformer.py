# coding: utf-8

# Copyright 2019 Hirofumi Inaguma
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse

import pytest
import torch

from espnet.nets.pytorch_backend.e2e_mt_transformer import E2E
from espnet.nets.pytorch_backend.transformer import plot


def make_arg(**kwargs):
    defaults = dict(
        adim=2,
        aheads=1,
        dropout_rate=0.0,
        transformer_attn_dropout_rate=None,
        elayers=2,
        eunits=2,
        dlayers=2,
        dunits=2,
        sym_space="<space>",
        sym_blank="<blank>",
        transformer_decoder_selfattn_layer_type="selfattn",
        transformer_encoder_selfattn_layer_type="selfattn",
        transformer_init="pytorch",
        transformer_input_layer="linear",
        transformer_length_normalized_loss=True,
        report_bleu=False,
        lsm_weight=0.001,
        char_list=["<blank>", "a", "e", "i", "o", "u"],
        tie_src_tgt_embedding=False,
        tie_classifier=False,
        multilingual=False,
        replace_sos=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def prepare(args):
    idim = 5
    odim = 5
    model = E2E(idim, odim, args)
    batchsize = 2
    n_token = odim - 1
    y_src = (torch.randn(batchsize, 4) * n_token % n_token).long() + 1
    y_tgt = (torch.randn(batchsize, 4) * n_token % n_token).long() + 1
    # NOTE: + 1 to avoid to assign idx:0
    ilens = [3, 4]
    olens = [4, 3]
    for i in range(batchsize):
        y_src[i, ilens[i] :] = model.pad
        y_tgt[i, olens[i] :] = model.ignore_id

    data = {}
    uttid_list = []
    for i in range(batchsize):
        data["utt%d" % i] = {
            "input": [{"shape": [ilens[i]]}],
            "output": [{"shape": [olens[i]]}],
        }
        uttid_list.append("utt%d" % i)

    return model, y_src, torch.tensor(ilens), y_tgt, data, uttid_list


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


def _savefn(*args, **kwargs):
    return


@pytest.mark.parametrize(
    "model_dict",
    [
        {},
        ldconv_lconv_args,
        ldconv_dconv_args,
        {"report_bleu": True},
        {"tie_src_tgt_embedding": True},
        {"tie_classifier": True},
        {"tie_src_tgt_embedding": True, "tie_classifier": True},
    ],
)
def test_transformer_trainable_and_decodable(model_dict):
    args = make_arg(**model_dict)
    model, y_src, ilens, y_tgt, data, uttid_list = prepare(args)

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
    # test trainable
    optim = torch.optim.Adam(model.parameters(), 0.01)
    loss = model(y_src, ilens, y_tgt)
    optim.zero_grad()
    loss.backward()
    optim.step()

    # test attention plot
    attn_dict = model.calculate_all_attentions(y_src[0:1], ilens[0:1], y_tgt[0:1])
    plot.plot_multi_head_attention(data, uttid_list, attn_dict, "", savefn=_savefn)

    # test decodable
    with torch.no_grad():
        nbest = model.translate(
            [y_src[0, : ilens[0]].numpy()], trans_args, args.char_list
        )
        print(y_tgt[0])
        print(nbest[0]["yseq"][1:-1])
