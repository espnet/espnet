# coding: utf-8

# Copyright 2019 Hirofumi Inaguma
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import argparse

import pytest
import torch

from espnet.nets.pytorch_backend.e2e_st_conformer import E2E
from espnet.nets.pytorch_backend.transformer import plot


def make_arg(**kwargs):
    defaults = dict(
        adim=2,
        aheads=1,
        dropout_rate=0.0,
        transformer_attn_dropout_rate=None,
        elayers=1,
        eunits=2,
        dlayers=1,
        dunits=2,
        sym_space="<space>",
        sym_blank="<blank>",
        transformer_decoder_selfattn_layer_type="selfattn",
        transformer_encoder_pos_enc_layer_type="rel_pos",
        transformer_encoder_selfattn_layer_type="rel_selfattn",
        macaron_style=True,
        use_cnn_module=True,
        cnn_module_kernel=3,
        transformer_init="pytorch",
        transformer_input_layer="conv2d",
        transformer_length_normalized_loss=True,
        report_bleu=False,
        report_cer=False,
        report_wer=False,
        mtlalpha=0.0,  # for CTC-ASR
        lsm_weight=0.001,
        char_list=["<blank>", "a", "e", "i", "o", "u"],
        ctc_type="builtin",
        asr_weight=0.0,
        mt_weight=0.0,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def prepare(args):
    idim = 10
    odim = 5
    model = E2E(idim, odim, args)
    batchsize = 2
    ilens = [10, 9]
    olens = [3, 4]
    n_token = odim - 1
    x = torch.randn(batchsize, max(ilens), idim)
    y_src = (torch.rand(batchsize, max(olens)) * n_token % n_token).long()
    y_tgt = (torch.rand(batchsize, max(olens)) * n_token % n_token).long()
    for i in range(batchsize):
        x[i, ilens[i] :] = -1
        y_tgt[i, olens[i] :] = model.ignore_id
        y_src[i, olens[i] :] = model.ignore_id

    data = {}
    uttid_list = []
    for i in range(batchsize):
        data["utt%d" % i] = {
            "input": [{"shape": [ilens[i], idim]}],
            "output": [{"shape": [olens[i]]}],
        }
        uttid_list.append("utt%d" % i)

    return model, x, torch.tensor(ilens), y_tgt, y_src, data, uttid_list


conformer_mcnn_args = dict(
    transformer_encoder_pos_enc_layer_type="rel_pos",
    transformer_encoder_selfattn_layer_type="rel_selfattn",
    macaron_style=True,
    use_cnn_module=False,
)

conformer_mcnn_mmacaron_args = dict(
    transformer_encoder_pos_enc_layer_type="rel_pos",
    transformer_encoder_selfattn_layer_type="rel_selfattn",
    macaron_style=False,
    use_cnn_module=False,
)

conformer_mcnn_mmacaron_mrelattn_args = dict(
    transformer_encoder_pos_enc_layer_type="abs_pos",
    transformer_encoder_selfattn_layer_type="selfattn",
    macaron_style=False,
    use_cnn_module=False,
)


def _savefn(*args, **kwargs):
    return


@pytest.mark.parametrize(
    "model_dict",
    [
        {},
        conformer_mcnn_args,
        conformer_mcnn_mmacaron_args,
        conformer_mcnn_mmacaron_mrelattn_args,
    ],
)
def test_transformer_trainable_and_decodable(model_dict):
    args = make_arg(**model_dict)
    model, x, ilens, y_tgt, y_src, data, uttid_list = prepare(args)

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
    loss = model(x, ilens, y_tgt, y_src)
    optim.zero_grad()
    loss.backward()
    optim.step()

    # test attention plot
    attn_dict = model.calculate_all_attentions(
        x[0:1], ilens[0:1], y_tgt[0:1], y_src[0:1]
    )
    plot.plot_multi_head_attention(data, uttid_list, attn_dict, "", savefn=_savefn)

    # test CTC plot
    ctc_probs = model.calculate_all_ctc_probs(
        x[0:1], ilens[0:1], y_tgt[0:1], y_src[0:1]
    )
    if args.asr_weight > 0 and args.mtlalpha > 0:
        print(ctc_probs.shape)
    else:
        assert ctc_probs is None

    # test decodable
    with torch.no_grad():
        nbest = model.translate(x[0, : ilens[0]].numpy(), trans_args, args.char_list)
        print(y_tgt[0])
        print(nbest[0]["yseq"][1:-1])
