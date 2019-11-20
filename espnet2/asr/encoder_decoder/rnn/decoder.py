import random
from typing import Union, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet.nets.e2e_asr_common import label_smoothing_dist
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.rnn.attentions import initial_att
from espnet.nets.scorer_interface import ScorerInterface
from espnet2.utils.get_default_kwargs import get_defaut_kwargs


def build_attention_list(
        eprojs,
        dunits,
        atype: str = 'location',
        num_att: int = 1,
        num_encs: int = 1,
        aheads: int = 4,
        adim: int = 320,
        awin: int = 5,
        aconv_chans: int = 10,
        aconv_filts: int = 100,
        han_mode: bool = False,
        han_type=None,
        han_heads: int = 4,
        han_dim: int = 320,
        han_conv_chans: int = -1,
        han_conv_filts: int = 100,
        han_win: int = 5,
        ):

    att_list = torch.nn.ModuleList()
    if num_encs == 1:
        for i in range(num_att):
            att = initial_att(atype,
                              eprojs,
                              dunits,
                              aheads,
                              adim,
                              awin,
                              aconv_chans,
                              aconv_filts)
            att_list.append(att)
    elif num_encs > 1:  # no multi-speaker mode
        if han_mode:
            att = initial_att(han_type,
                              eprojs,
                              dunits,
                              han_heads,
                              han_dim,
                              han_win,
                              han_conv_chans,
                              han_conv_filts,
                              han_mode=True)
            return att
        else:
            att_list = torch.nn.ModuleList()
            for idx in range(num_encs):
                att = initial_att(atype[idx], eprojs,
                                  dunits, aheads[idx],
                                  adim[idx],
                                  awin[idx],
                                  aconv_chans[idx],
                                  aconv_filts[idx])
                att_list.append(att)
    else:
        raise ValueError("Number of encoders needs to be more than one. {}"
                         .format(num_encs))
    return att_list


class Decoder(torch.nn.Module, ScorerInterface):
    @typechecked
    def __init__(self,
                 odim: int,
                 eprojs: int = 256,
                 dtype: str = 'lstm',
                 dlayers: int = 1,
                 dunits: int = 320,
                 char_list: Union[Tuple[str, ...], List[str]] = None,
                 lsm_type: str = None,
                 lsm_weight: float = 0.,
                 sampling_probability: float = 0.0,
                 dropout: float = 0.0,
                 context_residual: bool = False,
                 replace_sos: bool = False,
                 num_encs: int = 1,
                 ignore_id: int = -1,
                 att_conf: dict = get_defaut_kwargs(build_attention_list)):
        # FIXME(kamo): The parts of num_spk and num_encs need to be refactored.
        #  Hmm... too dirty beyond saving.

        torch.nn.Module.__init__(self)
        self.dtype = dtype
        self.dunits = dunits
        self.dlayers = dlayers
        self.context_residual = context_residual
        self.embed = torch.nn.Embedding(odim, dunits)
        self.dropout_emb = torch.nn.Dropout(p=dropout)

        self.decoder = torch.nn.ModuleList()
        self.dropout_dec = torch.nn.ModuleList()
        self.decoder += [
            torch.nn.LSTMCell(dunits + eprojs, dunits)
            if self.dtype == "lstm"
            else torch.nn.GRUCell(dunits + eprojs, dunits)]
        self.dropout_dec += [torch.nn.Dropout(p=dropout)]
        for _ in range(1, self.dlayers):
            self.decoder += [
                torch.nn.LSTMCell(dunits, dunits)
                if self.dtype == "lstm" else torch.nn.GRUCell(dunits, dunits)]
            self.dropout_dec += [torch.nn.Dropout(p=dropout)]
            # NOTE: dropout is applied only for the vertical connections
            # see https://arxiv.org/pdf/1409.2329.pdf
        self.ignore_id = ignore_id

        if context_residual:
            self.output = torch.nn.Linear(dunits + eprojs, odim)
        else:
            self.output = torch.nn.Linear(dunits, odim)

        self.att_list = build_attention_list(eprojs=eprojs,
                                             dunits=dunits,
                                             **att_conf)

        self.dunits = dunits
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.char_list = char_list
        if lsm_type is not None:
            raise NotImplementedError
            self.labeldist = label_smoothing_dist(odim, lsm_type,
                                                  transcript=train_json)
        else:
            self.labeldist = None
        self.vlabeldist = None
        self.lsm_weight = lsm_weight
        self.sampling_probability = sampling_probability
        self.dropout = dropout
        self.num_encs = num_encs

        # for multilingual translation
        self.replace_sos = replace_sos

    def zero_state(self, hs_pad):
        return hs_pad.new_zeros(hs_pad.size(0), self.dunits)

    def rnn_forward(self, ey, z_list, c_list, z_prev, c_prev):
        if self.dtype == "lstm":
            z_list[0], c_list[0] = self.decoder[0](ey, (z_prev[0], c_prev[0]))
            for l in range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    self.dropout_dec[l - 1](z_list[l - 1]),
                    (z_prev[l], c_prev[l]))
        else:
            z_list[0] = self.decoder[0](ey, z_prev[0])
            for l in range(1, self.dlayers):
                z_list[l] = self.decoder[l](
                    self.dropout_dec[l - 1](z_list[l - 1]), z_prev[l])
        return z_list, c_list

    def forward(self, hs_pad, hlens, ys_in_pad, strm_idx=0):
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            hs_pad = [hs_pad]
            hlens = [hlens]

        # attention index for the attention module
        # in SPA (speaker parallel attention),
        # att_idx is used to select attention module. In other cases, it is 0.
        att_idx = min(strm_idx, len(self.att_list) - 1)

        # hlens should be list of list of integer
        hlens = [list(map(int, hlens[idx])) for idx in range(self.num_encs)]

        # get dim, length info
        batch = ys_in_pad.size(0)
        olength = ys_in_pad.size(1)

        # initialization
        c_list = [self.zero_state(hs_pad[0])]
        z_list = [self.zero_state(hs_pad[0])]
        for _ in range(1, self.dlayers):
            c_list.append(self.zero_state(hs_pad[0]))
            z_list.append(self.zero_state(hs_pad[0]))
        z_all = []
        if self.num_encs == 1:
            att_w = None
            self.att_list[att_idx].reset()  # reset pre-computation of h
        else:
            att_w_list = [None] * (self.num_encs + 1)  # atts + han
            att_c_list = [None] * self.num_encs  # atts
            for idx in range(self.num_encs + 1):
                # reset pre-computation of h in atts and han
                self.att_list[idx].reset()

        # pre-computation of embedding
        eys = self.dropout_emb(self.embed(ys_in_pad))  # utt x olen x zdim

        # loop for an output sequence
        for i in range(olength):
            if self.num_encs == 1:
                att_c, att_w = self.att_list[att_idx](
                    hs_pad[0], hlens[0],
                    self.dropout_dec[0](z_list[0]), att_w)
            else:
                for idx in range(self.num_encs):
                    att_c_list[idx], att_w_list[idx] = \
                        self.att_list[idx](hs_pad[idx], hlens[idx],
                                           self.dropout_dec[0](z_list[0]),
                                           att_w_list[idx])
                hs_pad_han = torch.stack(att_c_list, dim=1)
                hlens_han = [self.num_encs] * len(ys_in)
                att_c, att_w_list[self.num_encs] = \
                    self.att_list[self.num_encs](hs_pad_han, hlens_han,
                                                 self.dropout_dec[0](z_list[0]),
                                                 att_w_list[self.num_encs])
            if i > 0 and random.random() < self.sampling_probability:
                z_out = self.output(z_all[-1])
                z_out = np.argmax(z_out.detach().cpu(), axis=1)
                z_out = self.dropout_emb(self.embed(to_device(self, z_out)))
                ey = torch.cat((z_out, att_c), dim=1)  # utt x (zdim + hdim)
            else:
                # utt x (zdim + hdim)
                ey = torch.cat((eys[:, i, :], att_c), dim=1)
            z_list, c_list = \
                self.rnn_forward(ey, z_list, c_list, z_list, c_list)
            if self.context_residual:
                z_all.append(torch.cat((self.dropout_dec[-1](z_list[-1]),
                                        att_c), dim=-1))  # utt x (zdim + hdim)
            else:
                z_all.append(self.dropout_dec[-1](z_list[-1]))  # utt x (zdim)

        z_all = torch.stack(z_all, dim=1).view(batch * olength, -1)
        # compute loss
        y_all = self.output(z_all)
        return y_all

    def init_state(self, x):
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            x = [x]

        c_list = [self.zero_state(x[0].unsqueeze(0))]
        z_list = [self.zero_state(x[0].unsqueeze(0))]
        for _ in range(1, self.dlayers):
            c_list.append(self.zero_state(x[0].unsqueeze(0)))
            z_list.append(self.zero_state(x[0].unsqueeze(0)))
        # TODO(karita): support strm_index for `asr_mix`
        strm_index = 0
        att_idx = min(strm_index, len(self.att_list) - 1)
        if self.num_encs == 1:
            a = None
            self.att_list[att_idx].reset()  # reset pre-computation of h
        else:
            a = [None] * (self.num_encs + 1)  # atts + han
            for idx in range(self.num_encs + 1):
                # reset pre-computation of h in atts and han
                self.att_list[idx].reset()
        return dict(c_prev=c_list[:], z_prev=z_list[:], a_prev=a,
                    workspace=(att_idx, z_list, c_list))

    def score(self, yseq, state, x):
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            x = [x]

        att_idx, z_list, c_list = state["workspace"]
        vy = yseq[-1].unsqueeze(0)
        ey = self.dropout_emb(self.embed(vy))  # utt list (1) x zdim
        if self.num_encs == 1:
            att_c, att_w = self.att_list[att_idx](
                x[0].unsqueeze(0), [x[0].size(0)],
                self.dropout_dec[0](state['z_prev'][0]), state['a_prev'])
        else:
            att_w = [None] * (self.num_encs + 1)  # atts + han
            att_c_list = [None] * self.num_encs  # atts
            for idx in range(self.num_encs):
                att_c_list[idx], att_w[idx] = \
                    self.att_list[idx](x[idx].unsqueeze(0), [x[idx].size(0)],
                                       self.dropout_dec[0](state['z_prev'][0]),
                                       state['a_prev'][idx])
            h_han = torch.stack(att_c_list, dim=1)
            att_c, att_w[self.num_encs] = self.att_list[self.num_encs](
                h_han, [self.num_encs],
                self.dropout_dec[0](state['z_prev'][0]),
                state['a_prev'][self.num_encs])
        ey = torch.cat((ey, att_c), dim=1)  # utt(1) x (zdim + hdim)
        z_list, c_list = \
            self.rnn_forward(ey, z_list, c_list,
                             state['z_prev'], state['c_prev'])
        if self.context_residual:
            logits = self.output(
                torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1))
        else:
            logits = self.output(self.dropout_dec[-1](z_list[-1]))
        logp = F.log_softmax(logits, dim=1).squeeze(0)
        return logp, dict(c_prev=c_list[:], z_prev=z_list[:],
                          a_prev=att_w, workspace=(att_idx, z_list, c_list))
