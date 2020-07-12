"""Transducer and transducer with attention implementation for training and decoding."""

import numpy as np
import six

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.rnn.attentions import att_to_numpy

from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device

from espnet.nets.pytorch_backend.transducer.utils import get_beam_lm_states
from espnet.nets.pytorch_backend.transducer.utils import get_idx_lm_state
from espnet.nets.pytorch_backend.transducer.utils import is_prefix
from espnet.nets.pytorch_backend.transducer.utils import substract


class DecoderRNNT(torch.nn.Module):
    """RNN-T Decoder module.

    Args:
        eprojs (int): # encoder projection units
        odim (int): dimension of outputs
        dtype (str): gru or lstm
        dlayers (int): # prediction layers
        dunits (int): # prediction units
        blank (int): blank symbol id
        embed_dim (init): dimension of embeddings
        joint_dim (int): dimension of joint space
        dropout (float): dropout rate
        dropout_embed (float): embedding dropout rate

    """

    def __init__(
        self,
        eprojs,
        odim,
        dtype,
        dlayers,
        dunits,
        blank,
        embed_dim,
        joint_dim,
        dropout=0.0,
        dropout_embed=0.0,
    ):
        """Transducer initializer."""
        super(DecoderRNNT, self).__init__()

        self.embed = torch.nn.Embedding(odim, embed_dim, padding_idx=blank)
        self.dropout_embed = torch.nn.Dropout(p=dropout_embed)

        if dtype == "lstm":
            dec_net = torch.nn.LSTMCell
        else:
            dec_net = torch.nn.GRUCell

        self.decoder = torch.nn.ModuleList([dec_net(embed_dim, dunits)])
        self.dropout_dec = torch.nn.ModuleList([torch.nn.Dropout(p=dropout)])

        for _ in six.moves.range(1, dlayers):
            self.decoder += [dec_net(dunits, dunits)]
            self.dropout_dec += [torch.nn.Dropout(p=dropout)]

        self.lin_enc = torch.nn.Linear(eprojs, joint_dim)
        self.lin_dec = torch.nn.Linear(dunits, joint_dim, bias=False)
        self.lin_out = torch.nn.Linear(joint_dim, odim)

        self.dlayers = dlayers
        self.dunits = dunits
        self.dtype = dtype
        self.embed_dim = embed_dim
        self.joint_dim = joint_dim
        self.odim = odim

        self.ignore_id = -1
        self.blank = blank

    def zero_state(self, ey):
        """Initialize decoder states.

        Args:
            ey (torch.Tensor): batch of input features (B, Emb_dim)

        Returns:
            (list): list of L zero-init hidden and cell state (B, Hdec)

        """
        z_list = [ey.new_zeros(ey.size(0), self.dunits)]
        c_list = [ey.new_zeros(ey.size(0), self.dunits)]

        for _ in six.moves.range(1, self.dlayers):
            z_list.append(ey.new_zeros(ey.size(0), self.dunits))
            c_list.append(ey.new_zeros(ey.size(0), self.dunits))

        return (z_list, c_list)

    def rnn_forward(self, ey, dstate):
        """RNN forward.

        Args:
            ey (torch.Tensor): batch of input features (B, Emb_dim)
            dstate (list): list of L input hidden and cell state (B, Hdec)

        Returns:
            output (torch.Tensor): batch of output features (B, Hdec)
            dstate (list): list of L output hidden and cell state (B, Hdec)

        """
        if dstate is None:
            z_prev, c_prev = self.zero_state(ey)
        else:
            z_prev, c_prev = dstate

        z_list, c_list = self.zero_state(ey)

        if self.dtype == "lstm":
            z_list[0], c_list[0] = self.decoder[0](ey, (z_prev[0], c_prev[0]))

            for i in six.moves.range(1, self.dlayers):
                z_list[i], c_list[i] = self.decoder[i](
                    self.dropout_dec[i - 1](z_list[i - 1]), (z_prev[i], c_prev[i])
                )
        else:
            z_list[0] = self.decoder[0](ey, z_prev[0])

            for i in six.moves.range(1, self.dlayers):
                z_list[i] = self.decoder[i](
                    self.dropout_dec[i - 1](z_list[i - 1]), z_prev[i]
                )
        y = self.dropout_dec[-1](z_list[-1])

        return y, (z_list, c_list)

    def joint(self, h_enc, h_dec):
        """Joint computation of z.

        Args:
            h_enc (torch.Tensor): batch of expanded hidden state (B, T, 1, Henc)
            h_dec (torch.Tensor): batch of expanded hidden state (B, 1, U, Hdec)

        Returns:
            z (torch.Tensor): output (B, T, U, odim)

        """
        z = torch.tanh(self.lin_enc(h_enc) + self.lin_dec(h_dec))
        z = self.lin_out(z)

        return z

    def forward(self, hs_pad, ys_in_pad, hlens=None):
        """Forward function for transducer.

        Args:
            hs_pad (torch.Tensor):
                batch of padded hidden state sequences (B, Tmax, D)
            ys_in_pad (torch.Tensor):
                batch of padded character id sequence tensor (B, Lmax+1)

        Returns:
            z (torch.Tensor): output (B, T, U, odim)

        """
        olength = ys_in_pad.size(1)

        z_list, c_list = self.zero_state(hs_pad)
        eys = self.dropout_embed(self.embed(ys_in_pad))

        z_all = []
        for i in six.moves.range(olength):
            y, (z_list, c_list) = self.rnn_forward(eys[:, i, :], (z_list, c_list))
            z_all.append(y)
        h_dec = torch.stack(z_all, dim=1)

        h_enc = hs_pad.unsqueeze(2)
        h_dec = h_dec.unsqueeze(1)

        z = self.joint(h_enc, h_dec)

        return z

    def recognize(self, h, recog_args):
        """Greedy search implementation.

        Args:
            h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
            recog_args (Namespace): argument Namespace containing options

        Returns:
            hyp (list of dicts): 1-best decoding results

        """
        z_list, c_list = self.zero_state(h.unsqueeze(0))
        ey = to_device(self, torch.zeros((1, self.embed_dim)))

        hyp = {"score": 0.0, "yseq": [self.blank]}

        y, (z_list, c_list) = self.rnn_forward(ey, (z_list, c_list))

        for hi in h:
            ytu = F.log_softmax(self.joint(hi, y[0]), dim=0)
            logp, pred = torch.max(ytu, dim=0)

            if pred != self.blank:
                hyp["yseq"].append(int(pred))
                hyp["score"] += float(logp)

                eys = to_device(
                    self, torch.full((1, 1), hyp["yseq"][-1], dtype=torch.long)
                )
                ey = self.embed(eys)

                y, (z_list, c_list) = self.rnn_forward(ey[0], (z_list, c_list))

        return [hyp]

    def recognize_beam_default(self, h, recog_args, rnnlm=None):
        """Beam search implementation.

        Args:
            h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
            recog_args (Namespace): argument Namespace containing options
            rnnlm (torch.nn.Module): language module

        Returns:
            nbest_hyps (list of dicts): n-best decoding results

        """
        beam = recog_args.beam_size
        k_range = min(beam, self.odim)

        nbest = recog_args.nbest
        normscore = recog_args.score_norm_transducer

        z_list, c_list = self.zero_state(h.unsqueeze(0))

        kept_hyps = [
            {
                "score": 0.0,
                "yseq": [self.blank],
                "z_prev": z_list,
                "c_prev": c_list,
                "lm_state": None,
            }
        ]

        for hi in h:
            hyps = kept_hyps
            kept_hyps = []

            while True:
                new_hyp = max(hyps, key=lambda x: x["score"])
                hyps.remove(new_hyp)

                vy = to_device(
                    self, torch.full((1, 1), new_hyp["yseq"][-1], dtype=torch.long)
                )
                ey = self.embed(vy)

                y, (z_list, c_list) = self.rnn_forward(
                    ey[0], (new_hyp["z_prev"], new_hyp["c_prev"])
                )

                ytu = F.log_softmax(self.joint(hi, y[0]), dim=0)

                if rnnlm:
                    rnnlm_state, rnnlm_scores = rnnlm.predict(
                        new_hyp["lm_state"], vy[0]
                    )

                for k in six.moves.range(self.odim):
                    beam_hyp = {
                        "score": new_hyp["score"] + float(ytu[k]),
                        "yseq": new_hyp["yseq"][:],
                        "z_prev": new_hyp["z_prev"],
                        "c_prev": new_hyp["c_prev"],
                        "lm_state": new_hyp["lm_state"],
                    }

                    if k == self.blank:
                        kept_hyps.append(beam_hyp)
                    else:
                        beam_hyp["z_prev"] = z_list[:]
                        beam_hyp["c_prev"] = c_list[:]

                        beam_hyp["yseq"].append(int(k))

                        if rnnlm:
                            beam_hyp["lm_state"] = rnnlm_state
                            beam_hyp["score"] += (
                                recog_args.lm_weight * rnnlm_scores[0][k]
                            )

                        hyps.append(beam_hyp)

                hyps_max = float(max(hyps, key=lambda x: x["score"])["score"])
                kept_most_prob = len(
                    sorted(kept_hyps, key=lambda x: float(x["score"]) > hyps_max)
                )
                if kept_most_prob >= k_range:
                    break

        if normscore:
            nbest_hyps = sorted(
                kept_hyps, key=lambda x: x["score"] / len(x["yseq"]), reverse=True
            )[:nbest]
        else:
            nbest_hyps = sorted(kept_hyps, key=lambda x: x["score"], reverse=True)[
                :nbest
            ]

        return nbest_hyps

    def recognize_beam_nsc(self, h, recog_args, rnnlm=None):
        """N-step constrained beam search implementation.

        Based and modified from https://arxiv.org/pdf/2002.03577.pdf

        Args:
            h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
            recog_args (Namespace): argument Namespace containing options
            rnnlm (torch.nn.Module): language module

        Returns:
            nbest_hyps (list of dicts): n-best decoding results

        """
        beam = recog_args.beam_size
        w_range = min(beam, self.odim)

        nstep = recog_args.nstep
        prefix_alpha = recog_args.prefix_alpha

        nbest = recog_args.nbest

        w_zlist, w_clist = self.zero_state(torch.zeros((w_range, self.dunits)))

        w_tokens = [self.blank for _ in range(w_range)]
        w_tokens = torch.LongTensor(w_tokens).view(w_range)

        w_ey = self.embed(w_tokens)

        w_y, (w_zlist, w_clist) = self.rnn_forward(w_ey, (w_zlist, w_clist))

        zlist = [w_zlist[layer][0] for layer in range(self.dlayers)]
        clist = [w_clist[layer][0] for layer in range(self.dlayers)]

        if rnnlm:
            w_rnnlm_states, w_rnnlm_scores = rnnlm.buff_predict(None, w_tokens, beam)

            if hasattr(rnnlm.predictor, "wordlm"):
                lm_type = "wordlm"
                lm_layers = len(w_rnnlm_states[0])
            else:
                lm_type = "lm"
                lm_layers = len(w_rnnlm_states["c"])

            rnnlm_states = get_idx_lm_state(w_rnnlm_states, 0, lm_type, lm_layers)
            rnnlm_scores = w_rnnlm_scores[0]
        else:
            rnnlm_states = None
            rnnlm_scores = None

        kept_hyps = [
            {
                "yseq": [self.blank],
                "score": 0.0,
                "zlist": zlist,
                "clist": clist,
                "y": [w_y[0]],
                "lm_states": rnnlm_states,
                "lm_scores": rnnlm_scores,
            }
        ]

        for hi in h:
            hyps = sorted(kept_hyps, key=lambda x: len(x["yseq"]), reverse=True)
            kept_hyps = []

            for j in range(len(hyps) - 1):
                for i in range((j + 1), len(hyps)):
                    if (
                        is_prefix(hyps[j]["yseq"], hyps[i]["yseq"])
                        and (len(hyps[j]["yseq"]) - len(hyps[i]["yseq"]))
                        <= prefix_alpha
                    ):
                        next_id = len(hyps[i]["yseq"])

                        ytu = F.log_softmax(self.joint(hi, hyps[i]["y"][-1]), dim=0)

                        curr_score = float(hyps[i]["score"]) + float(
                            ytu[hyps[j]["yseq"][next_id]]
                        )

                        for k in range(next_id, (len(hyps[j]["yseq"]) - 1)):
                            ytu = F.log_softmax(self.joint(hi, hyps[j]["y"][k]), dim=0)

                            curr_score += float(ytu[hyps[j]["yseq"][k + 1]])

                        hyps[j]["score"] = np.logaddexp(
                            float(hyps[j]["score"]), curr_score
                        )

            S = []
            V = []
            for n in range(nstep):
                h_enc = hi.unsqueeze(0).expand(w_range, -1)

                w_y = torch.stack([hyp["y"][-1] for hyp in hyps])

                if len(hyps) == 1:
                    w_y = w_y.expand(w_range, -1)

                w_logprobs = F.log_softmax(self.joint(h_enc, w_y), dim=-1).view(-1)

                if rnnlm:
                    w_rnnlm_scores = torch.stack([hyp["lm_scores"] for hyp in hyps])

                    if len(hyps) == 1:
                        w_rnnlm_scores = w_rnnlm_scores.expand(w_range, -1)

                    w_rnnlm_scores = w_rnnlm_scores.contiguous().view(-1)

                for i, hyp in enumerate(hyps):
                    pos_k = i * self.odim
                    k_i = w_logprobs.narrow(0, pos_k, self.odim)

                    if rnnlm:
                        lm_k_i = w_rnnlm_scores.narrow(0, pos_k, self.odim)

                    for k in range(self.odim):
                        curr_score = float(k_i[k])

                        w_hyp = {
                            "yseq": hyp["yseq"][:],
                            "score": hyp["score"] + curr_score,
                            "zlist": hyp["zlist"],
                            "clist": hyp["clist"],
                            "y": hyp["y"][:],
                            "lm_states": hyp["lm_states"],
                            "lm_scores": hyp["lm_scores"],
                        }

                        if k == self.blank:
                            S.append(w_hyp)
                        else:
                            w_hyp["yseq"].append(int(k))

                            if rnnlm:
                                w_hyp["score"] += recog_args.lm_weight * lm_k_i[k]

                            V.append(w_hyp)

                V = sorted(V, key=lambda x: x["score"], reverse=True)
                V = substract(V, hyps)[:w_range]

                w_tokens = [v["yseq"][-1] for v in V]
                w_tokens = torch.LongTensor(w_tokens).view(w_range)

                for layer in range(self.dlayers):
                    w_zlist[layer] = torch.stack([v["zlist"][layer] for v in V])
                    w_clist[layer] = torch.stack([v["clist"][layer] for v in V])

                w_ey = self.embed(w_tokens)

                w_y, (w_zlist, w_clist) = self.rnn_forward(w_ey, (w_zlist, w_clist))

                if rnnlm:
                    w_rnnlm_states = get_beam_lm_states(
                        [v["lm_states"] for v in V], lm_type, lm_layers
                    )

                    w_rnnlm_states, w_rnnlm_scores = rnnlm.buff_predict(
                        w_rnnlm_states, w_tokens, w_range
                    )

                if n < (nstep - 1):
                    for i, v in enumerate(V):
                        v["zlist"] = [
                            w_zlist[layer][i] for layer in range(self.dlayers)
                        ]
                        v["clist"] = [
                            w_clist[layer][i] for layer in range(self.dlayers)
                        ]

                        v["y"].append(w_y[i])

                        if rnnlm:
                            v["lm_states"] = get_idx_lm_state(
                                w_rnnlm_states, i, lm_type, lm_layers
                            )
                            v["lm_scores"] = w_rnnlm_scores[i]

                    hyps = V[:]
                else:
                    w_logprobs = F.log_softmax(self.joint(h_enc, w_y), dim=-1).view(-1)
                    blank_score = w_logprobs[0 :: self.odim]

                    for i, v in enumerate(V):
                        if nstep != 1:
                            v["score"] += float(blank_score[i])

                        v["zlist"] = [
                            w_zlist[layer][i] for layer in range(self.dlayers)
                        ]
                        v["clist"] = [
                            w_clist[layer][i] for layer in range(self.dlayers)
                        ]

                        v["y"].append(w_y[i])

                        if rnnlm:
                            v["lm_states"] = get_idx_lm_state(
                                w_rnnlm_states, i, lm_type, lm_layers
                            )
                            v["lm_scores"] = w_rnnlm_scores[i]

            kept_hyps = sorted((S + V), key=lambda x: x["score"], reverse=True)[
                :w_range
            ]

        nbest_hyps = sorted(
            kept_hyps, key=lambda x: x["score"] / len(x["yseq"]), reverse=True
        )[:nbest]

        return nbest_hyps


class DecoderRNNTAtt(torch.nn.Module):
    """RNNT-Att Decoder module.

    Args:
        eprojs (int): # encoder projection units
        odim (int): dimension of outputs
        dtype (str): gru or lstm
        dlayers (int): # decoder layers
        dunits (int): # decoder units
        blank (int): blank symbol id
        att (torch.nn.Module): attention module
        embed_dim (int): dimension of embeddings
        joint_dim (int): dimension of joint space
        dropout (float): dropout rate
        dropout_embed (float): embedding dropout rate

    """

    def __init__(
        self,
        eprojs,
        odim,
        dtype,
        dlayers,
        dunits,
        blank,
        att,
        embed_dim,
        joint_dim,
        dropout=0.0,
        dropout_embed=0.0,
    ):
        """Transducer with attention initializer."""
        super(DecoderRNNTAtt, self).__init__()

        self.embed = torch.nn.Embedding(odim, embed_dim, padding_idx=blank)
        self.dropout_emb = torch.nn.Dropout(p=dropout_embed)

        if dtype == "lstm":
            dec_net = torch.nn.LSTMCell
        else:
            dec_net = torch.nn.GRUCell

        self.decoder = torch.nn.ModuleList([dec_net((embed_dim + eprojs), dunits)])
        self.dropout_dec = torch.nn.ModuleList([torch.nn.Dropout(p=dropout)])

        for _ in six.moves.range(1, dlayers):
            self.decoder += [dec_net(dunits, dunits)]
            self.dropout_dec += [torch.nn.Dropout(p=dropout)]

        self.lin_enc = torch.nn.Linear(eprojs, joint_dim)
        self.lin_dec = torch.nn.Linear(dunits, joint_dim, bias=False)
        self.lin_out = torch.nn.Linear(joint_dim, odim)

        self.att = att

        self.dtype = dtype
        self.dlayers = dlayers
        self.dunits = dunits
        self.embed_dim = embed_dim
        self.joint_dim = joint_dim
        self.odim = odim

        self.ignore_id = -1
        self.blank = blank

    def zero_state(self, ey):
        """Initialize decoder states.

        Args:
            ey (torch.Tensor): batch of input features (B, (Emb_dim + Eprojs))

        Return:
            z_list : list of L zero-init hidden state (B, Hdec)
            c_list : list of L zero-init cell state (B, Hdec)

        """
        z_list = [ey.new_zeros(ey.size(0), self.dunits)]
        c_list = [ey.new_zeros(ey.size(0), self.dunits)]

        for _ in six.moves.range(1, self.dlayers):
            z_list.append(ey.new_zeros(ey.size(0), self.dunits))
            c_list.append(ey.new_zeros(ey.size(0), self.dunits))

        return z_list, c_list

    def rnn_forward(self, ey, dstate):
        """RNN forward.

        Args:
            ey (torch.Tensor): batch of input features (B, (Emb_dim + Eprojs))
            dstate (list): list of L input hidden and cell state (B, Hdec)
        Returns:
            y (torch.Tensor): decoder output for one step (B, Hdec)
            (list): list of L output hidden and cell state (B, Hdec)

        """
        if dstate is None:
            z_prev, c_prev = self.zero_state(ey)
        else:
            z_prev, c_prev = dstate

        z_list, c_list = self.zero_state(ey)

        if self.dtype == "lstm":
            z_list[0], c_list[0] = self.decoder[0](ey, (z_prev[0], c_prev[0]))

            for i in six.moves.range(1, self.dlayers):
                z_list[i], c_list[i] = self.decoder[i](
                    self.dropout_dec[i - 1](z_list[i - 1]), (z_prev[i], c_prev[i])
                )
        else:
            z_list[0] = self.decoder[0](ey, z_prev[0])

            for i in six.moves.range(1, self.dlayers):
                z_list[i] = self.decoder[i](
                    self.dropout_dec[i - 1](z_list[i - 1]), z_prev[i]
                )
        y = self.dropout_dec[-1](z_list[-1])

        return y, (z_list, c_list)

    def joint(self, h_enc, h_dec):
        """Joint computation of z.

        Args:
            h_enc (torch.Tensor): batch of expanded hidden state (B, T, 1, Henc)
            h_dec (torch.Tensor): batch of expanded hidden state (B, 1, U, Hdec)

        Returns:
            z (torch.Tensor): output (B, T, U, odim)

        """
        z = torch.tanh(self.lin_enc(h_enc) + self.lin_dec(h_dec))
        z = self.lin_out(z)

        return z

    def forward(self, hs_pad, ys_in_pad, hlens=None):
        """Forward function for transducer with attention.

        Args:
            hs_pad (torch.Tensor): batch of padded hidden state sequences (B, Tmax, D)
            ys_in_pad (torch.Tensor):
                batch of padded character id sequence tensor (B, Lmax+1)

        Returns:
            z (torch.Tensor): output (B, T, U, odim)

        """
        olength = ys_in_pad.size(1)

        hlens = list(map(int, hlens))

        att_w = None
        self.att[0].reset()

        z_list, c_list = self.zero_state(hs_pad)
        eys = self.dropout_emb(self.embed(ys_in_pad))

        z_all = []
        for i in six.moves.range(olength):
            att_c, att_w = self.att[0](
                hs_pad, hlens, self.dropout_dec[0](z_list[0]), att_w
            )

            ey = torch.cat((eys[:, i, :], att_c), dim=1)

            y, (z_list, c_list) = self.rnn_forward(ey, (z_list, c_list))
            z_all.append(y)

        h_dec = torch.stack(z_all, dim=1)

        h_enc = hs_pad.unsqueeze(2)
        h_dec = h_dec.unsqueeze(1)

        z = self.joint(h_enc, h_dec)

        return z

    def recognize(self, h, recog_args):
        """Greedy search implementation.

        Args:
            h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
            recog_args (Namespace): argument Namespace containing options

        Returns:
            hyp (list of dicts): 1-best decoding results

        """
        self.att[0].reset()

        z_list, c_list = self.zero_state(h.unsqueeze(0))
        eys = torch.zeros((1, self.embed_dim))

        att_c, att_w = self.att[0](
            h.unsqueeze(0), [h.size(0)], self.dropout_dec[0](z_list[0]), None
        )

        ey = torch.cat((eys, att_c), dim=1)

        hyp = {"score": 0.0, "yseq": [self.blank]}

        y, (z_list, c_list) = self.rnn_forward(ey, (z_list, c_list))

        for hi in h:
            ytu = F.log_softmax(self.joint(hi, y[0]), dim=0)
            logp, pred = torch.max(ytu, dim=0)

            if pred != self.blank:
                hyp["yseq"].append(int(pred))
                hyp["score"] += float(logp)

                eys = torch.full((1, 1), hyp["yseq"][-1], dtype=torch.long)
                ey = self.embed(eys)

                att_c, att_w = self.att[0](
                    h.unsqueeze(0), [h.size(0)], self.dropout_dec[0](z_list[0]), att_w
                )

                ey = torch.cat((ey[0], att_c), dim=1)

                y, (z_list, c_list) = self.rnn_forward(ey, (z_list, c_list))

        return [hyp]

    def recognize_beam_default(self, h, recog_args, rnnlm=None):
        """Beam search recognition.

        Args:
            h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
            recog_args (Namespace): argument Namespace containing options
            rnnlm (torch.nn.Module): language module

        Results:
            nbest_hyps (list of dicts): n-best decoding results

        """
        beam = recog_args.beam_size
        k_range = min(beam, self.odim)

        nbest = recog_args.nbest
        normscore = recog_args.score_norm_transducer

        self.att[0].reset()

        z_list, c_list = self.zero_state(h.unsqueeze(0))

        kept_hyps = [
            {
                "score": 0.0,
                "yseq": [self.blank],
                "z_prev": z_list,
                "c_prev": c_list,
                "a_prev": None,
                "lm_state": None,
            }
        ]

        for hi in h:
            hyps = kept_hyps
            kept_hyps = []

            while True:
                new_hyp = max(hyps, key=lambda x: x["score"])
                hyps.remove(new_hyp)

                vy = to_device(
                    self, torch.full((1, 1), new_hyp["yseq"][-1], dtype=torch.long)
                )
                ey = self.embed(vy)

                att_c, att_w = self.att[0](
                    h.unsqueeze(0),
                    [h.size(0)],
                    self.dropout_dec[0](new_hyp["z_prev"][0]),
                    new_hyp["a_prev"],
                )

                ey = torch.cat((ey[0], att_c), dim=1)

                y, (z_list, c_list) = self.rnn_forward(
                    ey, (new_hyp["z_prev"], new_hyp["c_prev"])
                )
                ytu = F.log_softmax(self.joint(hi, y[0]), dim=0)

                if rnnlm:
                    rnnlm_state, rnnlm_scores = rnnlm.predict(
                        new_hyp["lm_state"], vy[0]
                    )

                for k in six.moves.range(self.odim):
                    beam_hyp = {
                        "score": new_hyp["score"] + float(ytu[k]),
                        "yseq": new_hyp["yseq"][:],
                        "z_prev": new_hyp["z_prev"],
                        "c_prev": new_hyp["c_prev"],
                        "a_prev": new_hyp["a_prev"],
                        "lm_state": new_hyp["lm_state"],
                    }

                    if k == self.blank:
                        kept_hyps.append(beam_hyp)
                    else:
                        beam_hyp["z_prev"] = z_list[:]
                        beam_hyp["c_prev"] = c_list[:]
                        beam_hyp["a_prev"] = att_w[:]

                        beam_hyp["yseq"].append(int(k))

                        if rnnlm:
                            beam_hyp["lm_state"] = rnnlm_state
                            beam_hyp["score"] += (
                                recog_args.lm_weight * rnnlm_scores[0][k]
                            )

                        hyps.append(beam_hyp)

                hyps_max = float(max(hyps, key=lambda x: x["score"])["score"])
                kept_most_prob = len(
                    sorted(kept_hyps, key=lambda x: float(x["score"]) > hyps_max)
                )
                if kept_most_prob >= k_range:
                    break

        if normscore:
            nbest_hyps = sorted(
                kept_hyps, key=lambda x: x["score"] / len(x["yseq"]), reverse=True
            )[:nbest]
        else:
            nbest_hyps = sorted(kept_hyps, key=lambda x: x["score"], reverse=True)[
                :nbest
            ]

        return nbest_hyps

    def recognize_beam_nsc(self, h, recog_args, rnnlm=None):
        """N-step constrained beam search implementation.

        Based and modified from https://arxiv.org/pdf/2002.03577.pdf

        Args:
            h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
            recog_args (Namespace): argument Namespace containing options
            rnnlm (torch.nn.Module): language module

        Returns:
            nbest_hyps (list of dicts): n-best decoding results

        """
        beam = recog_args.beam_size
        w_range = min(beam, self.odim)

        nstep = recog_args.nstep
        prefix_alpha = recog_args.prefix_alpha

        nbest = recog_args.nbest

        self.att[0].reset()

        w_zlist, w_clist = self.zero_state(torch.zeros((w_range, self.dunits)))

        w_tokens = [self.blank for _ in range(w_range)]
        w_tokens = torch.LongTensor(w_tokens).view(w_range)

        w_ey = self.embed(w_tokens)

        hlens = [h.size(0)] * w_range
        w_h = h.unsqueeze(0).expand(w_range, -1, -1)

        w_att_c, w_att_w = self.att[0](w_h, hlens, None, None)

        w_ey = torch.cat((w_ey, w_att_c), dim=1)

        w_y, (w_zlist, w_clist) = self.rnn_forward(w_ey, (w_zlist, w_clist))

        zlist = [w_zlist[layer][0] for layer in range(self.dlayers)]
        clist = [w_zlist[layer][0] for layer in range(self.dlayers)]

        if rnnlm:
            w_rnnlm_states, w_rnnlm_scores = rnnlm.buff_predict(None, w_tokens, w_range)

            if hasattr(rnnlm.predictor, "wordlm"):
                lm_type = "wordlm"
                lm_layers = len(w_rnnlm_states[0])
            else:
                lm_type = "lm"
                lm_layers = len(w_rnnlm_states["c"])

            rnnlm_states = get_idx_lm_state(w_rnnlm_states, 0, lm_type, lm_layers)
            rnnlm_scores = w_rnnlm_scores[0]
        else:
            rnnlm_states = None
            rnnlm_scores = None

        kept_hyps = [
            {
                "yseq": [self.blank],
                "score": 0.0,
                "zlist": zlist,
                "clist": clist,
                "alist": w_att_w[0],
                "y": [w_y[0]],
                "lm_states": rnnlm_states,
                "lm_scores": rnnlm_scores,
            }
        ]

        for hi in h:
            hyps = sorted(kept_hyps, key=lambda x: len(x["yseq"]), reverse=True)
            kept_hyps = []

            for j in range(len(hyps) - 1):
                for i in range((j + 1), len(hyps)):
                    if (
                        is_prefix(hyps[j]["yseq"], hyps[i]["yseq"])
                        and (len(hyps[j]["yseq"]) - len(hyps[i]["yseq"]))
                        <= prefix_alpha
                    ):
                        next_id = len(hyps[i]["yseq"])

                        ytu = F.log_softmax(self.joint(hi, hyps[i]["y"][-1]), dim=0)

                        curr_score = float(hyps[i]["score"]) + float(
                            ytu[hyps[j]["yseq"][next_id]]
                        )

                        for k in range(next_id, (len(hyps[j]["yseq"]) - 1)):
                            ytu = F.log_softmax(self.joint(hi, hyps[j]["y"][k]), dim=0)

                            curr_score += float(ytu[hyps[j]["yseq"][k + 1]])

                        hyps[j]["score"] = np.logaddexp(
                            float(hyps[j]["score"]), curr_score
                        )

            S = []
            V = []
            for n in range(nstep):
                h_enc = hi.unsqueeze(0).expand(w_range, -1)

                w_y = torch.stack([hyp["y"][-1] for hyp in hyps])

                if len(hyps) == 1:
                    w_y = w_y.expand(w_range, -1)

                w_logprobs = F.log_softmax(self.joint(h_enc, w_y), dim=-1).view(-1)

                if rnnlm:
                    w_rnnlm_scores = torch.stack([hyp["lm_scores"] for hyp in hyps])

                    if len(hyps) == 1:
                        w_rnnlm_scores = w_rnnlm_scores.expand(w_range, -1)

                    w_rnnlm_scores = w_rnnlm_scores.contiguous().view(-1)

                for i, hyp in enumerate(hyps):
                    pos_k = i * self.odim
                    k_i = w_logprobs.narrow(0, pos_k, self.odim)

                    if rnnlm:
                        lm_k_i = w_rnnlm_scores.narrow(0, pos_k, self.odim)

                    for k in range(self.odim):
                        curr_score = float(k_i[k])

                        w_hyp = {
                            "yseq": hyp["yseq"][:],
                            "score": hyp["score"] + curr_score,
                            "zlist": hyp["zlist"],
                            "clist": hyp["clist"],
                            "alist": hyp["alist"],
                            "y": hyp["y"][:],
                            "lm_states": hyp["lm_states"],
                            "lm_scores": hyp["lm_scores"],
                        }

                        if k == self.blank:
                            S.append(w_hyp)
                        else:
                            w_hyp["yseq"].append(int(k))

                            if rnnlm:
                                w_hyp["score"] += recog_args.lm_weight * lm_k_i[k]

                            V.append(w_hyp)

                V = sorted(V, key=lambda x: x["score"], reverse=True)
                V = substract(V, hyps)[:w_range]

                w_tokens = [v["yseq"][-1] for v in V]
                w_tokens = torch.LongTensor(w_tokens).view(w_range)

                for layer in range(self.dlayers):
                    w_zlist[layer] = torch.stack([v["zlist"][layer] for v in V])
                    w_clist[layer] = torch.stack([v["clist"][layer] for v in V])

                w_att_w = torch.stack([v["alist"] for v in V])

                w_ey = self.embed(w_tokens)

                w_att_c, w_att_w = self.att[0](
                    w_h, hlens, self.dropout_dec[0](w_zlist[0]), w_att_w
                )

                w_ey = torch.cat((w_ey, w_att_c), dim=1)

                w_y, (w_zlist, w_clist) = self.rnn_forward(w_ey, (w_zlist, w_clist))

                if rnnlm:
                    w_rnnlm_states = get_beam_lm_states(
                        [v["lm_states"] for v in V], lm_type, lm_layers
                    )

                    w_rnnlm_states, w_rnnlm_scores = rnnlm.buff_predict(
                        w_rnnlm_states, w_tokens, w_range
                    )

                if n < (nstep - 1):
                    for i, v in enumerate(V):
                        v["zlist"] = [
                            w_zlist[layer][i] for layer in range(self.dlayers)
                        ]
                        v["clist"] = [
                            w_clist[layer][i] for layer in range(self.dlayers)
                        ]

                        v["alist"] = w_att_w[i]

                        v["y"].append(w_y[i])

                        if rnnlm:
                            v["lm_states"] = get_idx_lm_state(
                                w_rnnlm_states, i, lm_type, lm_layers
                            )
                            v["lm_scores"] = w_rnnlm_scores[i]

                    hyps = V[:]
                else:
                    w_logprobs = F.log_softmax(self.joint(h_enc, w_y), dim=-1).view(-1)
                    blank_score = w_logprobs[0 :: self.odim]

                    for i, v in enumerate(V):
                        if nstep != 1:
                            v["score"] += float(blank_score[i])

                        v["zlist"] = [
                            w_zlist[layer][i] for layer in range(self.dlayers)
                        ]
                        v["clist"] = [
                            w_clist[layer][i] for layer in range(self.dlayers)
                        ]

                        v["alist"] = w_att_w[i]

                        v["y"].append(w_y[i])

                        if rnnlm:
                            v["lm_states"] = get_idx_lm_state(
                                w_rnnlm_states, i, lm_type, lm_layers
                            )
                            v["lm_scores"] = w_rnnlm_scores[i]

            kept_hyps = sorted((S + V), key=lambda x: x["score"], reverse=True)[
                :w_range
            ]

        nbest_hyps = sorted(
            kept_hyps, key=lambda x: x["score"] / len(x["yseq"]), reverse=True
        )[:nbest]

        return nbest_hyps

    def calculate_all_attentions(self, hs_pad, hlens, ys_pad):
        """Calculate all of attentions.

        Args:
            hs_pad (torch.Tensor): batch of padded hidden state sequences (B, Tmax, D)
            hlens (torch.Tensor): batch of lengths of hidden state sequences (B)
            ys_pad (torch.Tensor):
                batch of padded character id sequence tensor (B, Lmax)

        Returns:
            att_ws (ndarray): attention weights with the following shape,
                1) multi-head case => attention weights (B, H, Lmax, Tmax),
                2) other case => attention weights (B, Lmax, Tmax).

        """
        ys = [y[y != self.ignore_id] for y in ys_pad]

        hlens = list(map(int, hlens))

        blank = ys[0].new([self.blank])

        ys_in = [torch.cat([blank, y], dim=0) for y in ys]
        ys_in_pad = pad_list(ys_in, self.blank)

        olength = ys_in_pad.size(1)

        att_w = None
        att_ws = []
        self.att[0].reset()

        eys = self.embed(ys_in_pad)
        z_list, c_list = self.zero_state(eys)

        for i in six.moves.range(olength):
            att_c, att_w = self.att[0](
                hs_pad, hlens, self.dropout_dec[0](z_list[0]), att_w
            )
            ey = torch.cat((eys[:, i, :], att_c), dim=1)
            _, (z_list, c_list) = self.rnn_forward(ey, (z_list, c_list))

            att_ws.append(att_w)

        att_ws = att_to_numpy(att_ws, self.att[0])

        return att_ws


def decoder_for(args, odim, att=None, blank=0):
    """Transducer mode selector."""
    if args.rnnt_mode == "rnnt":
        return DecoderRNNT(
            args.eprojs,
            odim,
            args.dtype,
            args.dlayers,
            args.dunits,
            blank,
            args.dec_embed_dim,
            args.joint_dim,
            args.dropout_rate_decoder,
            args.dropout_rate_embed_decoder,
        )
    elif args.rnnt_mode == "rnnt-att":
        return DecoderRNNTAtt(
            args.eprojs,
            odim,
            args.dtype,
            args.dlayers,
            args.dunits,
            blank,
            att,
            args.dec_embed_dim,
            args.joint_dim,
            args.dropout_rate_decoder,
            args.dropout_rate_embed_decoder,
        )
