"""Transducer and transducer with attention implementation for training and decoding."""

import copy
import six

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.rnn.attentions import att_to_numpy

from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device


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

            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    self.dropout_dec[l - 1](z_list[l - 1]), (z_prev[l], c_prev[l])
                )
        else:
            z_list[0] = self.decoder[0](ey, z_prev[0])

            for l in six.moves.range(1, self.dlayers):
                z_list[l] = self.decoder[l](
                    self.dropout_dec[l - 1](z_list[l - 1]), z_prev[l]
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
                ey = self.dropout_embed(self.embed(eys))

                y, (z_list, c_list) = self.rnn_forward(ey[0], (z_list, c_list))

        return [hyp]

    def recognize_beam(self, h, recog_args, rnnlm=None):
        """Default beam search implementation.

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
        eys = to_device(self, torch.zeros((1, self.embed_dim)))

        _, (z_list, c_list) = self.rnn_forward(eys, None)

        kept_hyps = [
            {
                "score": 0.0,
                "yseq": [self.blank],
                "z_prev": z_list,
                "c_prev": c_list,
                "lm_state": None,
            }
        ]

        for i, hi in enumerate(h):
            hyps = kept_hyps
            kept_hyps = []

            while True:
                new_hyp = max(hyps, key=lambda x: x["score"])
                hyps.remove(new_hyp)

                vy = to_device(
                    self, torch.full((1, 1), new_hyp["yseq"][-1], dtype=torch.long)
                )
                ey = self.dropout_embed(self.embed(vy))

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
                    }

                    if rnnlm:
                        beam_hyp["lm_state"] = new_hyp["lm_state"]

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

                if len(kept_hyps) >= k_range:
                    break

        if normscore:
            nbest_hyps = sorted(
                kept_hyps, key=lambda x: x["score"] / len(x["yseq"]), reverse=True
            )[:nbest]
        else:
            nbest_hyps = sorted(kept_hyps, key=lambda x: x["score"], reverse=True
            )[:nbest]

        return nbest_hyps

    def recognize_beam_osc(self, h, recog_args, rnnlm=None):
        """One-step constrained beam search implementation.

        Based on https://arxiv.org/pdf/2002.03577.pdf
        Args:
            h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
            recog_args (Namespace): argument Namespace containing options
            rnnlm (torch.nn.Module): language module

        Returns:
            nbest_hyps (list of dicts): n-best decoding results

        """
        def substract(x, subset):
            final = []

            for x_ in x:
                if any(x_['yseq'] == sub['yseq'] \
                       for sub in subset):
                    continue
                final.append(x_)

            return final

        beam = recog_args.beam_size
        w_range = min(beam, self.odim)
        k_dim = self.odim - 1

        nbest = recog_args.nbest
        normscore = recog_args.score_norm_transducer

        zlist, clist = self.zero_state(h.unsqueeze(0))
        w_zlist, w_clist = self.zero_state(torch.zeros((w_range, self.dunits)))

        w_tokens = [self.blank for _ in range(w_range)]
        w_tokens = torch.LongTensor(w_tokens).view(w_range)

        w_ey = self.dropout_embed(self.embed(w_tokens))

        w_y, (w_zlist, w_clist) = self.rnn_forward(w_ey, (w_zlist, w_clist))

        kept_hyps = [
            {
                "yseq": [self.blank],
                "score": 0.0,
                "zlist": zlist[:],
                "clist": clist[:],
                "y": w_y[0],
                "lm_state": None,
            } for _ in range(w_range)
        ]

        for w in six.moves.range(w_range):
            for l in six.moves.range(self.dlayers):
                kept_hyps[w]["zlist"][l] = w_zlist[l][w]
                kept_hyps[w]["clist"][l] = w_clist[l][w]
                
        for hi in h:
            hyps = kept_hyps
            kept_hyps = []

            S = []
            V = []

            h_enc = hi.unsqueeze(0).expand(w_range, -1)

            for i, hyp in enumerate(hyps):
                w_y[i] = hyp["y"]

            w_logprobs = F.log_softmax(self.joint(h_enc, w_y), dim=-1)
            w_logprobs = torch.flatten(w_logprobs)

            for i, hyp in enumerate(hyps):
                pos_k = (i * self.odim)
                k_i = w_logprobs.narrow(0, pos_k, self.odim)

                for k in range(self.odim):
                    curr_score = float(k_i[k])

                    w_hyp = {
                        "yseq": hyp["yseq"][:],
                        "score": hyp["score"] + curr_score,
                        "zlist": hyp["zlist"],
                        "clist": hyp["clist"],
                        "y": hyp["y"]
                    }

                    if k == self.blank:
                        S.append(w_hyp)
                    else:
                        w_hyp["yseq"].append(int(k))

                        V.append(w_hyp)

            V = sorted(V, key=lambda x: x["score"], reverse=True)[:w_range]

            V_ = substract(V, hyps)

            w_tokens = [v_["yseq"][-1] for v_ in V_]
            w_tokens = torch.LongTensor(w_tokens).view(w_range)

            for w in six.moves.range(w_range):
                for l in six.moves.range(self.dlayers):
                    w_zlist[l][w] = V_[w]["zlist"][l]
                    w_clist[l][w] = V_[w]["clist"][l]

            w_ey = self.dropout_embed(self.embed(w_tokens))

            w_y, (w_zlist, w_clist) = self.rnn_forward(w_ey, (w_zlist, w_clist))

            w_logprobs = F.log_softmax(self.joint(h_enc, w_y), dim=-1)
            w_logprobs = torch.flatten(w_logprobs)

            blank_score = w_logprobs[0::self.odim]

            for i, v_ in enumerate(V_):
                for l in six.moves.range(self.dlayers):
                    v_["zlist"][l] = w_zlist[l][i]
                    v_["clist"][l] = w_clist[l][i]

                v_["y"] = w_y[i]
                v_["score"] += float(blank_score[i])

            kept_hyps = sorted(
                (S + V_), key=lambda x: x["score"], reverse=True
            )[:w_range]

        if normscore:
            nbest_hyps = sorted(
                kept_hyps, key=lambda x: x["score"] / len(x["yseq"]), reverse=True
            )[:nbest]
        else:
            nbest_hyps = sorted(kept_hyps, key=lambda x: x["score"], reverse=True
            )[:nbest]

        return nbest_hyps

    def recognize_beam_nsc(self, h, recog_args, rnnlm=None):
        """N-step constrained beam search implementation.

        Args:
            h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
            recog_args (Namespace): argument Namespace containing options
            rnnlm (torch.nn.Module): language module

        Returns:
            nbest_hyps (list of dicts): n-best decoding results

        """
        def substract(x, subset):
            final = []

            for x_ in x:
                if any(x_['yseq'] == sub['yseq'] \
                       for sub in subset):
                    continue
                final.append(x_)

            return final

        beam = recog_args.beam_size
        w_range = min(beam, self.odim)

        extra_step = 1

        nbest = recog_args.nbest
        normscore = recog_args.score_norm_transducer

        zlist, clist = self.zero_state(h.unsqueeze(0))
        w_zlist, w_clist = self.zero_state(torch.zeros((w_range, self.dunits)))

        w_tokens = [self.blank for _ in range(w_range)]
        w_tokens = torch.LongTensor(w_tokens).view(w_range)

        w_ey = self.dropout_embed(self.embed(w_tokens))

        w_y, (w_zlist, w_clist) = self.rnn_forward(w_ey, (w_zlist, w_clist))

        kept_hyps = [
            {
                "yseq": [self.blank],
                "score": 0.0,
                "zlist": zlist,
                "clist": clist,
                "y": w_y[0],
                "lm_state": None
            } for _ in range(w_range)
        ]

        for w in six.moves.range(w_range):
            for l in six.moves.range(self.dlayers):
                kept_hyps[w]["zlist"][l] = w_zlist[l][w]
                kept_hyps[w]["clist"][l] = w_clist[l][w]
                
        for hi in h:
            hyps = kept_hyps
            kept_hyps = []

            S = []
            V = []
            for n in range(extra_step):
                h_enc = hi.unsqueeze(0).expand(w_range, -1)

                for i, hyp in enumerate(hyps):
                    w_y[i] = hyp["y"]

                w_logprobs = F.log_softmax(self.joint(h_enc, w_y), dim=-1)
                w_logprobs = torch.flatten(w_logprobs)

                for i, hyp in enumerate(hyps):
                    pos_k = (i * self.odim)
                    k_i = w_logprobs.narrow(0, pos_k, self.odim)

                    for k in range(self.odim):
                        curr_score = float(k_i[k])

                        w_hyp = {
                            "yseq": hyp["yseq"],
                            "score": hyp["score"] + curr_score,
                            "zlist": hyp["zlist"],
                            "clist": hyp["clist"],
                            "y": hyp["y"]
                        }

                        if k == self.blank:
                            S.append(w_hyp)
                        else:
                            w_hyp["yseq"].append(int(k))

                            V.append(w_hyp)

                V = sorted(V, key=lambda x: x["score"], reverse=True)[:w_range]
                V_ = substract(V, hyps)
                
                if n < extra_step:
                    w_tokens = [v["yseq"][-1] for v in V_]
                    w_tokens = torch.LongTensor(w_tokens).view(w_range)

                    for w in six.moves.range(w_range):
                        for l in six.moves.range(self.dlayers):
                            w_zlist[l][w] = V_[w]["zlist"][l]
                            w_clist[l][w] = V_[w]["clist"][l]

                    w_ey = self.dropout_embed(self.embed(w_tokens))

                    w_y, (w_zlist, w_clist) = self.rnn_forward(w_ey, (w_zlist, w_clist))

                    for i, v in enumerate(V_):
                        for l in six.moves.range(self.dlayers):
                            v["zlist"][l] = w_zlist[l][i]
                            v["clist"][l] = w_clist[l][i]

                        v["y"] = w_y[i]

                    hyps = V_
                    
            w_tokens = [v_["yseq"][-1] for v_ in V_]
            w_tokens = torch.LongTensor(w_tokens).view(w_range)

            for w in six.moves.range(w_range):
                for l in six.moves.range(self.dlayers):
                    w_zlist[l][w] = V_[w]["zlist"][l]
                    w_clist[l][w] = V_[w]["clist"][l]

            w_ey = self.dropout_embed(self.embed(w_tokens))

            w_y, (w_zlist, w_clist) = self.rnn_forward(w_ey, (w_zlist, w_clist))

            w_logprobs = F.log_softmax(self.joint(h_enc, w_y), dim=-1)
            w_logprobs = torch.flatten(w_logprobs)

            blank_score = w_logprobs[0::self.odim]

            for i, v_ in enumerate(V_):
                for l in six.moves.range(self.dlayers):
                    v_["zlist"][l] = w_zlist[l][i]
                    v_["clist"][l] = w_clist[l][i]

                    v_["y"] = w_y[i]
                    v_["score"] += float(blank_score[i])

            kept_hyps = sorted(
                (S + V_), key=lambda x: x["score"], reverse=True
            )[:w_range]

        if normscore:
            nbest_hyps = sorted(
                kept_hyps, key=lambda x: x["score"] / len(x["yseq"]), reverse=True
            )[:nbest]
        else:
            nbest_hyps = sorted(kept_hyps, key=lambda x: x["score"], reverse=True
            )[:nbest]

        return nbest_hyps

    def recognize_beam_breadth_first(self, h, recog_args, rnnlm=None):
        """Breadth-first beam search implementation.

        Based on https://ieeexplore.ieee.org/document/9003822

        Args:
            h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
            recog_args (Namespace): argument Namespace containing options
            rnnlm (torch.nn.Module): language module

        Returns:
            nbest_hyps (list of dicts): n-best decoding results

        """
        beam = recog_args.beam_size
        w_range = min(beam, self.odim)
        max_exp = 2

        nbest = recog_args.nbest
        normscore = recog_args.score_norm_transducer

        zlist, clist = self.zero_state(h.unsqueeze(0))
        w_zlist, w_clist = self.zero_state(torch.zeros((w_range, self.dunits)))
        
        kept_hyps = [
            {
                "score": 0.0,
                "yseq": [self.blank],
                "zlist": zlist[:],
                "clist": clist[:],
                "lm_state": None
            } for _ in range(w_range)
        ]

        for hi in h:
            hyps = kept_hyps
            expansions = 0
            kept_hyps = []

            while expansions < max_exp and hyps:
                w_tokens = [hyp["yseq"][-1] for hyp in hyps]
                w_tokens = torch.LongTensor(w_tokens).view(w_range)

                for w in six.moves.range(w_range):
                    for l in six.moves.range(self.dlayers):
                        w_zlist[l][w] = hyps[w]["zlist"][l]
                        w_clist[l][w] = hyps[w]["clist"][l]

                w_ey = self.dropout_embed(self.embed(w_tokens))

                w_y, (w_zlist, w_clist) = self.rnn_forward(w_ey, (w_zlist, w_clist))

                w_logprobs = F.log_softmax(self.joint(hi, w_y), dim=0)
                w_logprobs = torch.flatten(w_logprobs)
                
                expansions += len(kept_hyps)
                hyps_new = hyps
                hyps = []

                for i, hyp in enumerate(hyps_new):
                    pos_k = (i * self.odim)
                    k_i = w_logprobs.narrow(0, pos_k, self.odim)

                    for k in six.moves.range(self.odim):
                        curr_score = float(k_i[i])

                        beam_hyp = {
                            "score": hyp["score"] + curr_score,
                            "yseq": hyp["yseq"][:],
                            "zlist": hyp["zlist"],
                            "clist": hyp["clist"],
                        }

                        if k == self.blank:
                            kept_hyps.append(beam_hyp)
                        else:
                            beam_hyp["yseq"].append(int(k))

                            for l in six.moves.range(self.dlayers):
                                beam_hyp["zlist"][l] = w_zlist[l][i]
                                beam_hyp["clist"][l] = w_clist[l][i]

                            hyps.append(beam_hyp)

                hyps = sorted(
                    hyps, key=lambda x: x["score"], reverse=True
                )[:w_range]

            kept_hyps = sorted(
                kept_hyps, key=lambda x: x["score"], reverse=True
            )[:w_range]

        nbest_hyps = sorted(
            kept_hyps, key=lambda x: x["score"], reverse=True
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

            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    self.dropout_dec[l - 1](z_list[l - 1]), (z_prev[l], c_prev[l])
                )
        else:
            z_list[0] = self.decoder[0](ey, z_prev[0])

            for l in six.moves.range(1, self.dlayers):
                z_list[l] = self.decoder[l](
                    self.dropout_dec[l - 1](z_list[l - 1]), z_prev[l]
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
                ey = self.dropout_emb(self.embed(eys))
                att_c, att_w = self.att[0](
                    h.unsqueeze(0), [h.size(0)], self.dropout_dec[0](z_list[0]), att_w
                )
                ey = torch.cat((ey[0], att_c), dim=1)

                y, (z_list, c_list) = self.rnn_forward(ey, (z_list, c_list))

        return [hyp]

    def recognize_beam(self, h, recog_args, rnnlm=None):
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
        eys = torch.zeros((1, self.embed_dim))

        att_c, att_w = self.att[0](
            h.unsqueeze(0), [h.size(0)], self.dropout_dec[0](z_list[0]), None
        )

        ey = torch.cat((eys, att_c), dim=1)
        _, (z_list, c_list) = self.rnn_forward(ey, None)

        if rnnlm:
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
        else:
            kept_hyps = [
                {
                    "score": 0.0,
                    "yseq": [self.blank],
                    "z_prev": z_list,
                    "c_prev": c_list,
                    "a_prev": None,
                }
            ]

        for i, hi in enumerate(h):
            hyps = kept_hyps
            kept_hyps = []

            while True:
                new_hyp = max(hyps, key=lambda x: x["score"])
                hyps.remove(new_hyp)

                vy = to_device(
                    self, torch.full((1, 1), new_hyp["yseq"][-1], dtype=torch.long)
                )
                ey = self.dropout_emb(self.embed(vy))

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
                    }
                    if rnnlm:
                        beam_hyp["lm_state"] = new_hyp["lm_state"]

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

                if len(kept_hyps) >= k_range:
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

        eys = self.dropout_emb(self.embed(ys_in_pad))
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
