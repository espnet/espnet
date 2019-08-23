"""Transducer and transducer with attention implementation for training and decoding."""

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
        rnnt_type (str): type of rnn-t implementation

    """

    def __init__(self, eprojs, odim, dtype, dlayers, dunits, blank,
                 embed_dim, joint_dim, dropout=0.0, dropout_embed=0.0,
                 rnnt_type='warp-transducer'):
        """Transducer initializer."""
        super(DecoderRNNT, self).__init__()

        self.embed = torch.nn.Embedding(odim, embed_dim, padding_idx=blank)
        self.dropout_embed = torch.nn.Dropout(p=dropout_embed)

        if dtype == "lstm":
            dec_net = torch.nn.LSTM
        else:
            dec_net = torch.nn.GRU

        self.decoder = torch.nn.ModuleList([dec_net(embed_dim, dunits, 1,
                                                    bias=True, batch_first=True,
                                                    bidirectional=False)])
        self.dropout_dec = torch.nn.ModuleList([torch.nn.Dropout(p=dropout)])

        for _ in six.moves.range(1, dlayers):
            self.decoder += [dec_net(dunits, dunits, 1,
                                     bias=True, batch_first=True,
                                     bidirectional=False)]
            self.dropout_dec += [torch.nn.Dropout(p=dropout)]

        if rnnt_type == 'warp-transducer':
            from warprnnt_pytorch import RNNTLoss

            self.rnnt_loss = RNNTLoss(blank=blank)
        else:
            raise NotImplementedError

        self.lin_enc = torch.nn.Linear(eprojs, joint_dim)
        self.lin_dec = torch.nn.Linear(dunits, joint_dim, bias=False)
        self.lin_out = torch.nn.Linear(joint_dim, odim)

        self.dlayers = dlayers
        self.dunits = dunits
        self.dtype = dtype
        self.embed_dim = embed_dim
        self.joint_dim = joint_dim
        self.odim = odim

        self.rnnt_type = rnnt_type

        self.ignore_id = -1
        self.blank = blank

    def zero_state(self, ey):
        """Initialize decoder states.

        Args:
            ey (torch.Tensor): batch of input features (B, Lmax, Emb_dim)

        Returns:
            (list): list of L zero-init hidden and cell state (1, B, Hdec)

        """
        z_list = [ey.new_zeros(1, ey.size(0), self.dunits)]
        c_list = [ey.new_zeros(1, ey.size(0), self.dunits)]

        for _ in six.moves.range(1, self.dlayers):
            z_list.append(ey.new_zeros(1, ey.size(0), self.dunits))
            c_list.append(ey.new_zeros(1, ey.size(0), self.dunits))

        return (z_list, c_list)

    def rnn_forward(self, ey, dstate):
        """RNN forward.

        Args:
            ey (torch.Tensor): batch of input features (B, Lmax, Emb_dim)
            dstate (list): list of L input hidden and cell state (1, B, Hdec)

        Returns:
            output (torch.Tensor): batch of output features (B, Lmax, Hdec)
            dstate (list): list of L output hidden and cell state (1, B, Hdec)

        """
        if dstate is None:
            z_prev, c_prev = self.zero_state(ey)
        else:
            z_prev, c_prev = dstate

        z_list, c_list = self.zero_state(ey)
        if self.dtype == "lstm":
            y, (z_list[0], c_list[0]) = self.decoder[0](ey, (z_prev[0], c_prev[0]))

            for l in six.moves.range(1, self.dlayers):
                y, (z_list[l], c_list[l]) = self.decoder[l](y, (z_prev[l], c_prev[l]))
        else:
            y, z_list[0] = self.decoder[0](ey, z_prev[0])

            for l in six.moves.range(1, self.dlayers):
                y, z_list[l] = self.decoder[l](y, z_prev[l])

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

    def forward(self, hs_pad, hlens, ys_pad):
        """Forward function for transducer.

        Args:
            hs_pad (torch.Tensor): batch of padded hidden state sequences (B, Tmax, D)
            hlens (torch.Tensor): batch of lengths of hidden state sequences (B)
            ys_pad (torch.Tensor): batch of padded character id sequence tensor (B, Lmax)

        Returns:
           loss (float): rnnt loss value

        """
        ys = [y[y != self.ignore_id] for y in ys_pad]

        hlens = list(map(int, hlens))

        blank = ys[0].new([self.blank])

        ys_in = [torch.cat([blank, y], dim=0) for y in ys]
        ys_in_pad = pad_list(ys_in, self.blank)

        eys = self.dropout_embed(self.embed(ys_in_pad))

        h_dec, _ = self.rnn_forward(eys, None)

        h_enc = hs_pad.unsqueeze(2)
        h_dec = h_dec.unsqueeze(1)

        z = self.joint(h_enc, h_dec)
        y = pad_list(ys, self.blank).type(torch.int32)

        z_len = to_device(self, torch.IntTensor(hlens))
        y_len = to_device(self, torch.IntTensor([_y.size(0) for _y in ys]))

        loss = to_device(self, self.rnnt_loss(z, y, z_len, y_len))

        return loss

    def recognize(self, h, recog_args):
        """Greedy search implementation.

        Args:
            h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
            recog_args (Namespace): argument Namespace containing options

        Returns:
            hyp (list of dicts): 1-best decoding results

        """
        hyp = {'score': 0.0, 'yseq': [self.blank]}

        ey = torch.zeros((1, 1, self.embed_dim))
        y, (z_list, c_list) = self.rnn_forward(ey, None)

        for hi in h:
            ytu = F.log_softmax(self.joint(hi, y[0][0]), dim=0)
            logp, pred = torch.max(ytu, dim=0)

            if pred != self.blank:
                hyp['yseq'].append(int(pred))
                hyp['score'] += float(logp)

                eys = torch.full((1, 1), hyp['yseq'][-1], dtype=torch.long)
                ey = self.dropout_embed(self.embed(eys))

                y, (z_list, c_list) = self.rnn_forward(ey, (z_list, c_list))

        return [hyp]

    def recognize_beam(self, h, recog_args, rnnlm=None):
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

        ey = torch.zeros((1, 1, self.embed_dim))
        y, dstate = self.rnn_forward(ey, None)

        if rnnlm:
            kept_hyps = [{'score': 0.0, 'yseq': [self.blank], 'dstate': dstate, 'lm_state': None}]
        else:
            kept_hyps = [{'score': 0.0, 'yseq': [self.blank], 'dstate': dstate}]

        for i, hi in enumerate(h):
            hyps = kept_hyps
            kept_hyps = []

            while True:
                new_hyp = max(hyps, key=lambda x: x['score'])
                hyps.remove(new_hyp)

                vy = to_device(self, torch.full((1, 1), new_hyp['yseq'][-1], dtype=torch.long))
                ey = self.dropout_embed(self.embed(vy))

                y, dstate = self.rnn_forward(ey, new_hyp['dstate'])

                ytu = F.log_softmax(self.joint(hi, y[0][0]), dim=0)

                if rnnlm:
                    rnnlm_state, rnnlm_scores = rnnlm.predict(new_hyp['lm_state'], vy[0])

                for k in six.moves.range(self.odim):
                    beam_hyp = {'score': new_hyp['score'] + float(ytu[k]),
                                'yseq': new_hyp['yseq'][:],
                                'dstate': new_hyp['dstate']}
                    if rnnlm:
                        beam_hyp['lm_state'] = new_hyp['lm_state']

                    if k == self.blank:
                        kept_hyps.append(beam_hyp)
                    else:
                        beam_hyp['dstate'] = dstate
                        beam_hyp['yseq'].append(int(k))

                        if rnnlm:
                            beam_hyp['lm_state'] = rnnlm_state
                            beam_hyp['score'] += recog_args.lm_weight * rnnlm_scores[0][ytu[k]]

                        hyps.append(beam_hyp)

                if len(kept_hyps) >= k_range:
                    break

        if normscore:
            nbest_hyps = sorted(
                kept_hyps, key=lambda x: x['score'] / len(x['yseq']), reverse=True)[:nbest]
        else:
            nbest_hyps = sorted(
                kept_hyps, key=lambda x: x['score'], reverse=True)[:nbest]

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
        rnnt_type (str): type of rnnt implementation

    """

    def __init__(self, eprojs, odim, dtype, dlayers, dunits, blank, att,
                 embed_dim, joint_dim, dropout=0.0, dropout_embed=0.0,
                 rnnt_type='warp-transducer'):
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

        if rnnt_type == 'warp-transducer':
            from warprnnt_pytorch import RNNTLoss

            self.rnnt_loss = RNNTLoss(blank=blank)
        else:
            raise NotImplementedError

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

        self.rnnt_type = rnnt_type

        self.ignore_id = -1
        self.blank = blank

    def zero_state(self, ey):
        """Initialize decoder states.

        Args:
            ey (torch.Tensor): batch of input features (B, Lmax, Emb_dim)

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
                    self.dropout_dec[l - 1](z_list[l - 1]), (z_prev[l], c_prev[l]))
        else:
            z_list[0] = self.decoder[0](ey, z_prev[0])

            for l in six.moves.range(1, self.dlayers):
                z_list[l] = self.decoder[l](self.dropout_dec[l - 1](z_list[l - 1]), z_prev[l])
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

    def forward(self, hs_pad, hlens, ys_pad):
        """Forward function for transducer with attention.

        Args:
            hs_pad (torch.Tensor): batch of padded hidden state sequences (B, Tmax, D)
            hlens (torch.Tensor): batch of lengths of hidden state sequences (B)
            ys_pad (torch.Tensor): batch of padded character id sequence tensor (B, Lmax)

        Returns:
           loss (torch.Tensor): rnnt-att loss value

        """
        ys = [y[y != self.ignore_id] for y in ys_pad]

        hlens = list(map(int, hlens))

        blank = ys[0].new([self.blank])

        ys_in = [torch.cat([blank, y], dim=0) for y in ys]
        ys_in_pad = pad_list(ys_in, self.blank)

        olength = ys_in_pad.size(1)

        att_w = None
        self.att[0].reset()

        z_list, c_list = self.zero_state(hs_pad)
        eys = self.dropout_emb(self.embed(ys_in_pad))

        z_all = []
        for i in six.moves.range(olength):
            att_c, att_w = self.att[0](hs_pad, hlens, self.dropout_dec[0](z_list[0]), att_w)

            ey = torch.cat((eys[:, i, :], att_c), dim=1)
            y, (z_list, c_list) = self.rnn_forward(ey, (z_list, c_list))
            z_all.append(y)

        h_dec = torch.stack(z_all, dim=1)

        h_enc = hs_pad.unsqueeze(2)
        h_dec = h_dec.unsqueeze(1)

        z = self.joint(h_enc, h_dec)
        y = pad_list(ys, self.blank).type(torch.int32)

        z_len = to_device(self, torch.IntTensor(hlens))
        y_len = to_device(self, torch.IntTensor([_y.size(0) for _y in ys]))

        loss = to_device(self, self.rnnt_loss(z, y, z_len, y_len))

        return loss

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

        att_c, att_w = self.att[0](h.unsqueeze(0), [h.size(0)],
                                   self.dropout_dec[0](z_list[0]), None)

        ey = torch.cat((eys, att_c), dim=1)

        hyp = {'score': 0.0, 'yseq': [self.blank]}

        y, (z_list, c_list) = self.rnn_forward(ey, (z_list, c_list))

        for hi in h:
            ytu = F.log_softmax(self.joint(hi, y[0]), dim=0)
            logp, pred = torch.max(ytu, dim=0)

            if pred != self.blank:
                hyp['yseq'].append(int(pred))
                hyp['score'] += float(logp)

                eys = torch.full((1, 1), hyp['yseq'][-1], dtype=torch.long)
                ey = self.dropout_emb(self.embed(eys))
                att_c, att_w = self.att[0](h.unsqueeze(0), [h.size(0)],
                                           self.dropout_dec[0](z_list[0]),
                                           att_w)
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

        att_c, att_w = self.att[0](h.unsqueeze(0), [h.size(0)],
                                   self.dropout_dec[0](z_list[0]), None)

        ey = torch.cat((eys, att_c), dim=1)
        _, (z_list, c_list) = self.rnn_forward(ey, None)

        if rnnlm:
            kept_hyps = [{'score': 0.0, 'yseq': [self.blank], 'z_prev': z_list,
                          'c_prev': c_list, 'a_prev': None, 'lm_state': None}]
        else:
            kept_hyps = [{'score': 0.0, 'yseq': [self.blank], 'z_prev': z_list,
                          'c_prev': c_list, 'a_prev': None}]

        for i, hi in enumerate(h):
            hyps = kept_hyps
            kept_hyps = []

            while True:
                new_hyp = max(hyps, key=lambda x: x['score'])
                hyps.remove(new_hyp)

                vy = to_device(self, torch.full((1, 1), new_hyp['yseq'][-1], dtype=torch.long))
                ey = self.dropout_emb(self.embed(vy))

                att_c, att_w = self.att[0](h.unsqueeze(0), [h.size(0)],
                                           self.dropout_dec[0](new_hyp['z_prev'][0]),
                                           new_hyp['a_prev'])

                ey = torch.cat((ey[0], att_c), dim=1)
                y, (z_list, c_list) = self.rnn_forward(ey, (new_hyp['z_prev'], new_hyp['c_prev']))
                ytu = F.log_softmax(self.joint(hi, y[0]), dim=0)

                if rnnlm:
                    rnnlm_state, rnnlm_scores = rnnlm.predict(new_hyp['lm_state'], vy[0])

                for k in six.moves.range(self.odim):
                    beam_hyp = {'score': new_hyp['score'] + float(ytu[k]),
                                'yseq': new_hyp['yseq'][:],
                                'z_prev': new_hyp['z_prev'],
                                'c_prev': new_hyp['c_prev'],
                                'a_prev': new_hyp['a_prev']}
                    if rnnlm:
                        beam_hyp['lm_state'] = new_hyp['lm_state']

                    if k == self.blank:
                        kept_hyps.append(beam_hyp)
                    else:
                        beam_hyp['z_prev'] = z_list[:]
                        beam_hyp['c_prev'] = c_list[:]
                        beam_hyp['a_prev'] = att_w[:]
                        beam_hyp['yseq'].append(int(k))

                        if rnnlm:
                            beam_hyp['lm_state'] = rnnlm_state
                            beam_hyp['score'] += recog_args.lm_weight * rnnlm_scores[0][ytu[k]]

                        hyps.append(beam_hyp)

                if len(kept_hyps) >= k_range:
                    break

        if normscore:
            nbest_hyps = sorted(
                kept_hyps, key=lambda x: x['score'] / len(x['yseq']), reverse=True)[:nbest]
        else:
            nbest_hyps = sorted(
                kept_hyps, key=lambda x: x['score'], reverse=True)[:nbest]

        return nbest_hyps

    def calculate_all_attentions(self, hs_pad, hlens, ys_pad):
        """Calculate all of attentions.

        Args:
            hs_pad (torch.Tensor): batch of padded hidden state sequences (B, Tmax, D)
            hlens (torch.Tensor): batch of lengths of hidden state sequences (B)
            ys_pad (torch.Tensor): batch of padded character id sequence tensor (B, Lmax)

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
            att_c, att_w = self.att[0](hs_pad, hlens, self.dropout_dec[0](z_list[0]), att_w)
            ey = torch.cat((eys[:, i, :], att_c), dim=1)
            _, (z_list, c_list) = self.rnn_forward(ey, (z_list, c_list))

            att_ws.append(att_w)

        att_ws = att_to_numpy(att_ws, self.att[0])

        return att_ws


def decoder_for(args, odim, att=None, blank=0):
    """Transducer mode selector."""
    if args.rnnt_mode == 'rnnt':
        return DecoderRNNT(args.eprojs, odim, args.dtype, args.dlayers, args.dunits,
                           blank, args.dec_embed_dim, args.joint_dim,
                           args.dropout_rate_decoder, args.dropout_rate_embed_decoder,
                           args.rnnt_type)
    elif args.rnnt_mode == 'rnnt-att':
        return DecoderRNNTAtt(args.eprojs, odim, args.dtype, args.dlayers, args.dunits,
                              blank, att, args.dec_embed_dim, args.joint_dim,
                              args.dropout_rate_decoder, args.dropout_rate_embed_decoder,
                              args.rnnt_type)
