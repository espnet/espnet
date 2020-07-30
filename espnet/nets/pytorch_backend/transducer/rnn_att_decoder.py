"""RNN-Transducer with attention implementation for training and decoding."""

import torch

from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.rnn.attentions import att_to_numpy
from espnet.nets.pytorch_backend.transducer.decoder_interface import (
    TransducerDecoderInterface,  # noqa: H301
)


class DecoderRNNTAtt(TransducerDecoderInterface, torch.nn.Module):
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

        for _ in range(1, dlayers):
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
            ey (torch.Tensor): batch of input features (B, (emb_dim + eprojs))

        Return:
            z_list : list of L zero-init hidden state (B, dec_dim)
            c_list : list of L zero-init cell state (B, dec_dim)

        """
        z_list = [ey.new_zeros(ey.size(0), self.dunits)]
        c_list = [ey.new_zeros(ey.size(0), self.dunits)]

        for _ in range(1, self.dlayers):
            z_list.append(ey.new_zeros(ey.size(0), self.dunits))
            c_list.append(ey.new_zeros(ey.size(0), self.dunits))

        return z_list, c_list

    def rnn_forward(self, ey, dstate):
        """RNN forward.

        Args:
            ey (torch.Tensor): batch of input features (B, (emb_dim + eprojs))
            dstate (list): list of L input hidden and cell state (B, dec_dim)
        Returns:
            y (torch.Tensor): decoder output for one step (B, dec_dim)
            (list): list of L output hidden and cell state (B, dec_dim)

        """
        if dstate is None:
            z_prev, c_prev = self.zero_state(ey)
        else:
            z_prev, c_prev = dstate

        z_list, c_list = self.zero_state(ey)

        if self.dtype == "lstm":
            z_list[0], c_list[0] = self.decoder[0](ey, (z_prev[0], c_prev[0]))

            for i in range(1, self.dlayers):
                z_list[i], c_list[i] = self.decoder[i](
                    self.dropout_dec[i - 1](z_list[i - 1]), (z_prev[i], c_prev[i])
                )
        else:
            z_list[0] = self.decoder[0](ey, z_prev[0])

            for i in range(1, self.dlayers):
                z_list[i] = self.decoder[i](
                    self.dropout_dec[i - 1](z_list[i - 1]), z_prev[i]
                )
        y = self.dropout_dec[-1](z_list[-1])

        return y, (z_list, c_list)

    def joint(self, h_enc, h_dec):
        """Joint computation of z.

        Args:
            h_enc (torch.Tensor): batch of expanded hidden state (B, T, 1, enc_dim)
            h_dec (torch.Tensor): batch of expanded hidden state (B, 1, U, dec_dim)

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
        for i in range(olength):
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

    def forward_one_step(self, hyp, init_tensor):
        """Forward one step.

        Args:
            hyp (dict): hypothese
            init_tensor (torch.Tensor): initial tensor (1, max_len, dec_dim)

        Returns:
            y (torch.Tensor): decoder outputs (1, dec_dim)
            state (tuple): tuple of list of L decoder states (1, dec_dim)
            att_w (torch.Tensor): attention weights (1, max_len)
            lm_tokens (torch.Tensor): input token id for LM (1)

        """
        vy = to_device(self, torch.full((1, 1), hyp["yseq"][-1], dtype=torch.long))
        lm_tokens = vy[0]

        ey = self.embed(vy)

        att_c, att_w = self.att[0](
            init_tensor, [init_tensor.size(1)], hyp["dec_state"][0][0], hyp["att_w"],
        )

        ey = torch.cat((ey[0], att_c), dim=1)

        y, state = self.rnn_forward(ey, hyp["dec_state"])

        return y, state, att_w, lm_tokens

    def forward_batch_one_step(self, hyps, state, att_w, att_params):
        """Forward batch one step.

        Args:
            hyps (list of dict): batch of hypothesis
            state (tuple): tuple of list of L decoder states (B, dec_dim)
            att_w (torch.Tensor): batch of attention weights (B, max_len)
            att_params (list): attention parameters

        Returns:
            tgt (torch.Tensor): decoder output (B, dec_dim)
            new_state (tuple): tuple of list of L decoder states (B, dec_dim)
            att_w (torch.Tensor): batch of attention weights (B, max_len)
            lm_tokens (torch.Tensor): input token ids for LM (B)

        """
        tokens = [h["yseq"][-1] for h in hyps]
        batch = len(tokens)

        tokens = to_device(self, torch.LongTensor(tokens).view(batch))

        ey = self.embed(tokens)

        att_c, att_w = self.att[0](att_params[1], att_params[0], state[0][0], att_w)

        ey = torch.cat((ey, att_c), dim=1)

        y, state = self.rnn_forward(ey, state)

        return y, state, att_w, tokens

    def get_idx_dec_state(self, state, idx, att_state):
        """Get decoder and attention state from batch for given id.

        Args:
            state (tuple): tuple of list of L decoder states (B, dec_dim)
            idx (int): index to extract state from batch state
            att_state (torch.Tensor): attention state (B, max_len)

        Returns:
            state (tuple): tuple of list of L decoder states (dec_dim)
            att_w (torch.Tensor): attention_state (max_len)

        """
        zlist = [state[0][layer][idx] for layer in range(self.dlayers)]
        clist = [state[1][layer][idx] for layer in range(self.dlayers)]

        att_w = att_state[idx] if att_state is not None else att_state

        return (zlist, clist), att_w

    def get_batch_dec_states(self, state, hyps):
        """Create batch of decoder and attention states.

        Args:
            state (tuple): tuple of list of decoder states (B, dec_dim)
            hyps (list): batch of hypothesis

        Returns:
            state (tuple): tuple of list of L decoder states (B, dec_dim)
            att_w (torch.Tensor): batch of attention weights (B, max_len)
        """
        for layer in range(self.dlayers):
            state[0][layer] = torch.stack([h["dec_state"][0][layer] for h in hyps])
            state[1][layer] = torch.stack([h["dec_state"][1][layer] for h in hyps])

        att_w = torch.stack([h["att_w"] for h in hyps])

        return state, att_w

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

        for i in range(olength):
            att_c, att_w = self.att[0](
                hs_pad, hlens, self.dropout_dec[0](z_list[0]), att_w
            )
            ey = torch.cat((eys[:, i, :], att_c), dim=1)
            _, (z_list, c_list) = self.rnn_forward(ey, (z_list, c_list))

            att_ws.append(att_w)

        att_ws = att_to_numpy(att_ws, self.att[0])

        return att_ws
