"""RNN-Transducer implementation for training and decoding."""

import torch

from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.transducer.decoder_interface import (
    TransducerDecoderInterface,  # noqa: H301
)


class DecoderRNNT(TransducerDecoderInterface, torch.nn.Module):
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

        for _ in range(1, dlayers):
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

        for _ in range(1, self.dlayers):
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
        for i in range(olength):
            y, (z_list, c_list) = self.rnn_forward(eys[:, i, :], (z_list, c_list))
            z_all.append(y)
        h_dec = torch.stack(z_all, dim=1)

        h_enc = hs_pad.unsqueeze(2)
        h_dec = h_dec.unsqueeze(1)

        z = self.joint(h_enc, h_dec)

        return z

    def forward_one_step(self, hyp, init_tensor=None):
        """Forward one step.

        Args:
            hyp (dict): hypothese

        Returns:
            y (torch.Tensor): decoder outputs (1, dec_dim)
            state (tuple): tuple of list of L decoder states (1, dec_dim)
            lm_tokens (torch.Tensor): input token id for LM (1)

        """
        vy = to_device(self, torch.full((1, 1), hyp["yseq"][-1], dtype=torch.long))
        lm_tokens = vy[0]

        ey = self.embed(vy)

        y, state = self.rnn_forward(ey[0], hyp["dec_state"])

        return y, state, None, lm_tokens

    def forward_batch_one_step(self, hyps, state, att_w=None, att_params=None):
        """Forward batch one step.

        Args:
            hyps (list of dict): batch of hypothesis
            state (tuple): tuple of list of L decoder states (B, dec_dim)

        Returns:
            tgt (torch.Tensor): decoder output (B, dec_dim)
            new_state (tuple): tuple of list of L decoder states (B, dec_dim)
            lm_tokens (torch.Tensor): input token ids for LM (B)

        """
        tokens = [h["yseq"][-1] for h in hyps]
        batch = len(tokens)

        tokens = to_device(self, torch.LongTensor(tokens).view(batch))

        ey = self.embed(tokens)

        y, state = self.rnn_forward(ey, state)

        return y, state, None, tokens

    def get_idx_dec_state(self, state, idx, att_state=None):
        """Get decoder state from batch for given id.

        Args:
            state (tuple): tuple of list of L decoder states (B, dec_dim)
            idx (int): index to extract state from batch state

        Returns:
            state (tuple): tuple of list of L decoder states (dec_dim)

        """
        zlist = [state[0][layer][idx] for layer in range(self.dlayers)]
        clist = [state[1][layer][idx] for layer in range(self.dlayers)]

        return (zlist, clist), None

    def get_batch_dec_states(self, state, hyps):
        """Create batch of decoder states.

        Args:
            state (tuple): tuple of list of individual decoder states (B, dec_dim)
            hyps (list): batch of hypothesis

        Returns:
            state (tuple): tuple of list of L decoder states (B, dec_dim)

        """
        for layer in range(self.dlayers):
            state[0][layer] = torch.stack([h["dec_state"][0][layer] for h in hyps])
            state[1][layer] = torch.stack([h["dec_state"][1][layer] for h in hyps])

        return state, None
