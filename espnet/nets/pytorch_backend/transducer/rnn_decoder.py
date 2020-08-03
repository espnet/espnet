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

    def init_state(self, ey):
        """Initialize decoder states.

        Args:
            ey (torch.Tensor): batch of input features (B, emb_dim / dec_dim)

        Returns:
            (tuple): batch of decoder states (L x (B, dec_dim), L x (B, dec_dim))

        """
        z_list = [
            to_device(self, torch.zeros(ey.size(0), self.dunits))
            for _ in range(self.dlayers)
        ]
        c_list = [
            to_device(self, torch.zeros(ey.size(0), self.dunits))
            for _ in range(self.dlayers)
        ]

        return (z_list, c_list)

    def rnn_forward(self, ey, state):
        """RNN forward.

        Args:
            ey (torch.Tensor): batch of input features (B, emb_dim)
            (tuple): (tuple): batch of decoder states (L x (B, dec_dim), L x (B, dec_dim))

        Returns:
            output (torch.Tensor): batch of output features (B, dec_dim)
            (tuple): (tuple): batch of decoder states (L x (B, dec_dim), L x (B, dec_dim))

        """
        if state is None:
            z_prev, c_prev = self.init_state(ey)
        else:
            z_prev, c_prev = state

        z_list, c_list = self.init_state(ey)

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

        state = self.init_state(hs_pad)
        eys = self.dropout_embed(self.embed(ys_in_pad))

        z_all = []
        for i in range(olength):
            y, state = self.rnn_forward(eys[:, i, :], state)
            z_all.append(y)
        h_dec = torch.stack(z_all, dim=1)

        h_enc = hs_pad.unsqueeze(2)
        h_dec = h_dec.unsqueeze(1)

        z = self.joint(h_enc, h_dec)

        return z

    def score(self, hyp, init_tensor=None):
        """Forward one step.

        Args:
            hyp (dict): hypothese

        Returns:
            y (torch.Tensor): decoder outputs (1, dec_dim)
            (tuple): decoder and attention states
                ((L x (1, dec_dim), (L x (1, dec_dim)), None)
            lm_tokens (torch.Tensor): input token id for LM (1)

        """
        vy = to_device(self, torch.full((1, 1), hyp["yseq"][-1], dtype=torch.long))
        lm_tokens = vy[0]

        ey = self.embed(vy)

        y, state = self.rnn_forward(ey[0], hyp["dec_state"])

        return y, (state, None), lm_tokens

    def batch_score(self, hyps, state):
        """Forward batch one step.

        Args:
            hyps (list of dict): batch of hypothesis
            state (tuple): batch of decoder and attention states
                (
                 (L x (B, dec_dim), (L x (B, dec_dim)),
                 None,
                 None,
                )

        Returns:
            tgt (torch.Tensor): decoder output (B, dec_dim)
            (tuple): batch of decoder and attention states
                ((L x (B, dec_dim), (L x (B, dec_dim)), None)
            lm_tokens (torch.Tensor): input token ids for LM (B)

        """
        tokens = [h["yseq"][-1] for h in hyps]
        batch = len(tokens)

        tokens = to_device(self, torch.LongTensor(tokens).view(batch))

        ey = self.embed(tokens)

        y, state = self.rnn_forward(ey, state[0])

        return y, (state, None), tokens

    def select_state(self, state, idx):
        """Get decoder state from batch for given id.

        Args:
            state (tuple): batch of decoder and attention states
                ((L x (B, dec_dim), (L x (B, dec_dim)), None)
            idx (int): index to extract state from batch state

        Returns:
            (tuple): decoder and attention states
                ((L x (1, dec_dim), (L x (1, dec_dim)), None)

        """
        z_list = [state[0][0][layer][idx] for layer in range(self.dlayers)]
        c_list = [state[0][1][layer][idx] for layer in range(self.dlayers)]

        return ((z_list, c_list), None)

    def create_batch_state(self, state, hyps):
        """Create batch of decoder states.

        Args:
            state (tuple): batch of decoder and attention states
                ((L x (B, dec_dim), (L x (B, dec_dim)), None)
            hyps (list): batch of hypothesis

        Returns:
            (tuple): batch of decoder and attention states
                ((L x (B, dec_dim), (L x (B, dec_dim)), None)

        """
        for layer in range(self.dlayers):
            state[0][0][layer] = torch.stack([h["dec_state"][0][layer] for h in hyps])
            state[0][1][layer] = torch.stack([h["dec_state"][1][layer] for h in hyps])

        return (state[0], None)
