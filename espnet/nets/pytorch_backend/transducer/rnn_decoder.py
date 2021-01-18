"""RNN-Transducer implementation for training and decoding."""

import torch

from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.transducer.joint_network import JointNetwork
from espnet.nets.transducer_decoder_interface import TransducerDecoderInterface


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
        joint_activation_type (int): joint network activation
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
        joint_activation_type="tanh",
        dropout=0.0,
        dropout_embed=0.0,
    ):
        """Transducer initializer."""
        super().__init__()

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

        self.joint_network = JointNetwork(
            odim, eprojs, dunits, joint_dim, joint_activation_type
        )

        self.dlayers = dlayers
        self.dunits = dunits
        self.dtype = dtype
        self.embed_dim = embed_dim
        self.joint_dim = joint_dim
        self.odim = odim

        self.ignore_id = -1
        self.blank = blank

    def init_state(self, init_tensor):
        """Initialize decoder states.

        Args:
            init_tensor (torch.Tensor): batch of input features
                (B, emb_dim / dec_dim)

        Returns:
            (tuple): batch of decoder states
                ([L x (B, dec_dim)], [L x (B, dec_dim)])

        """
        dtype = init_tensor.dtype
        z_list = [
            to_device(init_tensor, torch.zeros(init_tensor.size(0), self.dunits)).to(
                dtype
            )
            for _ in range(self.dlayers)
        ]
        c_list = [
            to_device(init_tensor, torch.zeros(init_tensor.size(0), self.dunits)).to(
                dtype
            )
            for _ in range(self.dlayers)
        ]

        return (z_list, c_list)

    def rnn_forward(self, ey, state):
        """RNN forward.

        Args:
            ey (torch.Tensor): batch of input features (B, emb_dim)
            state (tuple): batch of decoder states
                (L x (B, dec_dim), L x (B, dec_dim))

        Returns:
            y (torch.Tensor): batch of output features (B, dec_dim)
            (tuple): batch of decoder states
                (L x (B, dec_dim), L x (B, dec_dim))

        """
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

        h_enc = hs_pad.unsqueeze(2)

        h_dec = torch.stack(z_all, dim=1)
        h_dec = h_dec.unsqueeze(1)

        z = self.joint_network(h_enc, h_dec)

        return z

    def score(self, hyp, cache, init_tensor=None):
        """Forward one step.

        Args:
            hyp (dataclass): hypothesis
            cache (dict): states cache

        Returns:
            y (torch.Tensor): decoder outputs (1, dec_dim)
            state (tuple): decoder states
                ([L x (1, dec_dim)], [L x (1, dec_dim)]),
            (torch.Tensor): token id for LM (1)

        """
        vy = to_device(self, torch.full((1, 1), hyp.yseq[-1], dtype=torch.long))

        str_yseq = "".join([str(x) for x in hyp.yseq])

        if str_yseq in cache:
            y, state = cache[str_yseq]
        else:
            ey = self.embed(vy)

            y, state = self.rnn_forward(ey[0], hyp.dec_state)
            cache[str_yseq] = (y, state)

        return y, state, vy[0]

    def batch_score(self, hyps, batch_states, cache, init_tensor=None):
        """Forward batch one step.

        Args:
            hyps (list): batch of hypotheses
            batch_states (tuple): batch of decoder states
                ([L x (B, dec_dim)], [L x (B, dec_dim)])
            cache (dict): states cache

        Returns:
            batch_y (torch.Tensor): decoder output (B, dec_dim)
            batch_states (tuple): batch of decoder states
                ([L x (B, dec_dim)], [L x (B, dec_dim)])
            lm_tokens (torch.Tensor): batch of token ids for LM (B)

        """
        final_batch = len(hyps)

        tokens = []
        process = []
        done = [None for _ in range(final_batch)]

        for i, hyp in enumerate(hyps):
            str_yseq = "".join([str(x) for x in hyp.yseq])

            if str_yseq in cache:
                done[i] = cache[str_yseq]
            else:
                tokens.append(hyp.yseq[-1])
                process.append((str_yseq, hyp.dec_state))

        if process:
            batch = len(process)

            tokens = to_device(self, torch.LongTensor(tokens).view(batch))

            dec_state = self.init_state(torch.zeros((batch, self.dunits)))
            dec_state = self.create_batch_states(dec_state, [p[1] for p in process])

            ey = self.embed(tokens)

            y, dec_state = self.rnn_forward(ey, dec_state)

        j = 0
        for i in range(final_batch):
            if done[i] is None:
                new_state = self.select_state(dec_state, j)

                done[i] = (y[j], new_state)
                cache[process[j][0]] = (y[j], new_state)

                j += 1

        batch_states = self.create_batch_states(batch_states, [d[1] for d in done])
        batch_y = torch.stack([d[0] for d in done])

        lm_tokens = to_device(
            self, torch.LongTensor([h.yseq[-1] for h in hyps]).view(final_batch)
        )

        return batch_y, batch_states, lm_tokens

    def select_state(self, batch_states, idx):
        """Get decoder state from batch of states, for given id.

        Args:
            batch_states (tuple): batch of decoder states
                ([L x (B, dec_dim)], [L x (B, dec_dim)])
            idx (int): index to extract state from batch of states

        Returns:
            (tuple): decoder states for given id
                ([L x (1, dec_dim)], [L x (1, dec_dim)])

        """
        z_list = [batch_states[0][layer][idx] for layer in range(self.dlayers)]
        c_list = [batch_states[1][layer][idx] for layer in range(self.dlayers)]

        return (z_list, c_list)

    def create_batch_states(self, batch_states, l_states, l_tokens=None):
        """Create batch of decoder states.

        Args:
            batch_states (tuple): batch of decoder states
               ([L x (B, dec_dim)], [L x (B, dec_dim)])
            l_states (list): list of decoder states
                [B x ([L x (1, dec_dim)], [L x (1, dec_dim)])]

        Returns:
            batch_states (tuple): batch of decoder states
                ([L x (B, dec_dim)], [L x (B, dec_dim)])

        """
        for layer in range(self.dlayers):
            batch_states[0][layer] = torch.stack([s[0][layer] for s in l_states])
            batch_states[1][layer] = torch.stack([s[1][layer] for s in l_states])

        return batch_states
