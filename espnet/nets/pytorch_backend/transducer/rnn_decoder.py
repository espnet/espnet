"""RNN-Transducer implementation for training and decoding."""

import torch

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
        embed_dim (int): dimension of embeddings
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
            dec_net = torch.nn.LSTM
        else:
            dec_net = torch.nn.GRU

        self.decoder = torch.nn.ModuleList(
            [dec_net(embed_dim, dunits, 1, batch_first=True)]
        )
        self.dropout_dec = torch.nn.Dropout(p=dropout)

        for _ in range(1, dlayers):
            self.decoder += [dec_net(dunits, dunits, 1, batch_first=True)]

        self.joint_network = JointNetwork(
            odim, eprojs, dunits, joint_dim, joint_activation_type
        )

        self.dlayers = dlayers
        self.dunits = dunits
        self.dtype = dtype
        self.joint_dim = joint_dim
        self.odim = odim

        self.ignore_id = -1
        self.blank = blank

    def init_state(self, batch_size, device, dtype):
        """Initialize decoder states.

        Args:
            batch_size (int): Batch size
            device (torch.device): device id
            dtype (torch.dtype): Tensor data type

        Returns:
            (tuple): batch of decoder states
                ((L, B, dec_dim), (L, B, dec_dim))

        """
        h_n = torch.zeros(
            self.dlayers,
            batch_size,
            self.dunits,
            device=device,
            dtype=dtype,
        )

        if self.dtype == "lstm":
            c_n = torch.zeros(
                self.dlayers,
                batch_size,
                self.dunits,
                device=device,
                dtype=dtype,
            )

            return (h_n, c_n)

        return (h_n, None)

    def rnn_forward(self, y, state):
        """RNN forward.

        Args:
            y (torch.Tensor): batch of input features (B, emb_dim)
            state (tuple): batch of decoder states
                ((L, B, dec_dim), (L, B, dec_dim))

        Returns:
            y (torch.Tensor): batch of output features (B, dec_dim)
            (tuple): batch of decoder states
                (L, B, dec_dim), (L, B, dec_dim))

        """
        h_prev, c_prev = state
        h_next, c_next = self.init_state(y.size(0), y.device, y.dtype)

        for layer in range(self.dlayers):
            if self.dtype == "lstm":
                y, (
                    h_next[layer : layer + 1],
                    c_next[layer : layer + 1],
                ) = self.decoder[layer](
                    y, hx=(h_prev[layer : layer + 1], c_prev[layer : layer + 1])
                )
            else:
                y, h_next[layer : layer + 1] = self.decoder[layer](
                    y, hx=h_prev[layer : layer + 1]
                )

            y = self.dropout_dec(y)

        return y, (h_next, c_next)

    def forward(self, hs_pad, ys_in_pad):
        """Forward function for transducer.

        Args:
            hs_pad (torch.Tensor):
                batch of padded hidden state sequences (B, Tmax, D)
            ys_in_pad (torch.Tensor):
                batch of padded character id sequence tensor (B, Lmax+1)

        Returns:
            z (torch.Tensor): output (B, T, U, odim)

        """
        batch = hs_pad.size(0)

        state = self.init_state(batch, hs_pad.device, hs_pad.dtype)
        eys = self.dropout_embed(self.embed(ys_in_pad))

        h_dec, _ = self.rnn_forward(eys, state)

        h_enc = hs_pad.unsqueeze(2)
        h_dec = h_dec.unsqueeze(1)

        z = self.joint_network(h_enc, h_dec)

        return z

    def score(self, hyp, cache):
        """Forward one step.

        Args:
            hyp (dataclass): hypothesis
            cache (dict): states cache

        Returns:
            y (torch.Tensor): decoder outputs (1, dec_dim)
            state (tuple): decoder states
                ((L, 1, dec_dim), (L, 1, dec_dim)),
            (torch.Tensor): token id for LM (1,)

        """
        device = next(self.parameters()).device

        vy = torch.full((1, 1), hyp.yseq[-1], dtype=torch.long, device=device)

        str_yseq = "".join([str(x) for x in hyp.yseq])

        if str_yseq in cache:
            y, state = cache[str_yseq]
        else:
            ey = self.embed(vy)

            y, state = self.rnn_forward(ey, hyp.dec_state)
            cache[str_yseq] = (y, state)

        return y[0][0], state, vy[0]

    def batch_score(self, hyps, batch_states, cache):
        """Forward batch one step.

        Args:
            hyps (list): batch of hypotheses
            batch_states (tuple): batch of decoder states
                ((L, B, dec_dim), (L, B, dec_dim))
            cache (dict): states cache

        Returns:
            batch_y (torch.Tensor): decoder output (B, dec_dim)
            batch_states (tuple): batch of decoder states
                ((L, B, dec_dim), (L, B, dec_dim))
            lm_tokens (torch.Tensor): batch of token ids for LM (B)

        """
        final_batch = len(hyps)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        process = []
        done = [None for _ in range(final_batch)]

        for i, hyp in enumerate(hyps):
            str_yseq = "".join([str(x) for x in hyp.yseq])

            if str_yseq in cache:
                done[i] = cache[str_yseq]
            else:
                process.append((str_yseq, hyp.yseq[-1], hyp.dec_state))

        if process:
            batch = len(process)
            _tokens = [p[1] for p in process]
            _states = [p[2] for p in process]

            tokens = torch.LongTensor(_tokens).view(batch, 1).to(device=device)

            dec_state = self.init_state(batch, device, dtype)
            dec_state = self.create_batch_states(dec_state, _states)

            ey = self.embed(tokens)
            y, dec_state = self.rnn_forward(ey, dec_state)

        j = 0
        for i in range(final_batch):
            if done[i] is None:
                new_state = self.select_state(dec_state, j)

                done[i] = (y[j], new_state)
                cache[process[j][0]] = (y[j], new_state)

                j += 1

        batch_y = torch.cat([d[0] for d in done], dim=0)
        batch_states = self.create_batch_states(batch_states, [d[1] for d in done])

        lm_tokens = (
            torch.LongTensor([h.yseq[-1] for h in hyps])
            .view(final_batch)
            .to(device=device)
        )

        return batch_y, batch_states, lm_tokens

    def select_state(self, batch_states, idx):
        """Get decoder state from batch of states, for given id.

        Args:
            batch_states (tuple): batch of decoder states
                ((L, B, dec_dim), (L, B, dec_dim))
            idx (int): index to extract state from batch of states

        Returns:
            (tuple): decoder states for given id
                ((L, 1, dec_dim), (L, 1, dec_dim))

        """
        h_idx = batch_states[0][:, idx : idx + 1, :]

        if self.dtype == "lstm":
            c_idx = batch_states[1][:, idx : idx + 1, :]

            return (h_idx, c_idx)

        return (h_idx, None)

    def create_batch_states(self, batch_states, l_states, l_tokens=None):
        """Create batch of decoder states.

        Args:
            batch_states (tuple): batch of decoder states
               ((L, B, dec_dim), (L, B, dec_dim))
            l_states (list): list of decoder states
               [L x ((1, dec_dim), (1, dec_dim))]

        Returns:
            batch_states (tuple): batch of decoder states
                ((L, B, dec_dim), (L, B, dec_dim))

        """
        h_n = torch.cat([s[0] for s in l_states], dim=1)

        if self.dtype == "lstm":
            c_n = torch.cat([s[1] for s in l_states], dim=1)

            return (h_n, c_n)

        return (h_n, None)
