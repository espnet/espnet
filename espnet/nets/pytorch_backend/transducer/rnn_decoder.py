"""RNN decoder for transducer-based models."""

import torch

from espnet.nets.transducer_decoder_interface import TransducerDecoderInterface


class DecoderRNNT(TransducerDecoderInterface, torch.nn.Module):
    """RNN-T Decoder module.

    Args:
        odim (int): dimension of outputs
        dtype (str): gru or lstm
        dlayers (int): # prediction layers
        dunits (int): # prediction units
        blank (int): blank symbol id
        embed_dim (int): dimension of embeddings
        dropout (float): dropout rate
        dropout_embed (float): embedding dropout rate

    """

    def __init__(
        self,
        odim,
        dtype,
        dlayers,
        dunits,
        blank,
        embed_dim,
        dropout=0.0,
        dropout_embed=0.0,
    ):
        """Transducer initializer."""
        super().__init__()

        self.embed = torch.nn.Embedding(odim, embed_dim, padding_idx=blank)
        self.dropout_embed = torch.nn.Dropout(p=dropout_embed)

        dec_net = torch.nn.LSTM if dtype == "lstm" else torch.nn.GRU

        self.decoder = torch.nn.ModuleList(
            [dec_net(embed_dim, dunits, 1, batch_first=True)]
        )
        self.dropout_dec = torch.nn.Dropout(p=dropout)

        for _ in range(1, dlayers):
            self.decoder += [dec_net(dunits, dunits, 1, batch_first=True)]

        self.dlayers = dlayers
        self.dunits = dunits
        self.dtype = dtype

        self.odim = odim

        self.ignore_id = -1
        self.blank = blank

        self.multi_gpus = torch.cuda.device_count() > 1

    def set_device(self, device):
        """Set GPU device to use.

        Args:
            device (torch.device): device id

        """
        self.device = device

    def set_data_type(self, data_type):
        """Set GPU device to use.

        Args:
            data_type (torch.dtype): Tensor data type

        """
        self.data_type = data_type

    def init_state(self, batch_size):
        """Initialize decoder states.

        Args:
            batch_size (int): Batch size

        Returns:
            (tuple): batch of decoder states
                ((L, B, dec_dim), (L, B, dec_dim))

        """
        h_n = torch.zeros(
            self.dlayers,
            batch_size,
            self.dunits,
            device=self.device,
            dtype=self.data_type,
        )

        if self.dtype == "lstm":
            c_n = torch.zeros(
                self.dlayers,
                batch_size,
                self.dunits,
                device=self.device,
                dtype=self.data_type,
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
        h_next, c_next = self.init_state(y.size(0))

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
        self.set_device(hs_pad.device)
        self.set_data_type(hs_pad.dtype)

        state = self.init_state(hs_pad.size(0))
        eys = self.dropout_embed(self.embed(ys_in_pad))

        h_dec, _ = self.rnn_forward(eys, state)

        return h_dec

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
        vy = torch.full((1, 1), hyp.yseq[-1], dtype=torch.long, device=self.device)

        str_yseq = "".join(list(map(str, hyp.yseq)))

        if str_yseq in cache:
            y, state = cache[str_yseq]
        else:
            ey = self.embed(vy)

            y, state = self.rnn_forward(ey, hyp.dec_state)
            cache[str_yseq] = (y, state)

        return y[0][0], state, vy[0]

    def batch_score(self, hyps, batch_states, cache, use_lm):
        """Forward batch one step.

        Args:
            hyps (list): batch of hypotheses
            batch_states (tuple): batch of decoder states
                ((L, B, dec_dim), (L, B, dec_dim))
            cache (dict): states cache
            use_lm (bool): whether a LM is used for decoding

        Returns:
            batch_y (torch.Tensor): decoder output (B, dec_dim)
            batch_states (tuple): batch of decoder states
                ((L, B, dec_dim), (L, B, dec_dim))
            lm_tokens (torch.Tensor): batch of token ids for LM (B)

        """
        final_batch = len(hyps)

        process = []
        done = [None] * final_batch

        for i, hyp in enumerate(hyps):
            str_yseq = "".join(list(map(str, hyp.yseq)))

            if str_yseq in cache:
                done[i] = cache[str_yseq]
            else:
                process.append((str_yseq, hyp.yseq[-1], hyp.dec_state))

        if process:
            tokens = torch.LongTensor([[p[1]] for p in process], device=self.device)
            dec_state = self.create_batch_states(
                self.init_state(tokens.size(0)), [p[2] for p in process]
            )

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

        if use_lm:
            lm_tokens = torch.LongTensor([h.yseq[-1] for h in hyps], device=self.device)

            return batch_y, batch_states, lm_tokens

        return batch_y, batch_states, None

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
        return (
            batch_states[0][:, idx : idx + 1, :],
            batch_states[1][:, idx : idx + 1, :] if self.dtype == "lstm" else None,
        )

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
        return (
            torch.cat([s[0] for s in l_states], dim=1),
            torch.cat([s[1] for s in l_states], dim=1)
            if self.dtype == "lstm"
            else None,
        )
