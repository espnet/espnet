"""RNN-Transducer with attention implementation for training and decoding."""

import torch

from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.rnn.attentions import att_to_numpy
from espnet.nets.pytorch_backend.transducer.joint_network import JointNetwork
from espnet.nets.transducer_decoder_interface import TransducerDecoderInterface


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
        att,
        embed_dim,
        joint_dim,
        joint_activation_type="tanh",
        dropout=0.0,
        dropout_embed=0.0,
    ):
        """Transducer with attention initializer."""
        super().__init__()

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

        self.joint_network = JointNetwork(
            odim, eprojs, dunits, joint_dim, joint_activation_type
        )

        self.att = att

        self.dtype = dtype
        self.dlayers = dlayers
        self.dunits = dunits
        self.embed_dim = embed_dim
        self.joint_dim = joint_dim
        self.odim = odim

        self.ignore_id = -1
        self.blank = blank

    def init_state(self, init_tensor):
        """Initialize decoder states.

        Args:
            init_tensor (torch.Tensor): batch of input features
                (B, (emb_dim + eprojs) / dec_dim)

        Return:
            (tuple): batch of decoder and attention states
                ([L x (B, dec_dim)], [L x (B, dec_dim)], None)

        """
        z_list = [
            to_device(init_tensor, torch.zeros(init_tensor.size(0), self.dunits))
            for _ in range(self.dlayers)
        ]
        c_list = [
            to_device(init_tensor, torch.zeros(init_tensor.size(0), self.dunits))
            for _ in range(self.dlayers)
        ]

        return ((z_list, c_list), None)

    def rnn_forward(self, ey, state):
        """RNN forward.

        Args:
            ey (torch.Tensor): batch of input features (B, (emb_dim + eprojs))
            state (tuple): batch of decoder states
                ([L x (B, dec_dim)], [L x (B, dec_dim)])
        Returns:
            y (torch.Tensor): decoder output for one step (B, dec_dim)
            (tuple): batch of decoder states
                ([L x (B, dec_dim)], [L x (B, dec_dim)])

        """
        z_prev, c_prev = state
        (z_list, c_list), _ = self.init_state(ey)

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

        self.att[0].reset()

        state, att_w = self.init_state(hs_pad)
        eys = self.dropout_emb(self.embed(ys_in_pad))

        z_all = []
        for i in range(olength):
            att_c, att_w = self.att[0](
                hs_pad, hlens, self.dropout_dec[0](state[0][0]), att_w
            )

            ey = torch.cat((eys[:, i, :], att_c), dim=1)

            y, state = self.rnn_forward(ey, state)
            z_all.append(y)

        h_dec = torch.stack(z_all, dim=1)

        h_enc = hs_pad.unsqueeze(2)
        h_dec = h_dec.unsqueeze(1)

        z = self.joint_network(h_enc, h_dec)

        return z

    def score(self, hyp, cache, init_tensor):
        """Forward one step.

        Args:
            hyp (dataclass): hypothese
            cache (dict): states cache
            init_tensor (torch.Tensor): initial tensor (1, max_len, dec_dim)

        Returns:
            y (torch.Tensor): decoder outputs (1, dec_dim)
            (tuple): decoder and attention states
                (([L x (1, dec_dim)], [L x (1, dec_dim)]), (1, max_len))
            lm_tokens (torch.Tensor): token id for LM (1)

        """
        vy = to_device(self, torch.full((1, 1), hyp.yseq[-1], dtype=torch.long))

        str_yseq = "".join([str(x) for x in hyp.yseq])

        if str_yseq in cache:
            y, state = cache[str_yseq]
        else:
            ey = self.embed(vy)

            att_c, att_w = self.att[0](
                init_tensor,
                [init_tensor.size(1)],
                hyp.dec_state[0][0][0],
                hyp.dec_state[1],
            )

            ey = torch.cat((ey[0], att_c), dim=1)

            y, dec_state = self.rnn_forward(ey, hyp.dec_state[0])
            state = (dec_state, att_w)

            cache[str_yseq] = (y, state)

        return y, state, vy[0]

    def batch_score(self, hyps, batch_states, cache, init_tensor):
        """Forward batch one step.

        Args:
            hyps (list): batch of hypotheses
            batch_states (tuple): batch of decoder and attention states
                (([L x (B, dec_dim)], [L x (B, dec_dim)]), (B, max_len))
            cache (dict): states cache
            init_tensor: encoder outputs for att. computation (1, max_enc_len)

        Returns:
            batch_y (torch.Tensor): decoder output (B, dec_dim)
            batch_states (tuple): batch of decoder and attention states
                (([L x (B, dec_dim)], [L x (B, dec_dim)]), (B, max_len))
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
            batch = len(tokens)

            tokens = to_device(self, torch.LongTensor(tokens).view(batch))

            state = self.init_state(init_tensor)
            dec_state = self.create_batch_states(state, [p[1] for p in process])

            ey = self.embed(tokens)

            enc_hs = init_tensor.expand(batch, -1, -1)
            enc_len = [init_tensor.squeeze(0).size(0)] * batch

            att_c, att_w = self.att[0](enc_hs, enc_len, state[0][0][0], state[1])

            ey = torch.cat((ey, att_c), dim=1)

            y, dec_state = self.rnn_forward(ey, state[0])

        j = 0

        for i in range(final_batch):
            if done[i] is None:
                new_state = self.select_state((dec_state, att_w), j)

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
        """Get decoder and attention state from batch of states, for given id.

        Args:
            batch_states (tuple): batch of decoder and attention states
                (([L x (B, dec_dim)], [L x (B, dec_dim)]), (B, max_len))
            idx (int): index to extract state from batch of states

        Returns:
            (tuple): decoder and attention states
                (([L x (1, dec_dim)], [L x (1, dec_dim)]), (1, max_len))

        """
        z_list = [batch_states[0][0][layer][idx] for layer in range(self.dlayers)]
        c_list = [batch_states[0][1][layer][idx] for layer in range(self.dlayers)]

        att_state = (
            batch_states[1][idx] if batch_states[1] is not None else batch_states[1]
        )

        return ((z_list, c_list), att_state)

    def create_batch_states(self, batch_states, l_states, l_tokens=None):
        """Create batch of decoder and attention states.

        Args:
            batch_states (tuple): batch of decoder and attention states
                (([L x (B, dec_dim)], [L x (B, dec_dim)]), (B, max_len))
            l_states (list): list of single decoder and attention states
                [B x (([L x (1, dec_dim)], [L x (1, dec_dim)]), (1, max_len))]

        Returns:
            (tuple): batch of decoder and attention states
                (([L x (B, dec_dim)], [L x (B, dec_dim)]), (B, max_len))

        """
        for layer in range(self.dlayers):
            batch_states[0][0][layer] = torch.stack([s[0][0][layer] for s in l_states])
            batch_states[0][1][layer] = torch.stack([s[0][1][layer] for s in l_states])

        att_states = (
            torch.stack([s[1] for s in l_states])
            if l_states[0][1] is not None
            else None
        )

        return (batch_states[0], att_states)

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

        att_ws = []
        self.att[0].reset()

        eys = self.embed(ys_in_pad)
        state, att_w = self.init_state(eys)

        for i in range(olength):
            att_c, att_w = self.att[0](
                hs_pad, hlens, self.dropout_dec[0](state[0][0]), att_w
            )
            ey = torch.cat((eys[:, i, :], att_c), dim=1)
            _, state = self.rnn_forward(ey, state)

            att_ws.append(att_w)

        att_ws = att_to_numpy(att_ws, self.att[0])

        return att_ws
