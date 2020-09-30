"""Transducer decoder implementation."""

from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.beam_search_transducer import Hypothesis
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.transducer.joint_network import JointNetwork
from espnet2.asr.decoder.rnn_decoder import build_attention_list
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.utils.get_default_kwargs import get_default_kwargs


class TransducerDecoder(RNNDecoder):
    """Transducer Decoder module.

    Args:
        vocab_size: Vocabulary size
        encoder_output_size: Dimension of encoder outputs
        blank: Blank symbol ID
        rnn_type: Type of decoder layers
        num_layers: Number of decoder layers
        hidden_size: Dimension of hidden layers
        embedding_size: Dimension of embedding layer
        dropout: Dropout rate of hidden layers
        joint_space_size: Dimension of joint space
        joint_activation_type: Activation type for joint network
        use_attention: Whether to use attention module
        att_conf: attention module configuration

    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        blank: int = 0,
        rnn_type: str = "lstm",
        num_layers: int = 1,
        hidden_size: int = 320,
        dropout: float = 0.0,
        joint_space_size: int = 320,
        joint_activation_type: str = "tanh",
        use_attention: bool = False,
        att_conf: dict = get_default_kwargs(build_attention_list),
    ):
        """Transducer initializer."""
        assert check_argument_types()

        super().__init__(
            vocab_size,
            encoder_output_size,
            rnn_type=rnn_type,
            num_layers=num_layers,
            hidden_size=hidden_size,
            embed_pad=blank,
            dropout=dropout,
            att_conf=att_conf,
            is_rnnt=not use_attention,
        )

        self.joint_network = JointNetwork(
            vocab_size,
            encoder_output_size,
            hidden_size,
            joint_space_size,
            joint_activation_type,
        )

        self.ignore_id = -1
        self.blank = blank

        self.use_attention = use_attention

    def init_state(self, init_tensor: torch.Tensor) -> torch.Tensor:
        """Initialize decoder states.

        Args:
            init_tensor: Batch of input features (B, D_emb / D_dec)

        Returns:
            (): Batch of decoder states ([L x (B, dec_dim)], [L x (B, D_dec)])

        """
        z_list = [
            to_device(init_tensor, torch.zeros(init_tensor.size(0), self.dunits))
            for _ in range(self.dlayers)
        ]
        c_list = [
            to_device(init_tensor, torch.zeros(init_tensor.size(0), self.dunits))
            for _ in range(self.dlayers)
        ]

        if self.use_attention:
            return ((z_list, c_list), None)
        else:
            return (z_list, c_list)

    def rnn_forward(
        self, ey: torch.Tensor, state: Tuple[List, List]
    ) -> Union[torch.Tensor, Tuple[List, List]]:
        """RNN forward.

        Args:
            ey: batch of input features (B, D_emb)
            state: batch of decoder states ([L x (B, D_dec)], [L x (B, D_dec)])

        Returns:
            y: batch of output features (B, D_dec)
            (): batch of decoder states ([L x (B, D_dec)], [L x (B, D_dec)])

        """
        z_prev, c_prev = state

        if self.use_attention:
            (z_list, c_list), _ = self.init_state(ey)
        else:
            z_list, c_list = self.init_state(ey)

        z_list, c_list = RNNDecoder.rnn_forward(
            self, ey, z_list, c_list, z_prev, c_prev
        )

        y = self.dropout_dec[-1](z_list[-1])

        return y, (z_list, c_list)

    def forward(self, hs_pad, ys_in_pad, hlens=None):
        """Forward function for transducer.

        Args:
            hs_pad: Batch of padded hidden state sequences (B, T_max, D_enc)
            ys_in_pad: Batch of padded character id sequence tensor (B, (L_max+1))

        Returns:
            z: Output (B, T, U, vocab_size)

        """
        olength = ys_in_pad.size(1)

        eys = self.dropout_emb(self.embed(ys_in_pad))

        if self.use_attention:
            self.att_list[0].reset()

            hlens = list(map(int, hlens))
            state, att_w = self.init_state(hs_pad)

            z_all = []
            for i in range(olength):
                att_c, att_w = self.att_list[0](
                    hs_pad, hlens, self.dropout_dec[0](state[0][0]), att_w
                )

                ey = torch.cat((eys[:, i, :], att_c), dim=1)

                y, state = self.rnn_forward(ey, state)
                z_all.append(y)
        else:
            state = self.init_state(hs_pad)

            z_all = []
            for i in range(olength):
                y, state = self.rnn_forward(eys[:, i, :], state)
                z_all.append(y)

        h_dec = torch.stack(z_all, dim=1)

        z = self.joint_network(hs_pad.unsqueeze(2), h_dec.unsqueeze(1))

        return z

    def score(
        self, hyp: Hypothesis, cache: Dict, init_tensor: torch.Tensor = None
    ) -> Union[torch.Tensor, Tuple[List, List]]:
        """Forward one step.

        Args:
            hyp: Hypothesis
            cache: States cache

        Returns:
            y: Decoder outputs (1, D_dec)
            state: Decoder states ([L x (1, dec_dim)], [L x (1, dec_dim)])
            (): Token id for LM (1,)

        """
        vy = to_device(self, torch.full((1, 1), hyp.yseq[-1], dtype=torch.long))

        str_yseq = "".join([str(x) for x in hyp.yseq])

        if str_yseq in cache:
            y, state = cache[str_yseq]
        else:
            ey = self.embed(vy)

            if self.use_attention:
                att_c, att_w = self.att_list[0](
                    init_tensor,
                    [init_tensor.size(1)],
                    hyp.dec_state[0][0][0],
                    hyp.dec_state[1],
                )

                ey = torch.cat((ey[0], att_c), dim=1)

                y, dec_state = self.rnn_forward(ey, hyp.dec_state[0])
                state = (dec_state, att_w)
            else:
                y, state = self.rnn_forward(ey[0], hyp.dec_state)

            cache[str_yseq] = (y, state)

        return y, state, vy[0]

    def batch_score(
        self,
        hyps: List,
        batch_states: Tuple[List, List],
        cache: Dict,
        init_tensor: torch.Tensor = None,
    ) -> Union[torch.Tensor, Tuple[List, List]]:
        """Forward batch one step.

        Args:
            hyps: Batch of hypotheses
            batch_states: Batch of decoder states
                ([L x (B, D_dec)], [L x (B, D_dec)])
            cache: States cache

        Returns:
            batch_y: Decoder output (B, D_dec)
            batch_states: Batch of decoder states
                ([L x (B, D_dec)], [L x (B, D_dec)])
            lm_tokens: Batch of token ids for LM (B)

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

            if self.use_attention:
                state = self.init_state(init_tensor)
                dec_state = self.create_batch_states(state, [p[1] for p in process])

                ey = self.embed(tokens)

                enc_hs = init_tensor.expand(batch, -1, -1)
                enc_len = [init_tensor.squeeze(0).size(0)] * batch

                att_c, att_w = self.att_list[0](
                    enc_hs, enc_len, state[0][0][0], state[1]
                )

                ey = torch.cat((ey, att_c), dim=1)

                y, dec_state = self.rnn_forward(ey, state[0])
            else:
                dec_state = self.init_state(torch.zeros((batch, self.dunits)))
                dec_state = self.create_batch_states(dec_state, [p[1] for p in process])

                ey = self.embed(tokens)

                y, dec_state = self.rnn_forward(ey, dec_state)

        j = 0
        for i in range(final_batch):
            if done[i] is None:
                if self.use_attention:
                    new_state = self.select_state((dec_state, att_w), j)
                else:
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

    def select_state(
        self, batch_states: Tuple[List, List], idx: int
    ) -> Tuple[List, List]:
        """Get decoder state from batch of states, for given id.

        Args:
            batch_states: Batch of decoder states
                ([L x (B, D_dec)], [L x (B, D_dec)])
            idx: Index to extract state from batch of states

        Returns:
            (): Decoder states for given id
                ([L x (1, D_dec)], [L x (1, D_dec)])

        """
        if self.use_attention:
            z_list = [batch_states[0][0][layer][idx] for layer in range(self.dlayers)]
            c_list = [batch_states[0][1][layer][idx] for layer in range(self.dlayers)]

            att_state = (
                batch_states[1][idx] if batch_states[1] is not None else batch_states[1]
            )
            return ((z_list, c_list), att_state)
        else:
            z_list = [batch_states[0][layer][idx] for layer in range(self.dlayers)]
            c_list = [batch_states[1][layer][idx] for layer in range(self.dlayers)]

            return (z_list, c_list)

    def create_batch_states(
        self,
        batch_states: Tuple[List, List],
        l_states: List,
        l_tokens: torch.Tensor = None,
    ) -> Tuple[List, List]:
        """Create batch of decoder states.

        Args:
            batch_states: Batch of decoder states
               ([L x (B, D_dec)], [L x (B, D_dec)])
            l_states: List of decoder states
                [B x ([L x (1, D_dec)], [L x (1, D_dec)])]

        Returns:
            batch_states: Batch of decoder states
                ([L x (B, D_dec)], [L x (B, D_dec)])

        """
        if self.use_attention:
            for layer in range(self.dlayers):
                batch_states[0][0][layer] = torch.stack(
                    [s[0][0][layer] for s in l_states]
                )
                batch_states[0][1][layer] = torch.stack(
                    [s[0][1][layer] for s in l_states]
                )

            att_states = (
                torch.stack([s[1] for s in l_states])
                if l_states[0][1] is not None
                else None
            )

            return (batch_states[0], att_states)
        else:
            for layer in range(self.dlayers):
                batch_states[0][layer] = torch.stack([s[0][layer] for s in l_states])
                batch_states[1][layer] = torch.stack([s[1][layer] for s in l_states])

            return batch_states
