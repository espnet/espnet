"""RNN-Transducer implementation for training and decoding."""

import torch

from typeguard import check_argument_types
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

from espnet.nets.beam_search_transducer_espnet2 import Hypothesis
from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.nets_utils import to_device

from espnet2.asr.decoder.abs_decoder import AbsDecoder


class RNNTDecoder(AbsDecoder):
    """RNN-T Decoder module.

    Args:
        vocab_size: Vocabulary size
        encoder_output_size: Dimension of encoder outputs
        blank: Blank symbol ID
        rnn_type: Type of decoder layers
        num_layers: Number of decoder layers
        hidden_size: Dimension of hidden layers
        embedding_size: Dimension of embedding layer
        dropout: Dropout rate of hidden layers
        dropout_embedding: Dropout rate of embedding layer
        joint_space_size: Dimension of joint space
        joint_activation_type: Activation type for joint network

    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        blank: int = 0,
        rnn_type: str = "lstm",
        num_layers: int = 1,
        hidden_size: int = 320,
        embedding_size: int = 320,
        dropout: float = 0.0,
        dropout_embedding: float = 0.0,
        joint_space_size: int = 320,
        joint_activation_type: str = "tanh",
    ):
        """Transducer initializer."""
        assert check_argument_types()

        if rnn_type not in {"lstm", "gru"}:
            raise ValueError(f"Not supported: rnn_type={rnn_type}")

        super().__init__()

        self.embed = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=blank)
        self.dropout_embed = torch.nn.Dropout(p=dropout_embedding)

        if rnn_type == "lstm":
            dec_net = torch.nn.LSTMCell
        else:
            dec_net = torch.nn.GRUCell

        self.decoder = torch.nn.ModuleList([dec_net(embedding_size, hidden_size)])
        self.dropout_dec = torch.nn.ModuleList([torch.nn.Dropout(p=dropout)])

        for _ in range(1, num_layers):
            self.decoder += [dec_net(hidden_size, hidden_size)]
            self.dropout_dec += [torch.nn.Dropout(p=dropout)]

        self.lin_enc = torch.nn.Linear(encoder_output_size, joint_space_size)
        self.lin_dec = torch.nn.Linear(hidden_size, joint_space_size, bias=False)
        self.lin_out = torch.nn.Linear(joint_space_size, vocab_size)

        self.joint_activation = get_activation(joint_activation_type)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.embedding_size = embedding_size
        self.joint_space_size = joint_space_size
        self.vocab_size = vocab_size

        self.ignore_id = -1
        self.blank = blank

    def init_state(self, init_tensor: torch.Tensor) -> torch.Tensor:
        """Initialize decoder states.

        Args:
            init_tensor: Batch of input features (B, D_emb / D_dec)

        Returns:
            (): Batch of decoder states ([L x (B, dec_dim)], [L x (B, D_dec)])

        """
        z_list = [
            to_device(init_tensor, torch.zeros(init_tensor.size(0), self.hidden_size))
            for _ in range(self.num_layers)
        ]
        c_list = [
            to_device(init_tensor, torch.zeros(init_tensor.size(0), self.hidden_size))
            for _ in range(self.num_layers)
        ]

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
        z_list, c_list = self.init_state(ey)

        if self.rnn_type == "lstm":
            z_list[0], c_list[0] = self.decoder[0](ey, (z_prev[0], c_prev[0]))

            for i in range(1, self.num_layers):
                z_list[i], c_list[i] = self.decoder[i](
                    self.dropout_dec[i - 1](z_list[i - 1]), (z_prev[i], c_prev[i])
                )
        else:
            z_list[0] = self.decoder[0](ey, z_prev[0])

            for i in range(1, self.num_layers):
                z_list[i] = self.decoder[i](
                    self.dropout_dec[i - 1](z_list[i - 1]), z_prev[i]
                )
        y = self.dropout_dec[-1](z_list[-1])

        return y, (z_list, c_list)

    def joint(self, h_enc: torch.Tensor, h_dec: torch.Tensor) -> torch.Tensor:
        """Joint computation of z.

        Args:
            h_enc: Batch of expanded hidden state (B, T, 1, D_enc)
            h_dec: Batch of expanded hidden state (B, 1, U, D_dec)

        Returns:
            z: Output (B, T, U, vocab_size)

        """
        z = self.joint_activation(self.lin_enc(h_enc) + self.lin_dec(h_dec))
        z = self.lin_out(z)

        return z

    def forward(self, hs_pad, ys_in_pad, hlens=None):
        """Forward function for transducer.

        Args:
            hs_pad: Batch of padded hidden state sequences (B, T_max, D_enc)
            ys_in_pad: Batch of padded character id sequence tensor (B, (L_max+1))

        Returns:
            z: Output (B, T, U, vocab_size)

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

        z = self.joint(h_enc, h_dec)

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

            dec_state = self.init_state(torch.zeros((batch, self.hidden_size)))
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
        z_list = [batch_states[0][layer][idx] for layer in range(self.num_layers)]
        c_list = [batch_states[1][layer][idx] for layer in range(self.num_layers)]

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
        for layer in range(self.num_layers):
            batch_states[0][layer] = torch.stack([s[0][layer] for s in l_states])
            batch_states[1][layer] = torch.stack([s[1][layer] for s in l_states])

        return batch_states
