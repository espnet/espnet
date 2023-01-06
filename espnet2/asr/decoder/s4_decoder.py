"""Decoder definition."""
from typing import Any, List, Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.state_spaces.model import SequenceModel
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.scorer_interface import BatchScorerInterface


class S4Decoder(AbsDecoder, BatchScorerInterface):
    """S4 decoder module.

    Args:
        vocab_size: output dim
        encoder_output_size: dimension of hidden vector
        input_layer: input layer type
        dropinp: input dropout
        dropout: dropout parameter applied on every residual and every layer
        prenorm: pre-norm vs. post-norm
        n_layers: number of layers
        transposed: transpose inputs so each layer receives (batch, dim, length)
        tie_dropout: tie dropout mask across sequence like nn.Dropout1d/nn.Dropout2d
        n_repeat: each layer is repeated n times per stage before applying pooling
        layer: layer config, must be specified
        residual: residual config
        norm: normalization config (e.g. layer vs batch)
        pool: config for pooling layer per stage
        track_norms: log norms of each layer output
        drop_path: drop rate for stochastic depth
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        input_layer: str = "embed",
        dropinp: float = 0.0,
        dropout: float = 0.25,
        prenorm: bool = True,
        n_layers: int = 16,
        transposed: bool = False,
        tie_dropout: bool = False,
        n_repeat=1,
        layer=None,
        residual=None,
        norm=None,
        pool=None,
        track_norms=True,
        drop_path: float = 0.0,
    ):
        assert check_argument_types()
        super().__init__()

        self.d_model = encoder_output_size
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.odim = vocab_size
        self.dropout = dropout

        if input_layer == "embed":
            self.embed = torch.nn.Embedding(vocab_size, self.d_model)
        else:
            raise NotImplementedError
        self.dropout_emb = torch.nn.Dropout(p=dropout)

        self.decoder = SequenceModel(
            self.d_model,
            n_layers=n_layers,
            transposed=transposed,
            dropout=dropout,
            tie_dropout=tie_dropout,
            prenorm=prenorm,
            n_repeat=n_repeat,
            layer=layer,
            residual=residual,
            norm=norm,
            pool=pool,
            track_norms=track_norms,
            dropinp=dropinp,
            drop_path=drop_path,
        )

        self.output = torch.nn.Linear(self.d_model, vocab_size)

    def init_state(self, x: torch.Tensor):
        """Initialize state"""
        return self.decoder.default_state(1, device=x.device)

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        state=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """

        memory = hs_pad
        memory_mask = (~make_pad_mask(hlens, maxlen=memory.size(1)))[:, None, :].to(
            memory.device
        )

        emb = self.embed(ys_in_pad)
        z, state = self.decoder(
            emb,
            state=state,
            memory=memory,
            lengths=ys_in_lens,
            mask=memory_mask,
        )

        decoded = self.output(z)
        return decoded, ys_in_lens

    def score(self, ys, state, x):
        raise NotImplementedError

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        ys = self.embed(ys[:, -1:])

        # workaround for remaining beam width of 1
        if type(states[0]) is list:
            states = states[0]

        assert ys.size(1) == 1, ys.shape
        ys = ys.squeeze(1)

        ys, states = self.decoder.step(ys, state=states, memory=xs)
        logp = self.output(ys).log_softmax(dim=-1)

        states_list = [
            [state[b].unsqueeze(0) if state is not None else None for state in states]
            for b in range(n_batch)
        ]

        return logp, states_list
