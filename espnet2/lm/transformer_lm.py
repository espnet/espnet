from typing import Any, List, Tuple

import torch
import torch.nn as nn

from espnet2.lm.abs_model import AbsLM
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask


class TransformerLM(AbsLM):
    """
        Transformer Language Model (TransformerLM) for generating and scoring tokens
    based on input sequences. This model is built upon the Transformer architecture
    and extends the abstract language model class.

    Attributes:
        embed (nn.Embedding): Embedding layer for input tokens.
        encoder (Encoder): Transformer encoder that processes the input embeddings.
        decoder (nn.Linear): Linear layer that maps encoder outputs to token scores.

    Args:
        vocab_size (int): Size of the vocabulary.
        pos_enc (str, optional): Type of positional encoding ('sinusoidal' or None).
        embed_unit (int, optional): Dimensionality of the embeddings. Default is 128.
        att_unit (int, optional): Dimensionality of the attention mechanism. Default is 256.
        head (int, optional): Number of attention heads. Default is 2.
        unit (int, optional): Dimensionality of the linear layers. Default is 1024.
        layer (int, optional): Number of layers in the encoder. Default is 4.
        dropout_rate (float, optional): Dropout rate for layers. Default is 0.1.
        positional_dropout_rate (float, optional): Dropout rate for positional encodings.
            Default is 0.1.
        attention_dropout_rate (float, optional): Dropout rate for attention weights.
            Default is 0.1.

    Raises:
        ValueError: If an unknown positional encoding option is provided.

    Examples:
        # Create a TransformerLM model
        model = TransformerLM(vocab_size=10000, pos_enc='sinusoidal')

        # Forward pass
        input_ids = torch.randint(0, 10000, (32, 20))  # Batch of 32 sequences
        output, _ = model(input_ids, None)

        # Scoring a new token
        new_token = torch.tensor([5])  # Example token
        state = None  # Initial state
        scores, new_state = model.score(new_token, state, output)

        # Batch scoring
        batch_tokens = torch.randint(0, 10000, (16, 10))  # Batch of 16 sequences
        states = [None] * 16  # Initial states for each sequence
        batch_scores, new_states = model.batch_score(batch_tokens, states, output)

    Note:
        This model assumes input sequences are padded with a token index of 0.

    Todo:
        - Add support for more advanced positional encoding options.
    """

    def __init__(
        self,
        vocab_size: int,
        pos_enc: str = None,
        embed_unit: int = 128,
        att_unit: int = 256,
        head: int = 2,
        unit: int = 1024,
        layer: int = 4,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
    ):
        super().__init__()
        if pos_enc == "sinusoidal":
            pos_enc_class = PositionalEncoding
        elif pos_enc is None:

            def pos_enc_class(*args, **kwargs):
                return nn.Sequential()  # indentity

        else:
            raise ValueError(f"unknown pos-enc option: {pos_enc}")

        self.embed = nn.Embedding(vocab_size, embed_unit)
        self.encoder = Encoder(
            idim=embed_unit,
            attention_dim=att_unit,
            attention_heads=head,
            linear_units=unit,
            num_blocks=layer,
            dropout_rate=dropout_rate,
            input_layer="linear",
            pos_enc_class=pos_enc_class,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
        )
        self.decoder = nn.Linear(att_unit, vocab_size)

    def _target_mask(self, ys_in_pad):
        ys_mask = ys_in_pad != 0
        m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
        return ys_mask.unsqueeze(-2) & m

    def forward(self, input: torch.Tensor, hidden: None) -> Tuple[torch.Tensor, None]:
        """
        Compute LM loss value from buffer sequences.

        This method processes input tensor through embedding, encoder, and
        decoder layers to produce the output tensor representing the
        predicted next tokens.

        Args:
            input (torch.Tensor): Input ids. Shape: (batch, len).
            hidden (None): This argument is not used in the current implementation.

        Returns:
            Tuple[torch.Tensor, None]: A tuple containing:
                - Output tensor of shape (batch, len, vocab_size) representing
                  the predicted token probabilities for each input token.
                - None, as the hidden state is not used.

        Examples:
            >>> model = TransformerLM(vocab_size=1000)
            >>> input_tensor = torch.randint(0, 1000, (32, 10))  # (batch, len)
            >>> output, _ = model.forward(input_tensor, None)
            >>> print(output.shape)  # Should output: torch.Size([32, 10, 1000])

        Note:
            The `hidden` argument is maintained for compatibility with
            other models but is not utilized in this method.
        """
        x = self.embed(input)
        mask = self._target_mask(input)
        h, _ = self.encoder(x, mask)
        y = self.decoder(h)
        return y, None

    def score(
        self, y: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """
            Score new token.

        This method computes the score for a new token based on the provided
        prefix tokens and the current state of the model. It takes the prefix
        tokens and generates a score for the next possible token in the
        vocabulary.

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens.
            x (torch.Tensor): Encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: A tuple containing:
                - torch.float32 scores for the next token (shape: vocab_size).
                - Next state for ys.

        Examples:
            >>> model = TransformerLM(vocab_size=1000)
            >>> prefix_tokens = torch.tensor([1, 2, 3])
            >>> state = None  # or some valid state
            >>> encoder_features = torch.randn(1, 10, 256)  # Example features
            >>> scores, next_state = model.score(prefix_tokens, state, encoder_features)
            >>> print(scores.shape)  # Output: torch.Size([1000])
        """
        y = y.unsqueeze(0)
        h, _, cache = self.encoder.forward_one_step(
            self.embed(y), self._target_mask(y), cache=state
        )
        h = self.decoder(h[:, -1])
        logp = h.log_softmax(dim=-1).squeeze(0)
        return logp, cache

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """
            Score new token batch.

        This method computes the scores for a batch of prefix tokens by processing
        the input sequences through the transformer model. It merges the states of
        the prefix tokens and utilizes the encoder to produce the output scores.

        Attributes:
            None

        Args:
            ys (torch.Tensor):
                A tensor of shape (n_batch, ylen) containing the prefix tokens,
                represented as torch.int64.
            states (List[Any]):
                A list of scorer states corresponding to each prefix token in the batch.
            xs (torch.Tensor):
                A tensor of shape (n_batch, xlen, n_feat) representing the encoder
                features that generate the scores for the prefix tokens.

        Returns:
            tuple[torch.Tensor, List[Any]]:
                A tuple containing:
                    - A tensor of shape (n_batch, vocab_size) with the batchified scores
                      for the next token.
                    - A list of next state lists for each prefix token in the batch.

        Examples:
            >>> model = TransformerLM(vocab_size=1000)
            >>> ys = torch.randint(0, 1000, (32, 10))  # Example prefix tokens
            >>> states = [None] * 32  # Initial states for the batch
            >>> xs = torch.randn(32, 15, 256)  # Example encoder features
            >>> scores, next_states = model.batch_score(ys, states, xs)

        Note:
            Ensure that the shapes of the input tensors are consistent with the
            model's architecture. The input tensors should be appropriately padded
            to match the expected dimensions.
        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.encoder.encoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        h, _, states = self.encoder.forward_one_step(
            self.embed(ys), self._target_mask(ys), cache=batch_state
        )
        h = self.decoder(h[:, -1])
        logp = h.log_softmax(dim=-1)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list
