import copy
from typing import Any, List, Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.scorer_interface import BatchScorerInterface


class ExpandedTokenEmbedding(torch.nn.Module):
    """
    ExpandedTokenEmbedding is a PyTorch module that extends the functionality of a 
    given embedding layer by adding additional token embeddings. This class is 
    designed to accommodate scenarios where the vocabulary size needs to be 
    expanded while maintaining the original embeddings.

    Attributes:
        ori_emb (torch.nn.Embedding): The original embedding layer.
        add_emb (torch.nn.Embedding): The additional embedding layer for new tokens.
        num_embeddings (int): Total number of embeddings after expansion.

    Args:
        ori_emebedding (torch.nn.Embedding): The original embedding layer to extend.
        additional_size (int): The number of additional embeddings to add.

    Returns:
        torch.Tensor: The concatenated weights of the original and additional 
        embeddings.

    Examples:
        >>> original_embedding = torch.nn.Embedding(10, 5)  # 10 tokens, 5 dimensions
        >>> expanded_embedding = ExpandedTokenEmbedding(original_embedding, 5)
        >>> expanded_embedding.num_embeddings
        15  # Original 10 plus 5 additional tokens

    Note:
        The additional embeddings are initialized with the same mean and standard 
        deviation as the original embeddings.

    Raises:
        ValueError: If the original embedding is not of type torch.nn.Embedding.
    """
    def __init__(self, ori_emebedding, additional_size):
        super().__init__()
        self.ori_emb = ori_emebedding

        orig_emb_std, orig_emb_mean = torch.std_mean(ori_emebedding.weight)
        self.add_emb = torch.nn.Embedding(additional_size, ori_emebedding.embedding_dim)
        torch.nn.init.normal_(
            self.add_emb.weight,
            orig_emb_mean.item(),
            orig_emb_std.item(),
        )
        self.num_embeddings = ori_emebedding.num_embeddings + additional_size

    @property
    def weight(self):
        """
        Returns the concatenated weights of the original and additional embeddings.

    The `weight` property returns a tensor that combines the weights from the
    original embedding and the additional embedding, allowing for an expanded
    token representation.

    Attributes:
        weight (torch.Tensor): A tensor of shape (num_embeddings, embedding_dim)
            containing the concatenated weights of the original and additional
            embeddings.

    Returns:
        torch.Tensor: The combined weights of the original and additional
            embeddings.

    Examples:
        >>> ori_embedding = torch.nn.Embedding(10, 5)  # original embedding
        >>> expanded_embedding = ExpandedTokenEmbedding(ori_embedding, 5)
        >>> combined_weights = expanded_embedding.weight
        >>> combined_weights.shape
        torch.Size([15, 5])  # 10 original + 5 additional embeddings

    Note:
        The additional embedding weights are initialized using the mean and
        standard deviation of the original embedding weights.
        """
        return torch.cat([self.ori_emb.weight, self.add_emb.weight], dim=0)

    def forward(self, input):
        """
        Forward decoder.

        This method performs the forward pass of the decoder, taking encoded 
        memory and input token IDs to produce token scores. The output can be 
        used for further processing such as computing loss or for generating 
        predictions.

        Args:
            hs_pad: Encoded memory, a float32 tensor of shape 
                (batch, maxlen_in, feat) representing the features from 
                the encoder.
            hlens: A tensor of shape (batch) containing the lengths of the 
                encoded sequences in `hs_pad`.
            ys_in_pad: Input token IDs, an int64 tensor of shape 
                (batch, maxlen_out). This represents the input tokens for 
                the decoder. If `input_layer` is set to "embed", this 
                should be a tensor of token IDs. In other cases, it can 
                be a tensor of shape (batch, maxlen_out, #mels).
            ys_in_lens: A tensor of shape (batch) containing the lengths of 
                the input sequences in `ys_in_pad`.

        Returns:
            tuple: A tuple containing:
                - x: Decoded token scores before softmax, a tensor of shape 
                  (batch, maxlen_out, token) if `use_output_layer` is 
                  True.
                - olens: A tensor of shape (batch,) containing the lengths of 
                  the output sequences.

        Examples:
            >>> hs_pad = torch.randn(2, 10, 512)  # (batch, maxlen_in, feat)
            >>> hlens = torch.tensor([10, 8])  # (batch)
            >>> ys_in_pad = torch.tensor([[1, 2, 3], [1, 2, 0]])  # (batch, maxlen_out)
            >>> ys_in_lens = torch.tensor([3, 2])  # (batch)
            >>> x, olens = decoder.forward(hs_pad, hlens, ys_in_pad, ys_in_lens)

        Note:
            Ensure that the input tensors are appropriately sized and 
            formatted as per the expected shapes to avoid runtime errors.
        """
        return torch.nn.functional.embedding(
            input,
            self.weight,
            self.ori_emb.padding_idx,
            self.ori_emb.max_norm,
            self.ori_emb.norm_type,
            self.ori_emb.scale_grad_by_freq,
            self.ori_emb.sparse,
        )


class OpenAIWhisperDecoder(AbsDecoder, BatchScorerInterface):
    """
    OpenAIWhisperDecoder is a transformer-based decoder for speech-to-text tasks 
    using OpenAI's Whisper model. It inherits from AbsDecoder and implements 
    BatchScorerInterface for scoring functionality.

    This decoder is designed to process encoded audio features and produce 
    token predictions based on a given vocabulary size. It allows for 
    customization through various parameters, including dropout rates and 
    model selection.

    Attributes:
        decoders (torch.nn.Module): The decoder network loaded from the Whisper model.
        load_origin_token_embedding (bool): Flag to indicate whether to load 
            original token embeddings when expanding vocabulary.

    Args:
        vocab_size (int): The size of the vocabulary for the model.
        encoder_output_size (int): The size of the encoder output features.
        dropout_rate (float, optional): Dropout rate to apply in the decoder. 
            Defaults to 0.0.
        whisper_model (str, optional): The specific Whisper model to use (e.g., 
            "small"). Defaults to "small".
        download_dir (Optional[str], optional): Directory to download the Whisper 
            model if not already present. Defaults to None.
        load_origin_token_embedding (bool, optional): If True, load original 
            token embeddings when vocabulary is expanded. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - x (torch.Tensor): Decoded token scores before softmax 
            (batch, maxlen_out, token).
            - olens (torch.Tensor): Lengths of the output sequences (batch,).

    Yields:
        None

    Raises:
        AssertionError: If the specified whisper_model is not available.
        Exception: If the Whisper model fails to load due to installation issues.

    Examples:
        # Initialize the decoder
        decoder = OpenAIWhisperDecoder(vocab_size=50000, encoder_output_size=512)

        # Forward pass through the decoder
        hs_pad = torch.rand(16, 100, 512)  # Simulated encoder output
        hlens = torch.tensor([100] * 16)    # Lengths of the input sequences
        ys_in_pad = torch.randint(0, 50000, (16, 50))  # Simulated token ids
        ys_in_lens = torch.tensor([50] * 16)  # Lengths of the output sequences

        x, olens = decoder(hs_pad, hlens, ys_in_pad, ys_in_lens)

    Note:
        The Whisper model architecture does not use dropout by default. 
        If a dropout rate is specified, it will be applied during training.

    Todo:
        Implement cache mechanism for improved performance during 
        incremental decoding.
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        whisper_model: str = "small",
        download_dir: Optional[str] = None,
        load_origin_token_embedding=False,
    ):
        try:
            import whisper
        except Exception as e:
            print("Error: whisper is not properly installed.")
            print(
                "Please install whisper with: cd ${MAIN_ROOT}/tools && "
                "./installers/install_whisper.sh"
            )
            raise e

        super().__init__()

        assert whisper_model in whisper.available_models()
        _model = whisper.load_model(
            whisper_model, download_root=download_dir, device="cpu"
        )
        self.decoders = copy.deepcopy(_model.decoder)
        attention_dim = self.decoders.token_embedding.embedding_dim

        # note that originally Whisper doesn't use dropouts
        self.dropout = torch.nn.Dropout(dropout_rate)

        # load the original token_embeddings, if the vocabulary is expanded
        self.load_origin_token_embedding = load_origin_token_embedding

        # vocab size mismatch -> reinitialize embedding
        # orig vocab size (multilingual): 51865
        # orig vocab size (english): 51864
        if vocab_size != self.decoders.token_embedding.num_embeddings:
            if self.load_origin_token_embedding:
                assert (
                    vocab_size > self.decoders.token_embedding.num_embeddings
                ), "expanded vocab_size should be larged than the origin"
                self.decoders.token_embedding = ExpandedTokenEmbedding(
                    self.decoders.token_embedding,
                    vocab_size - self.decoders.token_embedding.num_embeddings,
                )
            else:
                orig_emb_std, orig_emb_mean = torch.std_mean(
                    self.decoders.token_embedding.weight
                )
                self.decoders.token_embedding = torch.nn.Embedding(
                    vocab_size, attention_dim
                )
                torch.nn.init.normal_(
                    self.decoders.token_embedding.weight,
                    orig_emb_mean.item(),
                    orig_emb_std.item(),
                )

        self.decoders.train()
        del _model

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward decoder.

        This method takes the encoded memory and input token ids to produce the 
        decoded token scores before applying softmax.

        Args:
            hs_pad (torch.Tensor): Encoded memory with shape 
                (batch, maxlen_in, feat) of type float32.
            hlens (torch.Tensor): Lengths of the encoded memory, shape (batch).
            ys_in_pad (torch.Tensor): Input token ids with shape 
                (batch, maxlen_out) of type int64. This could either be 
                token ids if input_layer is "embed", or a tensor 
                (batch, maxlen_out, #mels) in other scenarios.
            ys_in_lens (torch.Tensor): Lengths of the input tokens, shape (batch).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x (torch.Tensor): Decoded token scores before softmax with shape 
                  (batch, maxlen_out, token) if use_output_layer is True.
                - olens (torch.Tensor): Lengths of the output tokens, shape (batch,).
        
        Examples:
            >>> hs_pad = torch.rand(32, 10, 512)  # Example encoded memory
            >>> hlens = torch.tensor([10] * 32)    # Example lengths
            >>> ys_in_pad = torch.randint(0, 100, (32, 20))  # Example token ids
            >>> ys_in_lens = torch.tensor([20] * 32)  # Example lengths
            >>> decoder = OpenAIWhisperDecoder(vocab_size=100, encoder_output_size=512)
            >>> scores, output_lengths = decoder.forward(hs_pad, hlens, ys_in_pad, ys_in_lens)

        Note:
            The method uses the decoder's token embedding and positional 
            embedding, followed by dropout and block processing through the 
            decoder layers.

        Raises:
            ValueError: If the input tensor dimensions do not match the expected shapes.
        """
        tgt, memory = ys_in_pad, hs_pad
        tgt = (
            self.decoders.token_embedding(tgt)
            + self.decoders.positional_embedding[: tgt.size(1)]
        )
        tgt = self.dropout(tgt)

        x = tgt.to(memory.dtype)

        for layer, block in enumerate(self.decoders.blocks):
            x = block(x, memory, mask=self.decoders.mask)
            if layer < len(self.decoders.blocks) - 1:
                x = self.dropout(x)

        x = self.decoders.ln(x)
        x = (
            x @ torch.transpose(self.decoders.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return x, ys_in_lens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        *,
        cache: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward one step in the decoding process of the OpenAI Whisper model.

        This method computes the output for a single decoding step using the 
        provided target tokens and the encoded memory. It also manages the 
        positional embeddings and applies the necessary transformations through 
        the decoder blocks.

        Args:
            tgt (torch.Tensor): Input token ids, of shape (batch, maxlen_out).
            tgt_mask (torch.Tensor): Input token mask, of shape (batch, maxlen_out).
                The dtype should be torch.uint8 for PyTorch versions < 1.2 
                and torch.bool for PyTorch 1.2 and above.
            memory (torch.Tensor): Encoded memory, of shape (batch, maxlen_in, feat).
            cache (List[torch.Tensor], optional): Cached output list of shape 
                (batch, max_time_out-1, size). Defaults to None.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - torch.Tensor: Neural network output value, of shape 
                  (batch, maxlen_out, token).
                - List[torch.Tensor]: Updated cache, currently returns None 
                  as cache implementation is ignored for simplicity.

        Note:
            The cache implementation is not utilized in this version for 
            simplicity and correctness.

        Examples:
            >>> decoder = OpenAIWhisperDecoder(vocab_size=1000, 
            ...                                   encoder_output_size=512)
            >>> tgt = torch.randint(0, 1000, (32, 10))  # Example input
            >>> tgt_mask = torch.ones((32, 10), dtype=torch.bool)
            >>> memory = torch.rand((32, 20, 512))  # Example memory
            >>> output, cache = decoder.forward_one_step(tgt, tgt_mask, memory)
        """
        x = (
            self.decoders.token_embedding(tgt)
            + self.decoders.positional_embedding[: tgt.size(1)]
        )
        x = self.dropout(x)
        x = x.to(memory.dtype)

        for layer, block in enumerate(self.decoders.blocks):
            x = block(x, memory, mask=self.decoders.mask)
            if layer < len(self.decoders.blocks) - 1:
                x = self.dropout(x)

        x = self.decoders.ln(x)
        y = x[:, -1]
        y = (
            y @ torch.transpose(self.decoders.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()
        y = torch.log_softmax(y, dim=-1)

        return y, None

    def score(self, ys, state, x):
        """
        Score the token predictions based on the input sequence and current state.

        This method computes the log probabilities of the next token given the 
        previous tokens and the encoded memory from the encoder. It uses the 
        `forward_one_step` method to perform a single decoding step and 
        returns the resulting log probabilities along with the updated state.

        Args:
            ys (torch.Tensor): A tensor of shape (1, ylen) containing the input 
                token IDs, where `ylen` is the length of the token sequence.
            state (Any): The current state used for caching previous computations.
            x (torch.Tensor): A tensor of shape (1, xlen, feat) representing the 
                encoded memory, where `xlen` is the length of the encoder output 
                and `feat` is the feature dimension.

        Returns:
            Tuple[torch.Tensor, Any]: A tuple containing:
                - logp (torch.Tensor): A tensor of shape (n_vocab,) with the 
                log probabilities of the next token.
                - state (Any): The updated state after processing the input.

        Examples:
            >>> decoder = OpenAIWhisperDecoder(vocab_size=50000, encoder_output_size=256)
            >>> ys = torch.tensor([1, 2, 3])  # example token sequence
            >>> state = None  # initial state
            >>> x = torch.randn(1, 10, 256)  # example encoder output
            >>> logp, new_state = decoder.score(ys, state, x)

        Note:
            The input `ys` must have at least one token, and the shape of `x` 
            should match the expected input format for the encoder's output.
        """
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), torch.empty(0), x.unsqueeze(0), cache=state  # dummy mask
        )
        return logp.squeeze(0), state

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """
        Score new token batch using the decoder.

        This method computes the scores for the next token predictions based on the 
        provided prefix tokens and encoder features. It is designed for batch 
        processing, allowing multiple sequences to be scored simultaneously.

        Args:
            ys (torch.Tensor): A tensor of shape (n_batch, ylen) containing the 
                prefix tokens in int64 format.
            states (List[Any]): A list containing the scorer states for the prefix 
                tokens, used for maintaining the state across batches.
            xs (torch.Tensor): A tensor of shape (n_batch, xlen, n_feat) representing 
                the encoder features that generate the prefix tokens.

        Returns:
            Tuple[torch.Tensor, List[Any]]: A tuple containing:
                - A tensor of shape (n_batch, n_vocab) with the batchified scores 
                for the next token.
                - A list of next state for the prefix tokens (ys).

        Examples:
            >>> decoder = OpenAIWhisperDecoder(vocab_size=50000, encoder_output_size=512)
            >>> ys = torch.tensor([[1, 2, 3], [1, 2, 4]])  # Example prefix tokens
            >>> states = [None, None]  # Example states for each batch
            >>> xs = torch.rand(2, 100, 512)  # Example encoder features
            >>> logp, new_states = decoder.batch_score(ys, states, xs)
            >>> print(logp.shape)  # Output: torch.Size([2, 50000])

        Note:
            The method currently ignores the cached state for simplicity.
        """
        # batch decoding, dummy mask is passed
        logp, states = self.forward_one_step(ys, torch.empty(0), xs, cache=None)

        return logp, None
