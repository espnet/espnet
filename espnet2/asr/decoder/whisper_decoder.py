import copy
from typing import Any, List, Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.scorer_interface import BatchScorerInterface


class ExpandedTokenEmbedding(torch.nn.Module):
    """
        A custom embedding layer that expands an existing embedding with additional tokens.

    This class extends the functionality of a given embedding layer by adding
    more token embeddings while preserving the original embeddings. The new
    embeddings are initialized with normal distribution based on the statistics
    of the original embedding weights.

    Attributes:
        ori_emb (torch.nn.Embedding): The original embedding layer.
        add_emb (torch.nn.Embedding): The additional embedding layer for new tokens.
        num_embeddings (int): Total number of embeddings (original + additional).

    Args:
        ori_emebedding (torch.nn.Embedding): The original embedding layer to be expanded.
        additional_size (int): Number of additional token embeddings to add.

    Note:
        The forward method is overridden to use the combined weights of both
        original and additional embeddings.

    Example:
        >>> original_embedding = torch.nn.Embedding(1000, 300)
        >>> expanded_embedding = ExpandedTokenEmbedding(original_embedding, 500)
        >>> input_tensor = torch.LongTensor([0, 1500, 999])
        >>> output = expanded_embedding(input_tensor)
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
                Combined weight tensor of the original and additional embeddings.

        Returns:
            torch.Tensor: A tensor containing the concatenated weights of the original
            embedding (ori_emb) and the additional embedding (add_emb) along dimension 0.

        Note:
            This property is used to provide a unified view of the entire embedding
            weight, including both original and additional token embeddings.
        """
        return torch.cat([self.ori_emb.weight, self.add_emb.weight], dim=0)

    def forward(self, input):
        """
                Performs a forward pass through the expanded embedding layer.

        This method applies the embedding operation using the combined weights
        of the original and additional embeddings. It preserves the properties
        of the original embedding layer, such as padding_idx, max_norm, etc.

        Args:
            input (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: The embedded representation of the input tokens.

        Note:
            This method overrides the default forward pass of torch.nn.Embedding
            to use the combined weights while maintaining other embedding properties.

        Example:
            >>> expanded_embedding = ExpandedTokenEmbedding(original_embedding, 500)
            >>> input_tensor = torch.LongTensor([0, 1500, 999])
            >>> output = expanded_embedding(input_tensor)
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
        A decoder class based on OpenAI's Whisper model for speech-to-text tasks.

    This class implements a Transformer-based decoder that utilizes the architecture
    from OpenAI's Whisper model. It can be used for various speech recognition and
    transcription tasks.

    Attributes:
        decoders (whisper.model.Decoder): The Whisper model's decoder.
        dropout (torch.nn.Dropout): Dropout layer for regularization.
        load_origin_token_embedding (bool): Flag to load original token embeddings.

    Args:
        vocab_size (int): Size of the vocabulary.
        encoder_output_size (int): Size of the encoder's output.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.0.
        whisper_model (str, optional): Whisper model size. Defaults to "small".
        download_dir (str, optional): Directory to download the Whisper model.
        load_origin_token_embedding (bool, optional): Whether to load original
            token embeddings when expanding vocabulary. Defaults to False.

    Raises:
        Exception: If the Whisper package is not properly installed.

    Note:
        This class inherits from AbsDecoder and BatchScorerInterface, providing
        compatibility with the ESPnet2 framework.

    Example:
        >>> decoder = OpenAIWhisperDecoder(vocab_size=10000, encoder_output_size=512)
        >>> encoder_output = torch.randn(1, 100, 512)
        >>> decoder_input = torch.LongTensor([[1, 2, 3, 4, 5]])
        >>> decoder_output, _ = decoder(encoder_output, torch.tensor([100]),
        ...                             decoder_input, torch.tensor([5]))
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
                Forward pass of the OpenAI Whisper decoder.

        This method processes the encoder output and the decoder input to generate
        the output token scores.

        Args:
            hs_pad (torch.Tensor): Encoded memory, float32 (batch, maxlen_in, feat).
            hlens (torch.Tensor): Lengths of encoded sequences (batch,).
            ys_in_pad (torch.Tensor): Input token ids, int64 (batch, maxlen_out).
            ys_in_lens (torch.Tensor): Lengths of input sequences (batch,).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x (torch.Tensor): Decoded token scores before softmax
                  (batch, maxlen_out, token).
                - ys_in_lens (torch.Tensor): Lengths of input sequences (batch,).

        Note:
            This method applies positional embedding, processes the input through
            the decoder blocks, and generates the final output scores.

        Example:
            >>> decoder = OpenAIWhisperDecoder(vocab_size=10000, encoder_output_size=512)
            >>> hs_pad = torch.randn(2, 100, 512)
            >>> hlens = torch.tensor([100, 80])
            >>> ys_in_pad = torch.randint(0, 10000, (2, 20))
            >>> ys_in_lens = torch.tensor([20, 15])
            >>> output, out_lens = decoder(hs_pad, hlens, ys_in_pad, ys_in_lens)
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
                Perform a single forward step in the decoder.

        This method processes one step of decoding, typically used in inference
        or beam search scenarios.

        Args:
            tgt (torch.Tensor): Input token ids, int64 (batch, maxlen_out).
            tgt_mask (torch.Tensor): Input token mask, (batch, maxlen_out).
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)
            memory (torch.Tensor): Encoded memory, float32 (batch, maxlen_in, feat).
            cache (List[torch.Tensor], optional): Cached output list of
                (batch, max_time_out-1, size). Defaults to None.

        Returns:
            tuple[torch.Tensor, None]: A tuple containing:
                - y (torch.Tensor): Log probabilities of next tokens (batch, vocab_size).
                - None: Placeholder for cache (currently not implemented).

        Note:
            - The cache implementation is currently ignored for simplicity and correctness.
            - This method applies positional embedding, processes through decoder blocks,
              and generates log probabilities for the next tokens.

        Example:
            >>> decoder = OpenAIWhisperDecoder(vocab_size=10000, encoder_output_size=512)
            >>> tgt = torch.LongTensor([[1, 2, 3]])
            >>> tgt_mask = torch.ones(1, 3, dtype=torch.bool)
            >>> memory = torch.randn(1, 100, 512)
            >>> output, _ = decoder.forward_one_step(tgt, tgt_mask, memory)
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
                Calculate the score for the next token.

        This method computes the log probability scores for the next token given
        the current state and encoder output.

        Args:
            ys (torch.Tensor): Current token sequence.
            state (Any): Current decoder state (unused in this implementation).
            x (torch.Tensor): Encoder output.

        Returns:
            tuple[torch.Tensor, None]: A tuple containing:
                - logp (torch.Tensor): Log probability scores for the next token.
                - None: Updated state (currently not implemented).

        Note:
            This method is typically used in beam search or other decoding algorithms
            to score possible next tokens.

        Example:
            >>> decoder = OpenAIWhisperDecoder(vocab_size=10000, encoder_output_size=512)
            >>> ys = torch.LongTensor([1, 2, 3])
            >>> x = torch.randn(100, 512)
            >>> logp, _ = decoder.score(ys, None, x)
        """
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), torch.empty(0), x.unsqueeze(0), cache=state  # dummy mask
        )
        return logp.squeeze(0), state

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """
                Score new token batch.

        This method computes scores for the next tokens given batched inputs of
        current token sequences and encoder features.

        Args:
            ys (torch.Tensor): Prefix tokens, torch.int64 (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens (unused in this implementation).
            xs (torch.Tensor): Encoder features that generate ys, (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, None]: A tuple containing:
                - logp (torch.Tensor): Batchified scores for next tokens, shape (n_batch, n_vocab).
                - None: Placeholder for next state list (currently not implemented).

        Note:
            This method is designed for batch processing, which can be more efficient
            than scoring individual sequences separately.

        Example:
            >>> decoder = OpenAIWhisperDecoder(vocab_size=10000, encoder_output_size=512)
            >>> ys = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
            >>> xs = torch.randn(2, 100, 512)
            >>> logp, _ = decoder.batch_score(ys, None, xs)
        """
        # batch decoding, dummy mask is passed
        logp, states = self.forward_one_step(ys, torch.empty(0), xs, cache=None)

        return logp, None
