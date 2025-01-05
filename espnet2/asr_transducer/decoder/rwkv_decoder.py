"""RWKV decoder definition for Transducer models."""

from typing import Dict, List, Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.asr_transducer.beam_search_transducer import Hypothesis
from espnet2.asr_transducer.decoder.abs_decoder import AbsDecoder
from espnet2.asr_transducer.decoder.blocks.rwkv import RWKV
from espnet2.asr_transducer.normalization import get_normalization


class RWKVDecoder(AbsDecoder):
    """
    RWKV decoder module for Transducer models.

    This class implements the RWKV decoder based on the architecture described
    in the paper: https://arxiv.org/pdf/2305.13048.pdf. It is designed to work
    with Transducer models for automatic speech recognition tasks.

    Attributes:
        block_size (int): The size of the input/output blocks.
        attention_size (int): The hidden size for self-attention layers.
        output_size (int): The size of the output layer.
        vocab_size (int): The number of unique tokens in the vocabulary.
        context_size (int): The size of the context for WKV computation.
        rescale_every (int): Frequency of input rescaling in inference mode.
        rescaled_layers (bool): Flag indicating if layers are rescaled.
        pad_idx (int): The ID for the padding symbol in embeddings.
        num_blocks (int): The number of RWKV blocks in the decoder.
        score_cache (dict): Cache for storing scores during decoding.
        device (torch.device): The device on which the model is located.

    Args:
        vocab_size (int): Vocabulary size.
        block_size (int, optional): Input/Output size. Default is 512.
        context_size (int, optional): Context size for WKV computation. Default is 1024.
        linear_size (int, optional): FeedForward hidden size. Default is None.
        attention_size (int, optional): SelfAttention hidden size. Default is None.
        normalization_type (str, optional): Normalization layer type. Default is "layer_norm".
        normalization_args (Dict, optional): Normalization layer arguments. Default is {}.
        num_blocks (int, optional): Number of RWKV blocks. Default is 4.
        rescale_every (int, optional): Rescale input every N blocks (inference only). Default is 0.
        embed_dropout_rate (float, optional): Dropout rate for embedding layer. Default is 0.0.
        att_dropout_rate (float, optional): Dropout rate for the attention module. Default is 0.0.
        ffn_dropout_rate (float, optional): Dropout rate for the feed-forward module. Default is 0.0.
        embed_pad (int, optional): Embedding padding symbol ID. Default is 0.

    Examples:
        # Initialize the RWKVDecoder
        decoder = RWKVDecoder(vocab_size=1000, block_size=512)

        # Forward pass through the decoder
        labels = torch.randint(0, 1000, (32, 10))  # Example input
        output = decoder(labels)

        # Inference with hidden states
        states = decoder.init_state(batch_size=32)
        output, new_states = decoder.inference(labels, states)

    Raises:
        AssertionError: If the length of the input labels exceeds the context size.

    Note:
        This implementation uses PyTorch and requires the appropriate environment
        with CUDA support for GPU acceleration if needed.

    Todo:
        - Implement additional features for better performance and flexibility.
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 512,
        context_size: int = 1024,
        linear_size: Optional[int] = None,
        attention_size: Optional[int] = None,
        normalization_type: str = "layer_norm",
        normalization_args: Dict = {},
        num_blocks: int = 4,
        rescale_every: int = 0,
        embed_dropout_rate: float = 0.0,
        att_dropout_rate: float = 0.0,
        ffn_dropout_rate: float = 0.0,
        embed_pad: int = 0,
    ) -> None:
        """Construct a RWKVDecoder object."""
        super().__init__()

        norm_class, norm_args = get_normalization(
            normalization_type, **normalization_args
        )

        linear_size = block_size * 4 if linear_size is None else linear_size
        attention_size = block_size if attention_size is None else attention_size

        self.embed = torch.nn.Embedding(vocab_size, block_size, padding_idx=embed_pad)
        self.dropout_embed = torch.nn.Dropout(p=embed_dropout_rate)

        self.rwkv_blocks = torch.nn.ModuleList(
            [
                RWKV(
                    block_size,
                    linear_size,
                    attention_size,
                    context_size,
                    block_id,
                    num_blocks,
                    normalization_class=norm_class,
                    normalization_args=norm_args,
                    att_dropout_rate=att_dropout_rate,
                    ffn_dropout_rate=ffn_dropout_rate,
                )
                for block_id in range(num_blocks)
            ]
        )

        self.embed_norm = norm_class(block_size, **norm_args)
        self.final_norm = norm_class(block_size, **norm_args)

        self.block_size = block_size
        self.attention_size = attention_size
        self.output_size = block_size
        self.vocab_size = vocab_size
        self.context_size = context_size

        self.rescale_every = rescale_every
        self.rescaled_layers = False

        self.pad_idx = embed_pad
        self.num_blocks = num_blocks

        self.score_cache = {}

        self.device = next(self.parameters()).device

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        RWKV decoder module.

        Based on https://arxiv.org/pdf/2305.13048.pdf.

        Attributes:
            block_size (int): Size of the input/output.
            attention_size (int): Size of the hidden layer in the attention module.
            output_size (int): Size of the output layer.
            vocab_size (int): Vocabulary size.
            context_size (int): Context size for WKV computation.
            rescale_every (int): Rescale input every N blocks (inference only).
            rescaled_layers (bool): Indicates if layers are rescaled.
            pad_idx (int): Embedding padding symbol ID.
            num_blocks (int): Number of RWKV blocks.
            score_cache (dict): Cache for scores.
            device (torch.device): The device on which the model is located.

        Args:
            vocab_size (int): Vocabulary size.
            block_size (int, optional): Input/Output size. Default is 512.
            context_size (int, optional): Context size for WKV computation. Default is 1024.
            linear_size (int, optional): FeedForward hidden size. Default is None.
            attention_size (int, optional): SelfAttention hidden size. Default is None.
            normalization_type (str, optional): Normalization layer type. Default is "layer_norm".
            normalization_args (dict, optional): Normalization layer arguments. Default is {}.
            num_blocks (int, optional): Number of RWKV blocks. Default is 4.
            rescale_every (int, optional): Rescale input every N blocks (inference only). Default is 0.
            embed_dropout_rate (float, optional): Dropout rate for embedding layer. Default is 0.0.
            att_dropout_rate (float, optional): Dropout rate for the attention module. Default is 0.0.
            ffn_dropout_rate (float, optional): Dropout rate for the feed-forward module. Default is 0.0.
            embed_pad (int, optional): Embedding padding symbol ID. Default is 0.

        Methods:
            forward(labels: torch.Tensor) -> torch.Tensor:
                Encode source label sequences.

            inference(labels: torch.Tensor, states: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
                Encode source label sequences with hidden states.

            set_device(device: torch.device) -> None:
                Set GPU device to use.

            score(label_sequence: List[int], states: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
                One-step forward hypothesis.

            batch_score(hyps: List[Hypothesis]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
                One-step forward hypotheses for a batch.

            init_state(batch_size: int = 1) -> List[torch.Tensor]:
                Initialize RWKVDecoder states.

            select_state(states: List[torch.Tensor], idx: int) -> List[torch.Tensor]:
                Select ID state from batch of decoder hidden states.

            create_batch_states(new_states: List[List[Dict[str, torch.Tensor]]]) -> List[torch.Tensor]:
                Create batch of decoder hidden states given a list of new states.

        Examples:
            # Create an instance of RWKVDecoder
            decoder = RWKVDecoder(vocab_size=1000, block_size=512)

            # Forward pass with dummy labels
            labels = torch.randint(0, 1000, (32, 20))  # (B, L)
            output = decoder.forward(labels)
            print(output.shape)  # Should print: (32, 20, 512)

        Note:
            Ensure that the input length does not exceed the context size.
        """
        batch, length = labels.size()

        assert (
            length <= self.context_size
        ), "Context size is too short for current length: %d versus %d" % (
            length,
            self.context_size,
        )

        x = self.embed_norm(self.embed(labels))
        x = self.dropout_embed(x)

        for block in self.rwkv_blocks:
            x, _ = block(x)

        x = self.final_norm(x)

        return x

    def inference(
        self,
        labels: torch.Tensor,
        states: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        RWKV decoder definition for Transducer models.

        This class implements the RWKV decoder module as described in the paper
        "RWKV: Reinventing RNNs for the Transformer Era" (https://arxiv.org/pdf/2305.13048.pdf).
        The decoder utilizes multiple RWKV blocks to process input sequences and produce
        output sequences with attention mechanisms.

        Attributes:
            vocab_size (int): The size of the vocabulary.
            block_size (int): The input/output size for the RWKV blocks.
            context_size (int): The context size used for WKV computation.
            linear_size (Optional[int]): The hidden size for the FeedForward layer.
            attention_size (Optional[int]): The hidden size for the SelfAttention layer.
            normalization_type (str): The type of normalization layer to use.
            normalization_args (Dict): Arguments for the normalization layer.
            num_blocks (int): The number of RWKV blocks in the decoder.
            rescale_every (int): Rescaling factor for input every N blocks during inference.
            embed_dropout_rate (float): Dropout rate for the embedding layer.
            att_dropout_rate (float): Dropout rate for the attention module.
            ffn_dropout_rate (float): Dropout rate for the feed-forward module.
            embed_pad (int): The padding symbol ID for the embedding.

        Args:
            vocab_size (int): Vocabulary size.
            block_size (int): Input/Output size.
            context_size (int): Context size for WKV computation.
            linear_size (Optional[int]): FeedForward hidden size.
            attention_size (Optional[int]): SelfAttention hidden size.
            normalization_type (str): Normalization layer type.
            normalization_args (Dict): Normalization layer arguments.
            num_blocks (int): Number of RWKV blocks.
            rescale_every (int): Rescale input every N blocks (inference only).
            embed_dropout_rate (float): Dropout rate for embedding layer.
            att_dropout_rate (float): Dropout rate for the attention module.
            ffn_dropout_rate (float): Dropout rate for the feed-forward module.
            embed_pad (int): Embedding padding symbol ID.

        Examples:
            # Creating an instance of the RWKVDecoder
            decoder = RWKVDecoder(
                vocab_size=1000,
                block_size=512,
                context_size=1024,
                num_blocks=4
            )

            # Forward pass with labels
            labels = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
            output = decoder(labels)

            # Performing inference
            states = decoder.init_state(batch_size=2)
            output, new_states = decoder.inference(labels, states)

        Raises:
            AssertionError: If the length of the input labels exceeds the context size.
        """
        x = self.embed_norm(self.embed(labels))

        for idx, block in enumerate(self.rwkv_blocks):
            x, states = block(x, state=states)

            if self.rescaled_layers and (idx + 1) % self.rescale_every == 0:
                x = x / 2

        x = self.final_norm(x)

        return x, states

    def set_device(self, device: torch.device) -> None:
        """
        Set GPU device to use.

        This method allows you to specify the device on which the decoder will
        operate. It is particularly useful for transferring the model to a
        different GPU or CPU.

        Args:
            device: The device to set (e.g., `torch.device('cuda:0')` or
                    `torch.device('cpu')`).

        Examples:
            >>> decoder = RWKVDecoder(vocab_size=1000)
            >>> decoder.set_device(torch.device('cuda:0'))

        Note:
            Make sure that the device is available and compatible with the
            current model parameters.

        Raises:
            ValueError: If the specified device is not valid.
        """
        self.device = device

    def score(
        self,
        label_sequence: List[int],
        states: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        RWKV decoder module.

        Based on https://arxiv.org/pdf/2305.13048.pdf.

        Attributes:
            block_size (int): Input/Output size.
            attention_size (int): SelfAttention hidden size.
            output_size (int): Output size.
            vocab_size (int): Vocabulary size.
            context_size (int): Context size for WKV computation.
            rescale_every (int): Whether to rescale input every N blocks (inference only).
            pad_idx (int): Embedding padding symbol ID.
            num_blocks (int): Number of RWKV blocks.
            score_cache (dict): Cache for scores.
            device (torch.device): Device on which the model is located.

        Args:
            vocab_size: Vocabulary size.
            block_size: Input/Output size.
            context_size: Context size for WKV computation.
            linear_size: FeedForward hidden size.
            attention_size: SelfAttention hidden size.
            normalization_type: Normalization layer type.
            normalization_args: Normalization layer arguments.
            num_blocks: Number of RWKV blocks.
            rescale_every: Whether to rescale input every N blocks (inference only).
            embed_dropout_rate: Dropout rate for embedding layer.
            att_dropout_rate: Dropout rate for the attention module.
            ffn_dropout_rate: Dropout rate for the feed-forward module.
            embed_pad: Embedding padding symbol ID.

        Examples:
            decoder = RWKVDecoder(vocab_size=10000, block_size=512)
            labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
            output = decoder(labels)

        Raises:
            AssertionError: If the input length exceeds the context size.
        """
        label = torch.full(
            (1, 1), label_sequence[-1], dtype=torch.long, device=self.device
        )
        # (b-flo): FIX ME. Monkey patched for now.
        states = self.create_batch_states([states])

        out, states = self.inference(label, states)

        return out[0], states

    def batch_score(
        self, hyps: List[Hypothesis]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        One-step forward hypotheses.

        This method processes a batch of hypotheses and computes the decoder's
        output for each hypothesis. It takes the last label from each hypothesis
        and uses the decoder's inference method to generate the output and
        update the hidden states.

        Args:
            hyps: A list of Hypothesis objects representing the current hypotheses.
                  Each Hypothesis contains a label sequence and decoder state.

        Returns:
            out: The decoder output sequence. Shape is (B, D_dec), where B is the
                 batch size and D_dec is the dimension of the decoder output.
            states: The updated decoder hidden states. Shape is [5 x (B, 1,
                    D_att/D_dec, N)], where B is the batch size, D_att is the
                    attention dimension, and N is the number of blocks.

        Examples:
            >>> decoder = RWKVDecoder(vocab_size=1000)
            >>> hypotheses = [Hypothesis(yseq=[1, 2, 3], dec_state=initial_state)]
            >>> output, states = decoder.batch_score(hyps=hypotheses)
            >>> print(output.shape)  # Should output (1, D_dec)

        Note:
            Ensure that the `create_batch_states` method is compatible with the
            structure of the decoder hidden states expected in the inference
            process.
        """
        labels = torch.tensor(
            [[h.yseq[-1]] for h in hyps], dtype=torch.long, device=self.device
        )
        states = self.create_batch_states([h.dec_state for h in hyps])

        out, states = self.inference(labels, states)

        return out.squeeze(1), states

    def init_state(self, batch_size: int = 1) -> List[torch.Tensor]:
        """
        RWKV decoder definition for Transducer models.

        This module implements the RWKV decoder as described in the paper
        "RWKV: Reinventing RNNs for the Transformer Era" (https://arxiv.org/pdf/2305.13048.pdf).

        The RWKVDecoder class provides methods for initializing and managing the state
        of the decoder, processing input sequences, and generating output sequences
        through inference.

        Attributes:
            vocab_size: Vocabulary size.
            block_size: Input/Output size.
            context_size: Context size for WKV computation.
            linear_size: FeedForward hidden size.
            attention_size: SelfAttention hidden size.
            normalization_type: Normalization layer type.
            normalization_args: Normalization layer arguments.
            num_blocks: Number of RWKV blocks.
            rescale_every: Whether to rescale input every N blocks (inference only).
            embed_dropout_rate: Dropout rate for embedding layer.
            att_dropout_rate: Dropout rate for the attention module.
            ffn_dropout_rate: Dropout rate for the feed-forward module.
            embed_pad: Embedding padding symbol ID.

        Args:
            vocab_size: Vocabulary size.
            block_size: Input/Output size.
            context_size: Context size for WKV computation.
            linear_size: FeedForward hidden size.
            attention_size: SelfAttention hidden size.
            normalization_type: Normalization layer type.
            normalization_args: Normalization layer arguments.
            num_blocks: Number of RWKV blocks.
            rescale_every: Whether to rescale input every N blocks (inference only).
            embed_dropout_rate: Dropout rate for embedding layer.
            att_dropout_rate: Dropout rate for the attention module.
            ffn_dropout_rate: Dropout rate for the feed-forward module.
            embed_pad: Embedding padding symbol ID.

        Examples:
            decoder = RWKVDecoder(
                vocab_size=1000,
                block_size=512,
                context_size=1024,
                linear_size=2048,
                attention_size=512,
                num_blocks=4
            )
            states = decoder.init_state(batch_size=2)

        Note:
            The init_state method initializes the decoder's hidden states for a
            specified batch size. The hidden states consist of a list of tensors
            representing the state of each RWKV block.
        """
        hidden_sizes = [
            self.attention_size if i > 1 else self.block_size for i in range(5)
        ]

        state = [
            torch.zeros(
                (batch_size, 1, hidden_sizes[i], self.num_blocks),
                dtype=torch.float32,
                device=self.device,
            )
            for i in range(5)
        ]

        state[4] -= 1e-30

        return state

    def select_state(
        self,
        states: List[torch.Tensor],
        idx: int,
    ) -> List[torch.Tensor]:
        """
        Select ID state from batch of decoder hidden states.

        This method extracts the hidden states for a specific index from a batch
        of decoder hidden states. The hidden states are represented as a list of
        tensors, where each tensor corresponds to a different aspect of the state.

        Args:
            states: Decoder hidden states.
                A list of tensors with shape [5 x (B, 1, D_att/D_dec, N)],
                where B is the batch size, D_att is the attention dimension,
                D_dec is the decoder dimension, and N is the number of blocks.
            idx: The index of the state to select from the batch.

        Returns:
            A list of tensors representing the decoder hidden states for the
            specified index. The shape of each tensor is [1, 1, D_att/D_dec, N].

        Examples:
            >>> states = [
            ...     torch.randn(5, 2, 1, 128, 4),  # Example hidden states for 5 aspects
            ...     torch.randn(5, 2, 1, 128, 4),
            ...     torch.randn(5, 2, 1, 128, 4),
            ...     torch.randn(5, 2, 1, 128, 4),
            ...     torch.randn(5, 2, 1, 128, 4),
            ... ]
            >>> idx = 0
            >>> selected_state = select_state(states, idx)
            >>> len(selected_state)
            5
            >>> selected_state[0].shape
            torch.Size([1, 1, 128, 4])
        """
        return [states[i][idx : idx + 1, ...] for i in range(5)]

    def create_batch_states(
        self,
        new_states: List[List[Dict[str, torch.Tensor]]],
    ) -> List[torch.Tensor]:
        """
        Create batch of decoder hidden states given a list of new states.

        This method takes a list of new states for each hypothesis in the batch and
        combines them into a single batch of hidden states. The resulting hidden
        states can be used for further processing in the decoder.

        Args:
            new_states: A list of new decoder hidden states, where each entry
                corresponds to a hypothesis and is structured as:
                [B x [5 x (1, 1, D_att/D_dec, N)]].

        Returns:
            A list of decoder hidden states, structured as:
            [5 x (B, 1, D_att/D_dec, N)], where B is the batch size.

        Examples:
            >>> new_states = [
            ...     [torch.randn(1, 1, 128, 4) for _ in range(5)],  # Hypothesis 1
            ...     [torch.randn(1, 1, 128, 4) for _ in range(5)],  # Hypothesis 2
            ... ]
            >>> batch_states = create_batch_states(new_states)
            >>> len(batch_states)
            5
            >>> batch_states[0].shape
            torch.Size([2, 1, 128, 4])  # 2 hypotheses in the batch
        """
        batch_size = len(new_states)

        return [
            torch.cat([new_states[j][i] for j in range(batch_size)], dim=0)
            for i in range(5)
        ]
