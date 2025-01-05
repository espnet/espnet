"""MEGA decoder definition for Transducer models."""

import math
from typing import Dict, List, Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.asr_transducer.activation import get_activation
from espnet2.asr_transducer.beam_search_transducer import Hypothesis
from espnet2.asr_transducer.decoder.abs_decoder import AbsDecoder
from espnet2.asr_transducer.decoder.blocks.mega import MEGA
from espnet2.asr_transducer.decoder.modules.mega.feed_forward import (
    NormalizedPositionwiseFeedForward,
)
from espnet2.asr_transducer.normalization import get_normalization


class MEGADecoder(AbsDecoder):
    """
    MEGA decoder module for Transducer models.

    This class implements the MEGA (Memory-Enhanced Generative Attention) decoder
    for sequence-to-sequence tasks, inspired by the paper:
    https://arxiv.org/pdf/2209.10655.pdf.

    Attributes:
        vocab_size (int): Vocabulary size for the decoder.
        output_size (int): Size of the decoder output.
        chunk_size (int): Chunk size for attention computation.
        mega_num_heads (int): Number of attention heads in MEGA.
        mega_att_k_size (int): Size of the keys in the attention mechanism.
        mega_att_v_size (int): Size of the values in the attention mechanism.
        mega_ema_size (int): Size of the EMA module.
        mega_ema_num_heads (int): Number of heads in the EMA.
        pad_idx (int): Padding symbol ID for embeddings.
        num_blocks (int): Number of MEGA blocks.
        score_cache (dict): Cache for storing computed scores.
        device (torch.device): Device (CPU or GPU) used for computation.

    Args:
        vocab_size (int): Vocabulary size.
        block_size (int, optional): Input/Output size. Defaults to 512.
        linear_size (int, optional): NormalizedPositionwiseFeedForward hidden size.
            Defaults to 1024.
        qk_size (int, optional): Shared query and key size for attention module.
            Defaults to 128.
        v_size (int, optional): Value size for attention module. Defaults to 1024.
        num_heads (int, optional): Number of EMA heads. Defaults to 4.
        rel_pos_bias_type (str, optional): Type of relative position bias in
            attention module. Defaults to "simple".
        max_positions (int, optional): Maximum number of position for
            RelativePositionBias. Defaults to 2048.
        truncation_length (Optional[int], optional): Maximum length for truncation
            in EMA module. Defaults to None.
        normalization_type (str, optional): Normalization layer type. Defaults to
            "layer_norm".
        normalization_args (Dict, optional): Normalization layer arguments.
            Defaults to an empty dictionary.
        activation_type (str, optional): Activation function type. Defaults to
            "swish".
        activation_args (Dict, optional): Activation function arguments.
            Defaults to an empty dictionary.
        chunk_size (int, optional): Chunk size for attention computation
            (-1 = full context). Defaults to -1.
        num_blocks (int, optional): Number of MEGA blocks. Defaults to 4.
        dropout_rate (float, optional): Dropout rate for MEGA internal modules.
            Defaults to 0.0.
        embed_dropout_rate (float, optional): Dropout rate for embedding layer.
            Defaults to 0.0.
        att_dropout_rate (float, optional): Dropout rate for the attention module.
            Defaults to 0.0.
        ema_dropout_rate (float, optional): Dropout rate for the EMA module.
            Defaults to 0.0.
        ffn_dropout_rate (float, optional): Dropout rate for the feed-forward module.
            Defaults to 0.0.
        embed_pad (int, optional): Embedding padding symbol ID. Defaults to 0.

    Examples:
        Initialize a MEGADecoder:
            decoder = MEGADecoder(
                vocab_size=10000,
                block_size=512,
                num_blocks=4,
                activation_type="relu"
            )

        Forward pass through the decoder:
            labels = torch.randint(0, 10000, (32, 20))  # Batch of 32, length 20
            outputs = decoder(labels)

        Inference with states:
            states = decoder.init_state(batch_size=32)
            output, new_states = decoder.inference(labels, states)

    Raises:
        ValueError: If any of the arguments are out of expected range or format.
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 512,
        linear_size: int = 1024,
        qk_size: int = 128,
        v_size: int = 1024,
        num_heads: int = 4,
        rel_pos_bias_type: str = "simple",
        max_positions: int = 2048,
        truncation_length: Optional[int] = None,
        normalization_type: str = "layer_norm",
        normalization_args: Dict = {},
        activation_type: str = "swish",
        activation_args: Dict = {},
        chunk_size: int = -1,
        num_blocks: int = 4,
        dropout_rate: float = 0.0,
        embed_dropout_rate: float = 0.0,
        att_dropout_rate: float = 0.0,
        ema_dropout_rate: float = 0.0,
        ffn_dropout_rate: float = 0.0,
        embed_pad: int = 0,
    ) -> None:
        """Construct a MEGADecoder object."""
        super().__init__()

        self.embed = torch.nn.Embedding(vocab_size, block_size, padding_idx=embed_pad)
        self.dropout_embed = torch.nn.Dropout(p=embed_dropout_rate)

        activation = get_activation(activation_type, **activation_args)
        norm_class, norm_args = get_normalization(
            normalization_type, **normalization_args
        )

        self.mega_blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [
                        MEGA(
                            block_size,
                            num_heads=num_heads,
                            qk_size=qk_size,
                            v_size=v_size,
                            activation=activation,
                            normalization=norm_class(block_size, **norm_args),
                            rel_pos_bias_type=rel_pos_bias_type,
                            max_positions=max_positions,
                            truncation_length=truncation_length,
                            chunk_size=chunk_size,
                            dropout_rate=dropout_rate,
                            att_dropout_rate=att_dropout_rate,
                            ema_dropout_rate=ema_dropout_rate,
                        ),
                        NormalizedPositionwiseFeedForward(
                            block_size,
                            linear_size,
                            normalization=norm_class(block_size, **norm_args),
                            activation=activation,
                            dropout_rate=ffn_dropout_rate,
                        ),
                    ]
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_norm = norm_class(block_size, **norm_args)

        self.vocab_size = vocab_size
        self.output_size = block_size
        self.chunk_size = chunk_size

        self.mega_num_heads = num_heads
        self.mega_att_k_size = qk_size
        self.mega_att_v_size = v_size
        self.mega_ema_size = block_size
        self.mega_ema_num_heads = num_heads

        self.pad_idx = embed_pad
        self.num_blocks = num_blocks

        self.score_cache = {}

        self.device = next(self.parameters()).device

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        MEGA decoder module.

        This class implements the MEGA decoder as described in the paper
        "MEGA: A New Decoder for ASR" (https://arxiv.org/pdf/2209.10655.pdf).

        Attributes:
            embed: Embedding layer for input sequences.
            dropout_embed: Dropout layer applied to the embedding output.
            mega_blocks: A list of MEGA blocks, each containing a MEGA
                module and a Normalized Positionwise Feed Forward module.
            final_norm: Final normalization layer applied to the output.
            vocab_size: Size of the vocabulary.
            output_size: Output size of the decoder.
            chunk_size: Chunk size for attention computation.
            mega_num_heads: Number of heads in the MEGA attention.
            mega_att_k_size: Size of the query and key in attention.
            mega_att_v_size: Size of the value in attention.
            mega_ema_size: Size of the EMA module.
            mega_ema_num_heads: Number of heads in the EMA module.
            pad_idx: Padding index for the embedding layer.
            num_blocks: Number of MEGA blocks.
            score_cache: Cache for previously computed scores.
            device: Device to which the model is allocated.

        Args:
            vocab_size: Vocabulary size.
            block_size: Input/Output size.
            linear_size: NormalizedPositionwiseFeedForward hidden size.
            qk_size: Shared query and key size for attention module.
            v_size: Value size for attention module.
            num_heads: Number of EMA heads.
            rel_pos_bias_type: Type of relative position bias in attention module.
            max_positions: Maximum number of positions for RelativePositionBias.
            truncation_length: Maximum length for truncation in EMA module.
            normalization_type: Normalization layer type.
            normalization_args: Normalization layer arguments.
            activation_type: Activation function type.
            activation_args: Activation function arguments.
            chunk_size: Chunk size for attention computation (-1 = full context).
            num_blocks: Number of MEGA blocks.
            dropout_rate: Dropout rate for MEGA internal modules.
            embed_dropout_rate: Dropout rate for embedding layer.
            att_dropout_rate: Dropout rate for the attention module.
            ema_dropout_rate: Dropout rate for the EMA module.
            ffn_dropout_rate: Dropout rate for the feed-forward module.
            embed_pad: Embedding padding symbol ID.

        Examples:
            >>> decoder = MEGADecoder(vocab_size=1000)
            >>> input_tensor = torch.randint(0, 1000, (32, 50))  # (B, L)
            >>> output = decoder(input_tensor)
            >>> output.shape
            torch.Size([32, 50, block_size])
        """
        batch, length = labels.size()

        if 0 < self.chunk_size < length and length % self.chunk_size != 0:
            num_paddings = (
                math.ceil(length / self.chunk_size) * self.chunk_size - length
            )
            labels = torch.nn.functional.pad(
                labels, (0, num_paddings), value=self.pad_idx
            )
        else:
            num_paddings = 0

        mask = (labels == self.pad_idx).unsqueeze(1)
        mask[..., 0] = False
        mask = mask.to(device=labels.device, dtype=torch.bool)

        _length = self.chunk_size if 0 < self.chunk_size < length else length

        attn_mask = torch.ones(
            (_length, _length), device=labels.device, dtype=torch.bool
        )
        attn_mask = torch.triu(attn_mask, 1, out=attn_mask).unsqueeze(0)

        x = self.dropout_embed(self.embed(labels)).transpose(0, 1)

        for idx, (mega_block, nffn) in enumerate(self.mega_blocks):
            x, _ = mega_block(x, mask=mask, attn_mask=attn_mask)

            x = nffn(x)

        out = self.final_norm(x).transpose(0, 1)

        if num_paddings > 0:
            out = out[:, :length, :]

        return out

    def inference(
        self,
        labels: torch.Tensor,
        states: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        MEGA decoder module for Transducer models.

        This class implements the MEGA decoder as described in the paper
        "MEGA: A Multiscale Encoder-Decoder Architecture for Speech Recognition"
        (https://arxiv.org/pdf/2209.10655.pdf).

        Attributes:
            vocab_size (int): Size of the vocabulary.
            output_size (int): Size of the output block.
            chunk_size (int): Size of the chunks for attention computation.
            mega_num_heads (int): Number of heads in the MEGA attention.
            mega_att_k_size (int): Size of the key in MEGA attention.
            mega_att_v_size (int): Size of the value in MEGA attention.
            mega_ema_size (int): Size of the EMA in MEGA.
            mega_ema_num_heads (int): Number of heads in EMA.
            pad_idx (int): Padding index for the embedding.
            num_blocks (int): Number of MEGA blocks.
            score_cache (dict): Cache for score computations.
            device (torch.device): Device on which the model is located.

        Args:
            vocab_size (int): Vocabulary size.
            block_size (int): Input/Output size.
            linear_size (int): NormalizedPositionwiseFeedForward hidden size.
            qk_size (int): Shared query and key size for attention module.
            v_size (int): Value size for attention module.
            num_heads (int): Number of EMA heads.
            rel_pos_bias_type (str): Type of relative position bias in attention module.
            max_positions (int): Maximum number of position for RelativePositionBias.
            truncation_length (Optional[int]): Maximum length for truncation in EMA.
            normalization_type (str): Normalization layer type.
            normalization_args (Dict): Normalization layer arguments.
            activation_type (str): Activation function type.
            activation_args (Dict): Activation function arguments.
            chunk_size (int): Chunk size for attention computation (-1 = full context).
            num_blocks (int): Number of MEGA blocks.
            dropout_rate (float): Dropout rate for MEGA internal modules.
            embed_dropout_rate (float): Dropout rate for embedding layer.
            att_dropout_rate (float): Dropout rate for the attention module.
            ema_dropout_rate (float): Dropout rate for the EMA module.
            ffn_dropout_rate (float): Dropout rate for the feed-forward module.
            embed_pad (int): Embedding padding symbol ID.

        Examples:
            # Initialize a MEGADecoder instance
            decoder = MEGADecoder(
                vocab_size=10000,
                block_size=512,
                linear_size=1024,
                num_heads=4,
                num_blocks=4,
            )

            # Forward pass with labels
            labels = torch.tensor([[1, 2, 3], [4, 5, 0]])
            output = decoder(labels)

            # Inference with previous states
            states = decoder.init_state(batch_size=2)
            out, new_states = decoder.inference(labels, states)

        Raises:
            ValueError: If the input tensor shapes do not match the expected dimensions.
        """
        x = self.embed(labels).transpose(0, 1)

        new_states = []
        for idx, (mega_block, nffn) in enumerate(self.mega_blocks):
            x, new_state = mega_block(x, state=states[idx])

            x = nffn(x)

            new_states.append(new_state)

        out = self.final_norm(x).transpose(0, 1)

        return out, new_states

    def set_device(self, device: torch.device) -> None:
        """
        Set GPU device to use.

        This method allows the user to specify the GPU device on which the
        MEGADecoder will run. It is important for managing the device
        placement of tensors and operations in PyTorch.

        Args:
            device: The device to set for the decoder. This should be a
                torch.device object representing the desired GPU or CPU
                device.

        Examples:
            >>> decoder = MEGADecoder(vocab_size=1000)
            >>> decoder.set_device(torch.device('cuda:0'))  # Use GPU 0
            >>> decoder.set_device(torch.device('cpu'))      # Use CPU

        Note:
            Ensure that the specified device is available and valid in your
            PyTorch installation. You can check available devices using
            `torch.cuda.is_available()` and `torch.cuda.device_count()`.
        """
        self.device = device

    def score(
        self,
        label_sequence: List[int],
        states: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        MEGA decoder module.

        Based on https://arxiv.org/pdf/2209.10655.pdf.

        This class implements the MEGA decoder, which is designed for transducer
        models in automatic speech recognition (ASR). The decoder uses a
        combination of attention mechanisms and feed-forward networks to process
        input sequences and generate output sequences.

        Attributes:
            vocab_size: Vocabulary size.
            output_size: Size of the output block.
            chunk_size: Size of chunks for attention computation.
            mega_num_heads: Number of heads in the MEGA attention.
            mega_att_k_size: Shared query and key size for the attention module.
            mega_att_v_size: Value size for the attention module.
            mega_ema_size: Size of the EMA (Exponential Moving Average).
            mega_ema_num_heads: Number of EMA heads.
            pad_idx: Padding index for the embedding layer.
            num_blocks: Number of MEGA blocks.
            score_cache: Cache for previously computed scores.
            device: The device (CPU or GPU) the model is currently using.

        Args:
            vocab_size: Vocabulary size.
            block_size: Input/Output size.
            linear_size: NormalizedPositionwiseFeedForward hidden size.
            qk_size: Shared query and key size for attention module.
            v_size: Value size for attention module.
            num_heads: Number of EMA heads.
            rel_pos_bias_type: Type of relative position bias in attention module.
            max_positions: Maximum number of position for RelativePositionBias.
            truncation_length: Maximum length for truncation in EMA module.
            normalization_type: Normalization layer type.
            normalization_args: Normalization layer arguments.
            activation_type: Activation function type.
            activation_args: Activation function arguments.
            chunk_size: Chunk size for attention computation (-1 = full context).
            num_blocks: Number of MEGA blocks.
            dropout_rate: Dropout rate for MEGA internal modules.
            embed_dropout_rate: Dropout rate for embedding layer.
            att_dropout_rate: Dropout rate for the attention module.
            ema_dropout_rate: Dropout rate for the EMA module.
            ffn_dropout_rate: Dropout rate for the feed-forward module.
            embed_pad: Embedding padding symbol ID.

        Examples:
            >>> decoder = MEGADecoder(vocab_size=1000, block_size=512)
            >>> labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> output = decoder(labels)
            >>> print(output.shape)  # Output shape will be (B, U, D_dec)
        """
        str_labels = "_".join(map(str, label_sequence))

        if str_labels in self.score_cache:
            out, states = self.score_cache[str_labels]
        else:
            label = torch.full(
                (1, 1), label_sequence[-1], dtype=torch.long, device=self.device
            )

            out, states = self.inference(label, states=states)

            self.score_cache[str_labels] = (out, states)

        return out[0], states

    def batch_score(
        self, hyps: List[Hypothesis]
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        One-step forward hypotheses.

        This method processes a batch of hypotheses and computes the decoder
        output for each hypothesis in the batch. It retrieves the last label
        from each hypothesis and creates a corresponding batch of states.

        Args:
            hyps: A list of Hypothesis objects, each containing the current
                  label sequence and decoder state.

        Returns:
            out: A tensor containing the decoder output sequence for each
                 hypothesis in the batch, shape (B, D_dec).
            states: A list of dictionaries containing the updated decoder
                    hidden states for each hypothesis.

        Examples:
            >>> from espnet2.asr_transducer.decoder.blocks.mega import Hypothesis
            >>> hyps = [Hypothesis(yseq=[1, 2, 3], dec_state={...}),
            ...         Hypothesis(yseq=[1, 2, 4], dec_state={...})]
            >>> decoder = MEGADecoder(...)
            >>> output, updated_states = decoder.batch_score(hyps)
        """
        labels = torch.tensor(
            [[h.yseq[-1]] for h in hyps], dtype=torch.long, device=self.device
        )
        states = self.create_batch_states([h.dec_state for h in hyps])

        out, states = self.inference(labels, states=states)

        return out.squeeze(1), states

    def init_state(self, batch_size: int = 0) -> List[Dict[str, torch.Tensor]]:
        """
        Initialize MEGADecoder states.

        This method creates the initial hidden states for the MEGADecoder,
        which are necessary for processing input sequences. The states are
        initialized to zero tensors based on the output size and the number
        of MEGA blocks.

        Args:
            batch_size: Batch size. This parameter is not used in the current
                implementation but can be useful for future enhancements.

        Returns:
            states: Decoder hidden states. A list of dictionaries where each
                dictionary corresponds to a block in the decoder. Each
                dictionary contains:
                    - "ema_state": A tensor of shape (output_size, num_heads)
                      representing the Exponential Moving Average state.
                    - "prev_key": A tensor of shape (1, 1, qk_size) representing
                      the previous key state.
                    - "prev_value": A tensor of shape (1, 1, v_size) representing
                      the previous value state.

        Examples:
            >>> decoder = MEGADecoder(vocab_size=1000)
            >>> states = decoder.init_state(batch_size=32)
            >>> len(states)
            4  # Assuming num_blocks is set to 4
        """
        return [
            {
                "ema_state": torch.zeros(
                    (self.output_size, self.mega_ema_num_heads), device=self.device
                ),
                "prev_key": torch.zeros(
                    (1, 1, self.mega_att_k_size), device=self.device
                ),
                "prev_value": torch.zeros(
                    (1, 1, self.mega_att_v_size), device=self.device
                ),
            }
            for _ in range(self.num_blocks)
        ]

    def select_state(
        self,
        states: List[Dict[str, torch.Tensor]],
        idx: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Select ID state from batch of decoder hidden states.

        This method retrieves the hidden states for a specific index from a batch
        of decoder states. It extracts the `ema_state`, `prev_key`, and `prev_value`
        for each block in the decoder.

        Args:
            states: Decoder hidden states. A list of dictionaries, where each
                dictionary contains the states for a specific block. Each dictionary
                should have the keys:
                - "ema_state": Tensor containing the EMA state for the block.
                - "prev_key": Tensor containing the previous key for the block.
                - "prev_value": Tensor containing the previous value for the block.
            idx: The index of the state to select from each block's hidden states.

        Returns:
            A list of dictionaries, where each dictionary contains the selected
            hidden states for the given index. The structure is the same as the
            input states but only contains the states corresponding to the specified
            index.

        Examples:
            >>> states = [
            ...     {"ema_state": torch.randn(5, 4), "prev_key": torch.randn(1, 1, 4),
            ...      "prev_value": torch.randn(1, 1, 4)},
            ...     {"ema_state": torch.randn(5, 4), "prev_key": torch.randn(1, 1, 4),
            ...      "prev_value": torch.randn(1, 1, 4)},
            ... ]
            >>> idx = 2
            >>> selected = select_state(states, idx)
            >>> len(selected)
            2  # Two blocks were selected from the states.
        """
        return [
            {
                "ema_state": states[n_b]["ema_state"][idx],
                "prev_key": states[n_b]["prev_key"][idx],
                "prev_value": states[n_b]["prev_value"][idx],
            }
            for n_b in range(self.num_blocks)
        ]

    def stack_qk_states(
        self, state_list: List[torch.Tensor], dim: int
    ) -> List[torch.Tensor]:
        """
        Stack query or key states with different lengths.

        This method takes a list of query or key states, which may have
        different lengths, and stacks them into a tensor of shape
        (num_states, max_len, dim). The shorter states are padded with
        zeros to match the length of the longest state in the list.

        Args:
            state_list: List of query or key states, where each state is
                a tensor of shape (length, dim).
            dim: The size of the last dimension of each state tensor.

        Returns:
            new_state: A tensor containing stacked query/key states with
            shape (num_states, max_len, dim), where num_states is the
            number of states in the input list and max_len is the length
            of the longest state.

        Examples:
            >>> states = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])]
            >>> stacked = stack_qk_states(states, dim=2)
            >>> print(stacked.shape)
            torch.Size([2, 2, 2])  # 2 states, max length 2, dimension 2
        """
        max_len = max([(state.size(0)) for state in state_list])

        new_state = torch.zeros((len(state_list), max_len, dim))

        for idx, state in enumerate(state_list):
            new_state[idx, -state.size(0) :, :] = state

        return new_state

    def create_batch_states(
        self,
        new_states: List[List[Dict[str, torch.Tensor]]],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Create batch of decoder hidden states given a list of new states.

        This method constructs a new batch of decoder hidden states from a
        list of individual states for each block. It aggregates the states
        across the batch dimension, allowing for efficient processing of
        hypotheses during inference.

        Args:
            new_states: A list containing decoder hidden states, where each
                element is a list of dictionaries representing the states
                for each block in the decoder. The structure is
                [B x [N x Dict]], where B is the batch size and N is the
                number of blocks.

        Returns:
            A list of dictionaries representing the aggregated decoder hidden
            states for each block. The structure is [N x Dict], where N
            is the number of blocks.

        Examples:
            >>> new_states = [
            ...     [{'ema_state': torch.tensor([[0.1, 0.2]]),
            ...       'prev_key': torch.tensor([[0.3]]),
            ...       'prev_value': torch.tensor([[0.4]])}],
            ...     [{'ema_state': torch.tensor([[0.5, 0.6]]),
            ...       'prev_key': torch.tensor([[0.7]]),
            ...       'prev_value': torch.tensor([[0.8]])}]
            ... ]
            >>> batch_states = decoder.create_batch_states(new_states)
            >>> print(batch_states)
            [{'ema_state': tensor([[0.1, 0.2], [0.5, 0.6]]),
              'prev_key': tensor([[0.3], [0.7]]),
              'prev_value': tensor([[0.4], [0.8]])}]
        """
        return [
            {
                "ema_state": torch.stack(
                    [state[n_b]["ema_state"] for state in new_states]
                ),
                "prev_key": self.stack_qk_states(
                    [state[n_b]["prev_key"] for state in new_states],
                    self.mega_att_k_size,
                ),
                "prev_value": self.stack_qk_states(
                    [state[n_b]["prev_value"] for state in new_states],
                    self.mega_att_v_size,
                ),
            }
            for n_b in range(self.num_blocks)
        ]
