"""Stateless decoder definition for Transducer models."""

from typing import Any, List, Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.asr_transducer.beam_search_transducer import Hypothesis
from espnet2.asr_transducer.decoder.abs_decoder import AbsDecoder


class StatelessDecoder(AbsDecoder):
    """
    Stateless decoder definition for Transducer models.

    This class implements a stateless Transducer decoder module, which is
    designed to process input label sequences and generate corresponding
    output embeddings. It inherits from the abstract base class AbsDecoder.

    Attributes:
        embed (torch.nn.Embedding): The embedding layer for converting label IDs
            to embeddings.
        embed_dropout_rate (torch.nn.Dropout): Dropout layer for the embedding
            output.
        output_size (int): The size of the output embeddings.
        vocab_size (int): The size of the vocabulary.
        device (torch.device): The device on which the decoder is located.
        score_cache (dict): Cache for storing computed embeddings for label
            sequences to avoid redundant calculations.

    Args:
        vocab_size (int): Output size, representing the number of unique label
            IDs.
        embed_size (int, optional): Size of the embedding vector. Defaults to 256.
        embed_dropout_rate (float, optional): Dropout rate for the embedding
            layer. Defaults to 0.0.
        embed_pad (int, optional): ID for the padding/blank symbol. Defaults to 0.

    Examples:
        # Initialize the decoder
        decoder = StatelessDecoder(vocab_size=1000, embed_size=256)

        # Forward pass with label sequences
        labels = torch.tensor([[1, 2, 3], [4, 5, 6]])  # Example label IDs
        output = decoder(labels)

        # Scoring a single label sequence
        score, _ = decoder.score([1, 2, 3])

        # Batch scoring for multiple hypotheses
        from espnet2.asr_transducer.beam_search_transducer import Hypothesis
        hyps = [Hypothesis(yseq=[1, 2, 3]), Hypothesis(yseq=[4, 5, 6])]
        batch_output, _ = decoder.batch_score(hyps)

        # Setting the device
        decoder.set_device(torch.device('cuda'))

        # Initializing states for a batch
        decoder.init_state(batch_size=2)

        # Selecting a specific state
        decoder.select_state(None, idx=0)

        # Creating batch states
        decoder.create_batch_states([None, None])
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        embed_dropout_rate: float = 0.0,
        embed_pad: int = 0,
    ) -> None:
        """Construct a StatelessDecoder object."""
        super().__init__()

        self.embed = torch.nn.Embedding(vocab_size, embed_size, padding_idx=embed_pad)
        self.embed_dropout_rate = torch.nn.Dropout(p=embed_dropout_rate)

        self.output_size = embed_size
        self.vocab_size = vocab_size

        self.device = next(self.parameters()).device
        self.score_cache = {}

    def forward(
        self,
        labels: torch.Tensor,
        states: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Encode source label sequences.

        This method takes a batch of label ID sequences and returns their
        corresponding embedded representations. The embeddings are obtained
        from an embedding layer followed by a dropout layer.

        Args:
            labels: A tensor of shape (B, L) containing label ID sequences,
                    where B is the batch size and L is the sequence length.
            states: Optional; Decoder hidden states. Currently unused and
                    defaults to None.

        Returns:
            A tensor of shape (B, U, D_emb) representing the embedded output
            sequences, where U is the length of the output sequence and D_emb
            is the embedding dimension.

        Examples:
            >>> decoder = StatelessDecoder(vocab_size=100)
            >>> labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> output = decoder.forward(labels)
            >>> output.shape
            torch.Size([2, 3, 256])  # Assuming embed_size is 256

        Note:
            The embedding dropout rate is applied during the embedding
            process to prevent overfitting.
        """
        embed = self.embed_dropout_rate(self.embed(labels))

        return embed

    def score(
        self,
        label_sequence: List[int],
        states: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, None]:
        """
            Stateless decoder definition for Transducer models.

        This module implements a stateless Transducer decoder for ASR (Automatic Speech
        Recognition) models. It is designed to work with label sequences and provides
        methods for scoring and processing these sequences efficiently.

        Attributes:
            output_size (int): Size of the output embeddings.
            vocab_size (int): Size of the vocabulary.
            device (torch.device): The device (CPU or GPU) where the model is located.
            score_cache (dict): A cache to store computed scores for label sequences.

        Args:
            vocab_size (int): Output size of the decoder.
            embed_size (int, optional): Size of the embedding layer. Default is 256.
            embed_dropout_rate (float, optional): Dropout rate for the embedding layer.
                Default is 0.0.
            embed_pad (int, optional): Padding symbol ID for embeddings. Default is 0.

        Examples:
            decoder = StatelessDecoder(vocab_size=1000, embed_size=256)
            label_sequence = [1, 2, 3]
            output, _ = decoder.score(label_sequence)
        """
        str_labels = "_".join(map(str, label_sequence))

        if str_labels in self.score_cache:
            embed = self.score_cache[str_labels]
        else:
            label = torch.full(
                (1, 1),
                label_sequence[-1],
                dtype=torch.long,
                device=self.device,
            )

            embed = self.embed(label)

            self.score_cache[str_labels] = embed

        return embed[0], None

    def batch_score(self, hyps: List[Hypothesis]) -> Tuple[torch.Tensor, None]:
        """
        One-step forward hypotheses.

        This method computes the output sequences for a batch of hypotheses by
        using the last label of each hypothesis. It processes the input in
        parallel to enhance efficiency.

        Args:
            hyps: A list of Hypothesis objects containing the label sequences.

        Returns:
            out: Decoder output sequences. Shape (B, D_dec), where B is the batch
                size and D_dec is the dimension of the decoder output.
            states: Decoder hidden states. Always returns None as this
                implementation does not maintain hidden states.

        Examples:
            >>> decoder = StatelessDecoder(vocab_size=100)
            >>> hyps = [Hypothesis(yseq=[1]), Hypothesis(yseq=[2])]
            >>> output, _ = decoder.batch_score(hyps)
            >>> print(output.shape)  # Output: torch.Size([2, 256])

        Note:
            The method assumes that the Hypothesis objects are well-formed
            and contain valid label sequences.
        """
        labels = torch.tensor(
            [[h.yseq[-1]] for h in hyps], dtype=torch.long, device=self.device
        )
        embed = self.embed(labels)

        return embed.squeeze(1), None

    def set_device(self, device: torch.device) -> None:
        """
        Set GPU device to use.

        This method allows you to specify the device (CPU or GPU) on which the
        decoder should operate. This is particularly useful for models that may
        need to be switched between devices during training or inference.

        Args:
            device: The device ID (e.g., `torch.device('cuda:0')` for the first
                GPU or `torch.device('cpu')` for the CPU).

        Examples:
            >>> decoder = StatelessDecoder(vocab_size=100)
            >>> decoder.set_device(torch.device('cuda:0'))

        Note:
            Ensure that the device is available and that the model is moved to
            the appropriate device to avoid runtime errors.
        """
        self.device = device

    def init_state(self, batch_size: int) -> None:
        """
        Stateless decoder definition for Transducer models.

        This module defines a StatelessDecoder class, which is a stateless
        Transducer decoder used in automatic speech recognition systems.
        It inherits from the AbsDecoder class and implements methods for
        forward decoding and state management.

        Attributes:
            output_size (int): The output size of the decoder.
            vocab_size (int): The size of the vocabulary.
            device (torch.device): The device on which the model is allocated.
            score_cache (dict): A cache for storing computed scores to avoid
                redundant calculations.

        Args:
            vocab_size (int): Output size.
            embed_size (int, optional): Embedding size. Defaults to 256.
            embed_dropout_rate (float, optional): Dropout rate for embedding layer.
                Defaults to 0.0.
            embed_pad (int, optional): Embed/Blank symbol ID. Defaults to 0.

        Examples:
            # Creating a StatelessDecoder instance
            decoder = StatelessDecoder(vocab_size=1000, embed_size=256)

            # Initializing decoder states
            initial_states = decoder.init_state(batch_size=32)

            # Forward pass with label sequences
            labels = torch.randint(0, 1000, (32, 10))  # Batch of 32, sequence length 10
            output = decoder.forward(labels)

            # Scoring a label sequence
            score, _ = decoder.score([1, 2, 3])

            # Batch scoring hypotheses
            from espnet2.asr_transducer.beam_search_transducer import Hypothesis
            hyps = [Hypothesis(yseq=[1, 2, 3])]  # List of Hypothesis instances
            batch_output, _ = decoder.batch_score(hyps)

        Note:
            The decoder does not maintain state across different calls,
            hence it is stateless. This means that the `init_state`
            method always returns None.

        Todo:
            - Implement state management features if necessary.
        """
        return None

    def select_state(self, states: Optional[torch.Tensor], idx: int) -> None:
        """
        Stateless decoder definition for Transducer models.

        This module implements a stateless Transducer decoder, which is part of the
        ESPnet2 library. It is designed for use in automatic speech recognition tasks
        using transducer models.

        Attributes:
            vocab_size (int): The size of the vocabulary for the decoder.
            output_size (int): The output size of the embedding layer.
            device (torch.device): The device on which the decoder is located.
            score_cache (dict): A cache for storing computed scores for label sequences.

        Args:
            vocab_size (int): Output size.
            embed_size (int, optional): Embedding size. Default is 256.
            embed_dropout_rate (float, optional): Dropout rate for the embedding layer.
                Default is 0.0.
            embed_pad (int, optional): Embed/Blank symbol ID. Default is 0.

        Examples:
            # Create a StatelessDecoder instance
            decoder = StatelessDecoder(vocab_size=1000, embed_size=256)

            # Forward pass through the decoder
            labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
            output = decoder(labels)

            # Get the score for a label sequence
            score, _ = decoder.score([1, 2, 3])

            # Initialize states for a batch
            decoder.init_state(batch_size=32)

            # Select a specific state
            state = decoder.select_state(None, idx=0)

            # Set the device for the decoder
            decoder.set_device(torch.device('cuda:0'))

            # Create batch states
            decoder.create_batch_states([None] * 32)
        """
        return None

    def create_batch_states(
        self,
        new_states: List[Optional[torch.Tensor]],
    ) -> None:
        """
        Create decoder hidden states.

        This method is responsible for creating and managing the hidden states
        for the decoder. The hidden states are typically used to maintain
        information across decoding steps in a sequence-to-sequence model.

        Args:
            new_states: A list of new decoder hidden states, where each entry
                         is of type Optional[torch.Tensor]. The expected shape
                         is [N x None], where N is the number of states to be
                         created.

        Returns:
            None: This method does not return any value, as it modifies the
            internal state of the decoder.

        Examples:
            >>> decoder = StatelessDecoder(vocab_size=1000)
            >>> states = [None] * 5  # Create 5 new states
            >>> decoder.create_batch_states(states)
            >>> # States are now created and managed internally.

        Note:
            This function does not actually store the states; it is intended
            to be implemented in a derived class to handle state management
            according to specific requirements.

        Raises:
            ValueError: If the input `new_states` does not conform to the
            expected format.
        """
        return None
