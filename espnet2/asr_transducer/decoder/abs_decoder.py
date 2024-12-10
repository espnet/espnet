"""Abstract decoder definition for Transducer models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


class AbsDecoder(torch.nn.Module, ABC):
    """
    Abstract decoder definition for Transducer models.

    This class serves as an abstract base class for all decoder modules 
    implemented for Transducer models in the ESPnet2 framework. It outlines 
    the necessary methods that any concrete decoder must implement to 
    function correctly within the architecture.

    Attributes:
        None

    Methods:
        forward: Encode source label sequences.
        score: One-step forward hypothesis.
        batch_score: One-step forward hypotheses for a batch.
        set_device: Set the GPU device to use.
        init_state: Initialize decoder states.
        select_state: Get specified ID state from batch of states.
        create_batch_states: Create a batch of decoder hidden states 
            given a list of new states.

    Args:
        labels (torch.Tensor): Label ID sequences for encoding.
        label_sequence (List[int]): Current label sequence for scoring.
        states (Union[List[Dict[str, torch.Tensor]], List[torch.Tensor], 
            Tuple[torch.Tensor, Optional[torch.Tensor]]]): Decoder hidden states.
        hyps (List[Any]): Hypotheses for batch scoring.
        device (torch.Tensor): Device ID to set.
        batch_size (int): Batch size for state initialization.
        idx (int, optional): State ID to extract (default is 0).
        new_states (List[Union[List[Dict[str, Optional[torch.Tensor]]], 
            List[List[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor]]]): 
            Decoder hidden states for batch creation.

    Returns:
        torch.Tensor: Decoder output sequences.
        Union[List[Dict[str, torch.Tensor]], List[torch.Tensor], 
            Tuple[torch.Tensor, Optional[torch.Tensor]]]: Decoder hidden states.

    Raises:
        NotImplementedError: If the method is not implemented in the 
            subclass.

    Examples:
        # Example of subclass implementation:
        class MyDecoder(AbsDecoder):
            def forward(self, labels):
                # Implement the forward method
                pass

            def score(self, label_sequence, states):
                # Implement the scoring method
                pass

        decoder = MyDecoder()
        output = decoder.forward(torch.tensor([1, 2, 3]))
        
    Note:
        This class is intended to be subclassed, and cannot be instantiated 
        directly.
    """

    @abstractmethod
    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Abstract decoder definition for Transducer models.

        This module defines the abstract base class for decoders in Transducer 
        models. The `AbsDecoder` class provides an interface for implementing 
        various decoding strategies.

        Attributes:
            None

        Methods:
            forward(labels: torch.Tensor) -> torch.Tensor:
                Encode source label sequences.
            
            score(label_sequence: List[int], states: Union[List[Dict[str, torch.Tensor]],
                List[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]) -> 
                Tuple[torch.Tensor, Union[List[Dict[str, torch.Tensor]], 
                List[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]]:
                One-step forward hypothesis.

            batch_score(hyps: List[Any]) -> Tuple[torch.Tensor, Union[List[Dict[str, 
                torch.Tensor]], List[torch.Tensor], Tuple[torch.Tensor, 
                Optional[torch.Tensor]]]]:
                One-step forward hypotheses.

            set_device(device: torch.Tensor) -> None:
                Set GPU device to use.

            init_state(batch_size: int) -> Union[List[Dict[str, torch.Tensor]], 
                List[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]:
                Initialize decoder states.

            select_state(states: Union[List[Dict[str, torch.Tensor]], 
                List[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]], 
                idx: int = 0) -> Union[List[Dict[str, torch.Tensor]], 
                List[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]:
                Get specified ID state from batch of states, if provided.

            create_batch_states(new_states: List[Union[List[Dict[str, 
                Optional[torch.Tensor]]], List[List[torch.Tensor]], 
                Tuple[torch.Tensor, Optional[torch.Tensor]]]]) -> Union[
                List[Dict[str, torch.Tensor]], List[torch.Tensor], 
                Tuple[torch.Tensor, Optional[torch.Tensor]]]:
                Create batch of decoder hidden states given a list of new states.

        Examples:
            To use the AbsDecoder class, you must subclass it and implement the 
            abstract methods:

            ```python
            class MyDecoder(AbsDecoder):
                def forward(self, labels: torch.Tensor) -> torch.Tensor:
                    # Implementation here
                    pass

                # Implement other abstract methods...
            ```

        Note:
            This class is not intended to be instantiated directly; it serves as 
            a blueprint for specific decoder implementations.

        Todo:
            Implement concrete subclasses for various decoding strategies.
        """
        raise NotImplementedError

    @abstractmethod
    def score(
        self,
        label_sequence: List[int],
        states: Union[
            List[Dict[str, torch.Tensor]],
            List[torch.Tensor],
            Tuple[torch.Tensor, Optional[torch.Tensor]],
        ],
    ) -> Tuple[
        torch.Tensor,
        Union[
            List[Dict[str, torch.Tensor]],
            List[torch.Tensor],
            Tuple[torch.Tensor, Optional[torch.Tensor]],
        ],
    ]:
        """
                Abstract decoder definition for Transducer models.

        This module defines the abstract class `AbsDecoder`, which serves as a base 
        for implementing various decoder architectures in a transducer model. The 
        class provides methods for encoding label sequences, scoring hypotheses, 
        and managing decoder states.

        Attributes:
            None

        Args:
            None

        Methods:
            forward: Encode source label sequences.
            score: One-step forward hypothesis.
            batch_score: One-step forward hypotheses.
            set_device: Set GPU device to use.
            init_state: Initialize decoder states.
            select_state: Get specified ID state from batch of states, if provided.
            create_batch_states: Create batch of decoder hidden states given a list 
                of new states.

        The following methods are abstract and must be implemented by any subclass:

        1. `forward(labels: torch.Tensor) -> torch.Tensor`: Encodes source label 
        sequences.
        
        2. `score(label_sequence: List[int], states: Union[List[Dict[str, torch.Tensor]], 
        List[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]) -> 
        Tuple[torch.Tensor, Union[List[Dict[str, torch.Tensor]], List[torch.Tensor], 
        Tuple[torch.Tensor, Optional[torch.Tensor]]]]`: Computes the score for 
        a single label sequence and returns the output sequence and updated states.
        
        3. `batch_score(hyps: List[Any]) -> Tuple[torch.Tensor, 
        Union[List[Dict[str, torch.Tensor]], List[torch.Tensor], 
        Tuple[torch.Tensor, Optional[torch.Tensor]]]]`: Computes the scores for 
        a batch of hypotheses and returns the output sequences and updated states.

        4. `set_device(device: torch.Tensor) -> None`: Sets the GPU device to use 
        for computations.

        5. `init_state(batch_size: int) -> Union[List[Dict[str, torch.Tensor]], 
        List[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]`: 
        Initializes decoder states based on the provided batch size.

        6. `select_state(states: Union[List[Dict[str, torch.Tensor]], 
        List[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]], 
        idx: int = 0) -> Union[List[Dict[str, torch.Tensor]], 
        List[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]`: 
        Retrieves the state corresponding to the specified index from the 
        batch of states.

        7. `create_batch_states(new_states: List[Union[List[Dict[str, 
        Optional[torch.Tensor]]], List[List[torch.Tensor]], 
        Tuple[torch.Tensor, Optional[torch.Tensor]]]]) -> 
        Union[List[Dict[str, torch.Tensor]], List[torch.Tensor], 
        Tuple[torch.Tensor, Optional[torch.Tensor]]]`: Creates a batch of 
        decoder hidden states from a list of new states.

        Examples:
            # Example subclass implementation
            class MyDecoder(AbsDecoder):
                def forward(self, labels):
                    pass
                
                def score(self, label_sequence, states):
                    pass
                
                def batch_score(self, hyps):
                    pass
                
                def set_device(self, device):
                    pass
                
                def init_state(self, batch_size):
                    pass
                
                def select_state(self, states, idx=0):
                    pass
                
                def create_batch_states(self, new_states):
                    pass
        """
        raise NotImplementedError

    @abstractmethod
    def batch_score(
        self,
        hyps: List[Any],
    ) -> Tuple[
        torch.Tensor,
        Union[
            List[Dict[str, torch.Tensor]],
            List[torch.Tensor],
            Tuple[torch.Tensor, Optional[torch.Tensor]],
        ],
    ]:
        """
        Compute the score for a batch of hypotheses.

        This method evaluates a batch of hypotheses and returns the 
        corresponding decoder output sequences along with their 
        associated hidden states. It is designed to facilitate the 
        scoring of multiple hypotheses simultaneously, which can 
        improve efficiency during the decoding process.

        Args:
            hyps: A list of hypotheses to score. Each hypothesis can 
                  be of any type, but typically it would be a sequence 
                  of label IDs or a structured representation of a 
                  hypothesis.

        Returns:
            out: A tensor containing the decoder output sequences for 
                 each hypothesis in the batch.
            states: The hidden states corresponding to the output 
                    sequences, which can be in various formats, 
                    including a list of dictionaries, a list of 
                    tensors, or a tuple of tensors.

        Examples:
            >>> decoder = MyDecoder()  # Assume MyDecoder implements AbsDecoder
            >>> hypotheses = [[1, 2, 3], [4, 5, 6]]  # Example hypotheses
            >>> outputs, states = decoder.batch_score(hyps=hypotheses)
        """
        raise NotImplementedError

    @abstractmethod
    def set_device(self, device: torch.Tensor) -> None:
        """
        Set GPU device to use.

        This method configures the decoder to use the specified GPU device for
        all tensor operations. It is essential to call this method before
        performing any computations to ensure that the decoder is using the
        correct device.

        Args:
            device: The target device ID, typically a string such as 'cuda:0'
                or an integer representing the GPU device number.

        Raises:
            ValueError: If the provided device is not a valid GPU device.

        Examples:
            >>> decoder = AbsDecoder()
            >>> decoder.set_device('cuda:0')

        Note:
            Make sure that the specified device is available and can be used
            for PyTorch operations. You can check available devices using
            `torch.cuda.is_available()`.
        """
        raise NotImplementedError

    @abstractmethod
    def init_state(self, batch_size: int) -> Union[
        List[Dict[str, torch.Tensor]],
        List[torch.Tensor],
        Tuple[torch.Tensor, Optional[torch.tensor]],
    ]:
        """
        Initialize decoder states.

        This method is responsible for creating the initial hidden states of the 
        decoder. It takes the batch size as input and returns a structure suitable 
        for holding the decoder's hidden states, which can vary based on the 
        implementation of the decoder.

        Attributes:
            batch_size (int): The size of the batch for which to initialize states.

        Args:
            batch_size: The number of sequences in the batch.

        Returns:
            A structure containing the initial decoder hidden states, which can be 
            one of the following types:
                - List[Dict[str, torch.Tensor]]: A list of dictionaries where each 
                dictionary contains tensor states for a single sequence.
                - List[torch.Tensor]: A list of tensor states for each sequence.
                - Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple where the first 
                element is a tensor of states, and the second element is an optional 
                tensor.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.

        Examples:
            Suppose you have a batch size of 32, you can initialize the states as follows:
            
            ```python
            decoder = YourDecoderSubclass()  # YourDecoderSubclass should inherit from AbsDecoder
            initial_states = decoder.init_state(batch_size=32)
            ```

        Note:
            The exact format of the returned states may depend on the specific 
            implementation of the decoder and should be documented in the subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def select_state(
        self,
        states: Union[
            List[Dict[str, torch.Tensor]],
            List[torch.Tensor],
            Tuple[torch.Tensor, Optional[torch.Tensor]],
        ],
        idx: int = 0,
    ) -> Union[
        List[Dict[str, torch.Tensor]],
        List[torch.Tensor],
        Tuple[torch.Tensor, Optional[torch.Tensor]],
    ]:
        """
        Get specified ID state from batch of states, if provided.

    This method retrieves a specific decoder hidden state from a batch of states 
    based on the provided index. If the index is out of range, the method should 
    handle it gracefully, typically by raising an appropriate error or returning 
    None.

    Args:
        states: Decoder hidden states, which can be a list of dictionaries, a 
            list of tensors, or a tuple containing a tensor and an optional 
            tensor.
        idx: State ID to extract (default is 0).

    Returns:
        Decoder hidden state for the given ID, which can be of the same type 
        as the input states.

    Raises:
        IndexError: If the provided index is out of range for the states.

    Examples:
        # Example with a list of dictionaries
        states = [{'hidden': torch.tensor([1.0, 2.0])}, {'hidden': torch.tensor([3.0, 4.0])}]
        selected_state = self.select_state(states, idx=1)
        # selected_state will be {'hidden': tensor([3.0, 4.0])}

        # Example with a list of tensors
        states = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        selected_state = self.select_state(states, idx=0)
        # selected_state will be tensor([1.0, 2.0])

        # Example with a tuple of tensors
        states = (torch.tensor([1.0, 2.0]), None)
        selected_state = self.select_state(states, idx=0)
        # selected_state will be tensor([1.0, 2.0])
        """
        raise NotImplementedError

    @abstractmethod
    def create_batch_states(
        self,
        new_states: List[
            Union[
                List[Dict[str, Optional[torch.Tensor]]],
                List[List[torch.Tensor]],
                Tuple[torch.Tensor, Optional[torch.Tensor]],
            ],
        ],
    ) -> Union[
        List[Dict[str, torch.Tensor]],
        List[torch.Tensor],
        Tuple[torch.Tensor, Optional[torch.Tensor]],
    ]:
        """
        Create batch of decoder hidden states given a list of new states.

    This method constructs a batch of decoder hidden states from a provided list 
    of new states. The new states can be in various formats, including a list of 
    dictionaries containing tensors or a list of tensor sequences. This is useful 
    for efficiently managing and updating decoder states during the decoding 
    process.

    Args:
        new_states: A list of new decoder hidden states, which can be one of the 
            following formats:
            - List of dictionaries, where each dictionary contains tensors 
              representing the hidden states.
            - List of lists, where each list contains tensors.
            - Tuple of a tensor and an optional tensor.

    Returns:
        A unified representation of the decoder hidden states, which can be one of 
        the following formats:
        - List of dictionaries containing tensors.
        - List of tensors.
        - Tuple of a tensor and an optional tensor.

    Raises:
        ValueError: If the format of new_states is not recognized or is invalid.

    Examples:
        # Example with a list of dictionaries
        new_states = [{'hidden': torch.randn(10, 256)}, {'hidden': torch.randn(10, 256)}]
        batch_states = decoder.create_batch_states(new_states)

        # Example with a list of tensors
        new_states = [torch.randn(10, 256), torch.randn(10, 256)]
        batch_states = decoder.create_batch_states(new_states)

        # Example with a tuple
        new_states = (torch.randn(10, 256), torch.randn(10, 256))
        batch_states = decoder.create_batch_states(new_states)
        """
        raise NotImplementedError
