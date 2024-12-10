from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch

from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract


class AbsTgtFeatsExtract(AbsFeatsExtract, ABC):
    """
        Abstract base class for target feature extraction in speech processing.

    This class serves as a blueprint for implementing various target feature
    extraction methods. It inherits from the `AbsFeatsExtract` class and defines
    the essential methods that must be implemented by any subclass.

    Attributes:
        None

    Methods:
        output_size: Returns the size of the output features.
        get_parameters: Returns a dictionary of parameters used in the feature
            extraction process.
        forward: Processes the input tensor and returns the extracted features
            along with their lengths.
        spectrogram: Indicates whether the output is a spectrogram.

    Args:
        input (torch.Tensor): The input tensor representing the audio features.
        input_lengths (torch.Tensor): The lengths of the input sequences.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the extracted
        features and their corresponding lengths.

    Raises:
        NotImplementedError: If the abstract methods are called without
        being implemented in a subclass.

    Examples:
        class MyTgtFeatsExtract(AbsTgtFeatsExtract):
            def output_size(self) -> int:
                return 128

            def get_parameters(self) -> Dict[str, Any]:
                return {"param1": 1, "param2": 2}

            def forward(
                self, input: torch.Tensor, input_lengths: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                # Example processing logic
                return input, input_lengths

            def spectrogram(self) -> bool:
                return True
    """

    @abstractmethod
    def output_size(self) -> int:
        """
                Abstract base class for target feature extraction.

        This class provides an interface for target feature extraction methods in
        text-to-speech systems. It defines the required methods for subclasses to
        implement, including methods for obtaining the output size, parameters,
        and processing input data.

        Attributes:
            None

        Args:
            None

        Returns:
            None

        Yields:
            None

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Examples:
            This class is intended to be subclassed. An example of a subclass could be:

            class MyTgtFeatsExtract(AbsTgtFeatsExtract):
                def output_size(self) -> int:
                    return 256

                def get_parameters(self) -> Dict[str, Any]:
                    return {'param1': 1, 'param2': 2}

                def forward(self, input: torch.Tensor, input_lengths: torch.Tensor) ->
                Tuple[torch.Tensor, torch.Tensor]:
                    # Implementation goes here
                    pass

                def spectrogram(self) -> bool:
                    return True
        """
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
            Retrieves the parameters of the target feature extractor.

        This method should be implemented by subclasses to provide the necessary
        parameters that define the behavior of the target feature extractor. The
        returned dictionary should contain relevant configurations and settings
        required for the feature extraction process.

        Returns:
            Dict[str, Any]: A dictionary containing the parameters of the feature
            extractor, where keys are parameter names and values are the corresponding
            parameter values.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Examples:
            class MyTgtFeatsExtract(AbsTgtFeatsExtract):
                def get_parameters(self) -> Dict[str, Any]:
                    return {
                        'sample_rate': 22050,
                        'n_fft': 2048,
                        'hop_length': 512,
                    }

            extractor = MyTgtFeatsExtract()
            params = extractor.get_parameters()
            print(params)
            # Output: {'sample_rate': 22050, 'n_fft': 2048, 'hop_length': 512}
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Computes the target features from the input tensor.

        This method is responsible for processing the input tensor and producing
        the corresponding output tensor along with the output lengths. The exact
        implementation of the feature extraction is to be defined in subclasses.

        Args:
            input (torch.Tensor): The input tensor of shape (B, T, F) where B is the
                batch size, T is the sequence length, and F is the number of features.
            input_lengths (torch.Tensor): A tensor of shape (B,) containing the actual
                lengths of each input sequence in the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - output (torch.Tensor): The output tensor of extracted features.
                - output_lengths (torch.Tensor): A tensor of shape (B,) containing
                  the lengths of each output sequence.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Examples:
            >>> model = MyTgtFeatsExtract()  # Assuming MyTgtFeatsExtract is a subclass
            >>> input_tensor = torch.randn(32, 100, 80)  # Example input
            >>> input_lengths = torch.tensor([100] * 32)  # All sequences are 100 long
            >>> output, output_lengths = model.forward(input_tensor, input_lengths)

        Note:
            This is an abstract method and must be implemented in subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def spectrogram(self) -> bool:
        """
                Abstract base class for target feature extraction in speech processing.

        This class provides an interface for extracting various target features
        from input audio data. It inherits from the `AbsFeatsExtract` class
        and defines several abstract methods that must be implemented by
        subclasses. These methods include functionality for determining the
        output size of features, retrieving parameters, performing the forward
        pass for feature extraction, and indicating whether the feature
        extraction includes a spectrogram.

        Attributes:
            None

        Args:
            None

        Returns:
            None

        Yields:
            None

        Raises:
            NotImplementedError: If any abstract method is not implemented in a
            subclass.

        Examples:
            To create a concrete implementation of this abstract class, you would
            subclass it and implement the abstract methods. For instance:

            class MyTgtFeatsExtract(AbsTgtFeatsExtract):
                def output_size(self) -> int:
                    return 128  # Example output size

                def get_parameters(self) -> Dict[str, Any]:
                    return {"param1": 1, "param2": 2}

                def forward(
                    self, input: torch.Tensor, input_lengths: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
                    # Implement the forward pass logic here
                    pass

                def spectrogram(self) -> bool:
                    return True  # Indicate that this feature extraction includes
                                  # a spectrogram

        Note:
            This class is intended to be subclassed. It should not be
            instantiated directly.

        Todo:
            Implement concrete classes for different feature extraction methods.
        """
        raise NotImplementedError
