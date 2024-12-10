from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch


class AbsFeatsExtract(torch.nn.Module, ABC):
    """
        Abstract base class for feature extraction modules in the ESPnet TTS framework.

    This class serves as a blueprint for creating feature extraction components.
    Subclasses must implement the methods defined here to provide specific
    functionality for extracting features from audio inputs.

    Attributes:
        None

    Methods:
        output_size() -> int:
            Returns the size of the output features.

        get_parameters() -> Dict[str, Any]:
            Returns the parameters of the feature extraction module.

        forward(input: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            Processes the input tensor and returns the extracted features along with
            the lengths of the input sequences.

    Raises:
        NotImplementedError:
            If a subclass does not implement any of the abstract methods.

    Examples:
        To create a concrete implementation of this class, subclass it and implement
        the abstract methods as follows:

        class MyFeatureExtractor(AbsFeatsExtract):
            def output_size(self) -> int:
                return 128  # Example output size

            def get_parameters(self) -> Dict[str, Any]:
                return {"param1": 1, "param2": 2}  # Example parameters

            def forward(self, input: torch.Tensor, input_lengths: torch.Tensor) ->
            Tuple[torch.Tensor, torch.Tensor]:
                # Example feature extraction logic
                features = input.mean(dim=1, keepdim=True)  # Simplified example
                return features, input_lengths
    """

    @abstractmethod
    def output_size(self) -> int:
        """
                Abstract base class for feature extraction modules.

        This class defines the interface for feature extraction modules, including
        methods to obtain the output size, parameters, and perform the forward
        pass. Subclasses must implement the abstract methods defined in this class.

        Attributes:
            None

        Methods:
            output_size: Returns the size of the output features.
            get_parameters: Returns the parameters of the model as a dictionary.
            forward: Performs the forward pass of the model on the input tensor.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Examples:
            class MyFeatureExtractor(AbsFeatsExtract):
                def output_size(self) -> int:
                    return 256

                def get_parameters(self) -> Dict[str, Any]:
                    return {"param1": 1, "param2": 2}

                def forward(self, input: torch.Tensor, input_lengths: torch.Tensor) ->
                    Tuple[torch.Tensor, torch.Tensor]:
                    # Your forward implementation here
                    pass
        """
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Retrieves the parameters of the feature extraction module.

        This method should be implemented by subclasses to return a dictionary
        containing the relevant parameters for the feature extraction process.
        The parameters may include configuration settings, hyperparameters, or
        any other relevant information needed to initialize or replicate the
        feature extraction behavior.

        Returns:
            A dictionary where the keys are parameter names (as strings) and
            the values are the corresponding parameter values (of any type).

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Examples:
            If a subclass implements this method, it might return a dictionary
            like the following:

            >>> extractor = MyFeatureExtractor()
            >>> params = extractor.get_parameters()
            >>> print(params)
            {'learning_rate': 0.001, 'num_layers': 3}
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Computes the forward pass of the feature extraction module.

        This method takes an input tensor and its corresponding lengths, and
        processes them to produce the extracted features along with their
        lengths.

        Args:
            input (torch.Tensor): A tensor containing the input data.
            input_lengths (torch.Tensor): A tensor containing the lengths of
                the input sequences.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - A tensor of extracted features.
                - A tensor of lengths for the extracted features.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Examples:
            >>> model = SomeConcreteFeatsExtractModel()
            >>> input_tensor = torch.randn(10, 20)  # Example input
            >>> lengths = torch.tensor([20] * 10)  # All sequences have length 20
            >>> features, lengths = model.forward(input_tensor, lengths)
            >>> print(features.shape)  # Expected output shape based on model design
        """
        raise NotImplementedError
