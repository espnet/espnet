from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch


class AbsESPnetModel(torch.nn.Module, ABC):
    """
    The common abstract class among each task.

    This class, `AbsESPnetModel`, is an abstract base class that inherits from
    `torch.nn.Module`. It serves as a blueprint for creating deep neural network
    models specific to various tasks within the ESPnet framework. It employs a
    delegate pattern to manage the forward pass of the model and defines key
    components such as "loss", "stats", and "weight" for the associated task.

    When implementing a new task in ESPnet, it is essential to inherit from this
    class. The interaction between the training system and your task class is
    mediated through the loss, stats, and weight values.

    Example:
        >>> from espnet2.tasks.abs_task import AbsTask
        >>> class YourESPnetModel(AbsESPnetModel):
        ...     def forward(self, input, input_lengths):
        ...         ...
        ...         return loss, stats, weight
        >>> class YourTask(AbsTask):
        ...     @classmethod
        ...     def build_model(cls, args: argparse.Namespace) -> YourESPnetModel:

    Attributes:
        None

    Args:
        None

    Returns:
        None

    Yields:
        None

    Raises:
        NotImplementedError: If the abstract methods are not implemented in a
        subclass.

    Note:
        This class is meant to be subclassed, and cannot be instantiated directly.
    """

    @abstractmethod
    def forward(
        self, **batch: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
            The forward method that defines the computation performed at every call.

        This method must be implemented by subclasses of AbsESPnetModel. It takes
        a variable number of tensor inputs and returns a tuple consisting of the
        computed loss, statistics, and weight for the task.

        Args:
            **batch: A variable number of keyword arguments containing tensors
                that represent the input data for the model.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
                - A tensor representing the computed loss.
                - A dictionary containing statistics relevant to the task.
                - A tensor representing the weight associated with the loss.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.

        Examples:
            >>> class YourESPnetModel(AbsESPnetModel):
            ...     def forward(self, input, input_lengths):
            ...         # Implement your forward logic here
            ...         return loss, stats, weight
        """
        raise NotImplementedError

    @abstractmethod
    def collect_feats(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
            Collect features from the given batch of input tensors.

        This method processes the input tensors and extracts relevant features
        that can be used for further computations or model training. The exact
        implementation of feature extraction will depend on the specific model
        derived from this abstract class.

        Args:
            **batch: Variable length keyword arguments representing input tensors.

        Returns:
            A dictionary containing the extracted features, where keys are
            feature names and values are the corresponding tensors.

        Raises:
            NotImplementedError: If the method is not implemented in the derived
            class.

        Examples:
            >>> model = YourESPnetModel()
            >>> features = model.collect_feats(input_tensor=input_data)
            >>> print(features.keys())  # Should print the feature names
        """
        raise NotImplementedError
