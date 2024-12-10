from abc import ABC, abstractmethod

import torch


class AbsDiffusion(torch.nn.Module, ABC):
    """
    Abstract base class for diffusion models used in audio enhancement.

This class defines the interface for diffusion models, providing methods for 
forward propagation and audio enhancement. Derived classes must implement the 
abstract methods defined in this class.

Attributes:
    None

Args:
    None

Methods:
    forward(input: torch.Tensor, ilens: torch.Tensor):
        Abstract method for performing the forward pass.
        
    enhance(input: torch.Tensor):
        Abstract method for enhancing the audio input.

Raises:
    NotImplementedError: If the derived class does not implement the abstract 
    methods.

Examples:
    class MyDiffusionModel(AbsDiffusion):
        def forward(self, input: torch.Tensor, ilens: torch.Tensor):
            # Implementation of the forward method
            pass
        
        def enhance(self, input: torch.Tensor):
            # Implementation of the enhance method
            pass

    model = MyDiffusionModel()
    output = model.forward(torch.randn(1, 16000), torch.tensor([16000]))
    enhanced_output = model.enhance(torch.randn(1, 16000))

Note:
    This class should not be instantiated directly.
    """
    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
    ):
        """
        Computes the forward pass of the diffusion model.

    This method takes an input tensor and its corresponding lengths, 
    processing them to generate the output of the diffusion model.

    Args:
        input (torch.Tensor): The input tensor containing data to be processed.
        ilens (torch.Tensor): A tensor representing the lengths of each input 
            sequence in the batch.

    Returns:
        torch.Tensor: The output tensor after applying the diffusion process 
            to the input.

    Raises:
        NotImplementedError: If the method is not implemented in a derived 
            class.

    Examples:
        >>> model = MyDiffusionModel()  # MyDiffusionModel is a subclass of AbsDiffusion
        >>> input_data = torch.randn(32, 10)  # Example input tensor
        >>> input_lengths = torch.tensor([10] * 32)  # Lengths of each sequence
        >>> output = model.forward(input_data, input_lengths)
        >>> print(output.shape)  # Expected output shape depends on the model

    Note:
        This method must be overridden in any subclass of AbsDiffusion.
        """
        raise NotImplementedError

    @abstractmethod
    def enhance(self, input: torch.Tensor):
        """
        Abstract base class for diffusion models in the AbsDiffusion package.

This class defines the interface for diffusion models, including the 
forward pass and an enhancement method. Subclasses must implement the 
abstract methods to provide specific functionalities.

Attributes:
    None

Args:
    input (torch.Tensor): The input tensor to the model.
    ilens (torch.Tensor): The lengths of the input sequences.

Methods:
    forward(input: torch.Tensor, ilens: torch.Tensor):
        Defines the forward pass of the model.
        
    enhance(input: torch.Tensor):
        Enhances the input tensor using the model's specific 
        enhancement method.

Raises:
    NotImplementedError: If the abstract methods are not implemented 
    in the subclass.

Examples:
    class MyDiffusionModel(AbsDiffusion):
        def forward(self, input, ilens):
            # Implementation of the forward method
            pass
        
        def enhance(self, input):
            # Implementation of the enhance method
            pass

    model = MyDiffusionModel()
    enhanced_output = model.enhance(torch.randn(1, 3, 64, 64))

Note:
    This class should not be instantiated directly.
        """
        raise NotImplementedError
