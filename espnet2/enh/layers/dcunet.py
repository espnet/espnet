# The implementation of DCUNet in
# S. Welker, et al.  “Speech Enhancement with Score-Based
# Generative Models in the Complex STFT Domain”
# The implementation is based on:
# https://github.com/sp-uhh/sgmse
# Licensed under MIT
#

import functools
from functools import partial

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm


class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.

    This module generates Gaussian random features for encoding time steps,
    which can be used in neural networks to capture temporal information.
    Depending on the `complex_valued` flag, it can produce either complex
    or real-valued outputs.

    Attributes:
        complex_valued (bool): Indicates whether the output should be complex
            valued.
        W (nn.Parameter): Fixed random weights sampled from a Gaussian
            distribution, scaled by a specified factor.

    Args:
        embed_dim (int): The dimension of the embedding space.
        scale (float, optional): Scaling factor for the weights. Default is 16.
        complex_valued (bool, optional): If True, the output will be complex
            valued. Default is False.

    Returns:
        Tensor: A tensor of shape `(batch_size, embed_dim)` containing the
        encoded time steps.

    Examples:
        >>> gfp = GaussianFourierProjection(embed_dim=128, scale=16)
        >>> time_steps = torch.tensor([[0.1], [0.2], [0.3]])
        >>> output = gfp(time_steps)
        >>> output.shape
        torch.Size([3, 128])  # For real-valued output
        >>> gfp_complex = GaussianFourierProjection(embed_dim=128, scale=16,
        ...                                          complex_valued=True)
        >>> output_complex = gfp_complex(time_steps)
        >>> output_complex.shape
        torch.Size([3, 128])  # For complex-valued output

    Note:
        The output for real-valued features consists of concatenated sine and
        cosine components, while for complex-valued features, it directly
        outputs complex exponentials.
    """

    def __init__(self, embed_dim, scale=16, complex_valued=False):
        super().__init__()
        self.complex_valued = complex_valued
        if not complex_valued:
            # If the output is real-valued, we concatenate sin+cos
            #  of the features to avoid ambiguities.
            # Therefore, in this case the effective embed_dim is
            # cut in half. For the complex-valued case,
            # we use complex numbers which each represent sin+cos
            # directly, so the ambiguity is avoided directly,
            # and this halving is not necessary.
            embed_dim = embed_dim // 2
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim) * scale, requires_grad=False)

    def forward(self, t):
        """
            Forward pass for the DCUNet model.

        This method processes the input complex spectrogram tensor through the
        network layers to produce an output tensor. The input is expected to
        be of shape `(batch, nfreqs, time)`, where `nfreqs - 1` must be divisible
        by the frequency strides of the encoders, and `time - 1` must be divisible
        by the time strides of the encoders.

        Args:
            spec (Tensor): A complex spectrogram tensor with shape
                `(batch, nfreqs, time)`. It can be a 1D, 2D, or 3D tensor,
                where the time dimension is expected to be the last dimension.

        Returns:
            Tensor: The output tensor, which has a shape of `(batch, time)`
                or `(time)` depending on the configuration of the model.

        Examples:
            >>> net = DCUNet()
            >>> dnn_input = torch.randn(4, 2, 257, 256) + 1j * torch.randn(4, 2, 257, 256)
            >>> output = net(dnn_input, torch.randn(4))
            >>> print(output.shape)
            torch.Size([4, 2, 257, 256])  # Example output shape, actual may vary.

        Note:
            Ensure that the input dimensions meet the requirements of the
            model configuration to avoid runtime errors.

        Raises:
            TypeError: If the input shape is not compatible with the expected
                dimensions or if the input is not divisible by the specified
                frequency or time products.
        """
        t_proj = t[:, None] * self.W[None, :] * 2 * np.pi
        if self.complex_valued:
            return torch.exp(1j * t_proj)
        else:
            return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


class DiffusionStepEmbedding(nn.Module):
    """
    Diffusion-Step embedding as in DiffWave / Vaswani et al. 2017.

    This class implements the diffusion-step embedding for a neural network.
    The embedding is based on sinusoidal functions which help in capturing the
    temporal structure of the input data. It supports both complex and real
    valued embeddings.

    Attributes:
        complex_valued (bool): Indicates if the embedding is complex-valued.
        embed_dim (int): The dimension of the embedding.

    Args:
        embed_dim (int): The dimension of the embedding.
        complex_valued (bool, optional): If True, the embedding will be
            complex-valued. Defaults to False.

    Returns:
        Tensor: The computed diffusion-step embedding.

    Examples:
        >>> embedding = DiffusionStepEmbedding(embed_dim=128)
        >>> t = torch.tensor([0.1, 0.2, 0.3])
        >>> output = embedding(t)
        >>> output.shape
        torch.Size([3, 128])  # For real-valued embedding

        >>> embedding_complex = DiffusionStepEmbedding(embed_dim=128, complex_valued=True)
        >>> output_complex = embedding_complex(t)
        >>> output_complex.shape
        torch.Size([3, 128])  # For complex-valued embedding

    Note:
        The effective embedding dimension is halved when the output is real-valued
        to avoid ambiguities.
    """

    def __init__(self, embed_dim, complex_valued=False):
        super().__init__()
        self.complex_valued = complex_valued
        if not complex_valued:
            # If the output is real-valued, we concatenate sin+cos of the features to
            # avoid ambiguities. Therefore, in this case the effective embed_dim is cut
            # in half. For the complex-valued case, we use complex numbers which each
            # represent sin+cos directly, so the ambiguity is avoided directly,
            # and this halving is not necessary.
            embed_dim = embed_dim // 2
        self.embed_dim = embed_dim

    def forward(self, t):
        """
            Forward pass for the DCUNet model.

        This method takes a complex spectrogram tensor and a time embedding
        tensor as input and processes them through the encoder-decoder
        architecture of the DCUNet model. The input shape is expected to be
        $(batch, nfreqs, time)$, where $nfreqs - 1$ is divisible by
        $f_0 * f_1 * ... * f_N$ (the frequency strides of the encoders)
        and $time - 1$ is divisible by $t_0 * t_1 * ... * t_N$ (the time
        strides of the encoders).

        Args:
            spec (Tensor): A complex spectrogram tensor with shape
                (batch, input_channels, n_freqs, time).
                The tensor can be 1D, 2D, or 3D with time as the last
                dimension.

            t (Tensor): A tensor representing the time step embeddings,
                typically a 1D tensor.

        Returns:
            Tensor: The output tensor, with shape (batch, time) or (time)
                depending on the model architecture and input.

        Examples:
            >>> net = DCUNet()
            >>> dnn_input = torch.randn(4, 2, 257, 256) + 1j * torch.randn(4, 2, 257, 256)
            >>> time_embedding = torch.randn(4)
            >>> output = net(dnn_input, time_embedding)
            >>> print(output.shape)  # Output shape: (4, 1, n_fft, frames)

        Raises:
            TypeError: If the input tensor does not conform to the required
                dimensions.
        """
        fac = 10 ** (
            4 * torch.arange(self.embed_dim, device=t.device) / (self.embed_dim - 1)
        )
        inner = t[:, None] * fac[None, :]
        if self.complex_valued:
            return torch.exp(1j * inner)
        else:
            return torch.cat([torch.sin(inner), torch.cos(inner)], dim=-1)


class ComplexLinear(nn.Module):
    """
    A potentially complex-valued linear layer.

    This layer can operate in a complex-valued space when
    `complex_valued=True`. If `complex_valued=False`, it behaves as a
    standard linear layer. The layer performs linear transformations
    on complex input tensors by separately processing the real and
    imaginary parts.

    Attributes:
        complex_valued (bool): Indicates if the layer operates in
            complex-valued space.
        re (nn.Linear): Linear layer for the real part.
        im (nn.Linear): Linear layer for the imaginary part.
        lin (nn.Linear): Linear layer when operating in real-valued
            space.

    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        complex_valued (bool): Flag to determine if the layer is
            complex-valued.

    Returns:
        Tensor: The transformed output tensor.

    Examples:
        >>> # Example for complex-valued operation
        >>> layer = ComplexLinear(input_dim=4, output_dim=2, complex_valued=True)
        >>> input_tensor = torch.randn(10, 4) + 1j * torch.randn(10, 4)
        >>> output_tensor = layer(input_tensor)
        >>> print(output_tensor.shape)  # Output: torch.Size([10, 2])

        >>> # Example for real-valued operation
        >>> layer = ComplexLinear(input_dim=4, output_dim=2, complex_valued=False)
        >>> input_tensor = torch.randn(10, 4)
        >>> output_tensor = layer(input_tensor)
        >>> print(output_tensor.shape)  # Output: torch.Size([10, 2])
    """

    def __init__(self, input_dim, output_dim, complex_valued):
        super().__init__()
        self.complex_valued = complex_valued
        if self.complex_valued:
            self.re = nn.Linear(input_dim, output_dim)
            self.im = nn.Linear(input_dim, output_dim)
        else:
            self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
            Perform the forward pass of the ComplexLinear layer.

        This method computes the output of the linear layer. If the layer is
        complex-valued, it applies the complex multiplication rules defined
        in the class. If it is not complex-valued, it simply performs the
        standard linear transformation.

        Args:
            x (Tensor): Input tensor, which can be complex-valued. It should
                have the shape (batch_size, input_dim) for real-valued or
                (batch_size, input_dim, 2) for complex-valued inputs.

        Returns:
            Tensor: The output tensor after applying the linear transformation.
                The output shape will be (batch_size, output_dim) for
                real-valued or (batch_size, output_dim, 2) for complex-valued
                outputs.

        Examples:
            >>> import torch
            >>> linear_layer = ComplexLinear(3, 2, complex_valued=True)
            >>> input_tensor = torch.randn(4, 3) + 1j * torch.randn(4, 3)
            >>> output = linear_layer(input_tensor)
            >>> print(output.shape)  # Should print torch.Size([4, 2])

            >>> linear_layer_real = ComplexLinear(3, 2, complex_valued=False)
            >>> input_tensor_real = torch.randn(4, 3)
            >>> output_real = linear_layer_real(input_tensor_real)
            >>> print(output_real.shape)  # Should print torch.Size([4, 2])
        """
        if self.complex_valued:
            return (self.re(x.real) - self.im(x.imag)) + 1j * (
                self.re(x.imag) + self.im(x.real)
            )
        else:
            return self.lin(x)


class FeatureMapDense(nn.Module):
    """
    A fully connected layer that reshapes outputs to feature maps.

    This layer is designed to be used in the context of complex-valued
    neural networks, where it applies a dense linear transformation to
    the input and reshapes the output to add two additional dimensions
    for feature maps. It utilizes the ComplexLinear layer to handle
    complex-valued inputs.

    Attributes:
        complex_valued (bool): Indicates if the layer processes complex-valued
            inputs.
        dense (ComplexLinear): The underlying dense layer that performs
            the linear transformation.

    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        complex_valued (bool): Whether the layer should support complex
            values. Defaults to False.

    Returns:
        Tensor: The reshaped output tensor with added dimensions for feature maps.

    Examples:
        >>> layer = FeatureMapDense(input_dim=128, output_dim=64, complex_valued=True)
        >>> input_tensor = torch.randn(10, 128) + 1j * torch.randn(10, 128)
        >>> output_tensor = layer(input_tensor)
        >>> output_tensor.shape
        torch.Size([10, 64, 1, 1])

    Note:
        This layer is primarily intended for use in complex-valued models,
        such as those used for audio signal processing or speech enhancement.
    """

    def __init__(self, input_dim, output_dim, complex_valued=False):
        super().__init__()
        self.complex_valued = complex_valued
        self.dense = ComplexLinear(input_dim, output_dim, complex_valued=complex_valued)

    def forward(self, x):
        """
            A fully connected layer that reshapes outputs to feature maps.

        This layer takes an input tensor and transforms it into a shape suitable
        for subsequent operations in a neural network. It is designed to handle
        both real and complex-valued inputs, depending on the configuration.

        Attributes:
            complex_valued (bool): Indicates whether the layer operates on complex
                numbers.
            dense (ComplexLinear): A linear layer that handles the transformation
                of input dimensions.

        Args:
            input_dim (int): The number of input features.
            output_dim (int): The number of output features.
            complex_valued (bool, optional): If True, the layer will handle complex
                inputs. Defaults to False.

        Returns:
            Tensor: The reshaped output tensor.

        Examples:
            >>> import torch
            >>> layer = FeatureMapDense(input_dim=128, output_dim=256)
            >>> input_tensor = torch.randn(10, 128)  # Batch of 10 samples
            >>> output_tensor = layer(input_tensor)
            >>> print(output_tensor.shape)  # Should print (10, 256, 1, 1)

        Note:
            The output tensor will have two additional dimensions (1, 1) added to
            facilitate reshaping into feature maps for subsequent layers.
        """
        return self.dense(x)[..., None, None]


def torch_complex_from_reim(re, im):
    """
    Create a complex tensor from real and imaginary parts.

    This function takes two tensors representing the real and imaginary
    components of a complex number and combines them into a single complex
    tensor using PyTorch's built-in complex number support.

    Args:
        re (Tensor): A tensor representing the real part of the complex number.
        im (Tensor): A tensor representing the imaginary part of the complex number.

    Returns:
        Tensor: A complex tensor formed by combining the real and imaginary parts.

    Examples:
        >>> re = torch.tensor([1.0, 2.0])
        >>> im = torch.tensor([3.0, 4.0])
        >>> complex_tensor = torch_complex_from_reim(re, im)
        >>> print(complex_tensor)
        tensor([1. + 3.j, 2. + 4.j])

    Note:
        The input tensors must have the same shape. If they do not, a runtime
        error will occur.

    Raises:
        ValueError: If the input tensors do not have the same shape.
    """
    return torch.view_as_complex(torch.stack([re, im], dim=-1))


class ArgsComplexMultiplicationWrapper(nn.Module):
    """
    Adapted from `asteroid`'s `complex_nn.py`, allowing

    args/kwargs to be passed through forward().

    Make a complex-valued module `F` from a real-valued module `f` by applying
    complex multiplication rules:

    F(a + i b) = f1(a) - f1(b) + i (f2(b) + f2(a))

    where `f1`, `f2` are instances of `f` that do *not* share weights.

    Args:
        module_cls (callable): A class or function that returns a Torch
            module/functional. Constructor of `f` in the formula above.
            Called 2x with `*args`, `**kwargs`, to construct the real and
            imaginary component modules.

    Examples:
        >>> from torch import nn
        >>> wrapper = ArgsComplexMultiplicationWrapper(nn.Linear, 10, 5)
        >>> input_tensor = torch.randn(2, 10) + 1j * torch.randn(2, 10)
        >>> output = wrapper(input_tensor)
        >>> print(output.shape)  # Output shape depends on the module used.
    """

    def __init__(self, module_cls, *args, **kwargs):
        super().__init__()
        self.re_module = module_cls(*args, **kwargs)
        self.im_module = module_cls(*args, **kwargs)

    def forward(self, x, *args, **kwargs):
        """
            Wraps a real-valued module to enable complex multiplication rules.

        This module adapts a real-valued function `f` into a complex-valued
        module `F` by applying complex multiplication rules defined as follows:

        F(a + i b) = f1(a) - f1(b) + i (f2(b) + f2(a))

        where `f1` and `f2` are instances of `f` that do not share weights.

        Args:
            module_cls (callable): A class or function that returns a Torch
                module/functional. Constructor of `f` in the formula above.
                Called twice with `*args`, `**kwargs` to construct the real and
                imaginary component modules.

        Attributes:
            re_module (nn.Module): Module for the real part of the complex input.
            im_module (nn.Module): Module for the imaginary part of the complex input.

        Examples:
            >>> import torch
            >>> from torch import nn
            >>> complex_module = ArgsComplexMultiplicationWrapper(nn.Linear, 10, 5)
            >>> input_tensor = torch.randn(2, 10) + 1j * torch.randn(2, 10)
            >>> output = complex_module(input_tensor)
            >>> print(output.shape)  # Should reflect the output shape of the module

        Raises:
            NotImplementedError: If the wrapped module does not support complex
                operations.
        """
        return torch_complex_from_reim(
            self.re_module(x.real, *args, **kwargs)
            - self.im_module(x.imag, *args, **kwargs),
            self.re_module(x.imag, *args, **kwargs)
            + self.im_module(x.real, *args, **kwargs),
        )


ComplexConv2d = functools.partial(ArgsComplexMultiplicationWrapper, nn.Conv2d)
ComplexConvTranspose2d = functools.partial(
    ArgsComplexMultiplicationWrapper, nn.ConvTranspose2d
)


def get_activation(name):
    """
    Get the activation function based on the provided name.

    This function returns the corresponding activation function class
    from PyTorch based on the input string. The following activation
    functions are supported:

    - 'silu': SiLU (Sigmoid Linear Unit)
    - 'relu': ReLU (Rectified Linear Unit)
    - 'leaky_relu': Leaky ReLU

    Args:
        name (str): The name of the activation function to retrieve.

    Returns:
        Callable: The corresponding activation function class.

    Raises:
        NotImplementedError: If the provided name does not match any
        supported activation function.

    Examples:
        >>> activation = get_activation("relu")
        >>> print(activation)  # Output: <class 'torch.nn.modules.activation.ReLU'>

        >>> activation = get_activation("silu")
        >>> print(activation)  # Output: <class 'torch.nn.modules.activation.SiLU'>

        >>> activation = get_activation("unknown")
        Traceback (most recent call last):
            ...
        NotImplementedError: Unknown activation: unknown
    """
    if name == "silu":
        return nn.SiLU
    elif name == "relu":
        return nn.ReLU
    elif name == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise NotImplementedError(f"Unknown activation: {name}")


class BatchNorm(_BatchNorm):
    """
    Batch Normalization layer for complex-valued inputs.

    This class extends the standard BatchNorm class to handle complex
    input tensors. It checks the input dimensions to ensure that the
    inputs are compatible with the expected dimensionality for batch
    normalization.

    Attributes:
        num_features (int): Number of features in the input.
        eps (float): A value added to the denominator for numerical stability.
        momentum (float): The value used for the running_mean and running_var
            computation.
        affine (bool): If True, this module has learnable parameters.
        track_running_stats (bool): If True, tracks running statistics.

    Args:
        num_features (int): Number of features in the input.
        eps (float, optional): A value added to the denominator for numerical
            stability (default: 1e-5).
        momentum (float, optional): The value used for the running_mean and
            running_var computation (default: 0.1).
        affine (bool, optional): If True, this module has learnable parameters
            (default: True).
        track_running_stats (bool, optional): If True, tracks running statistics
            (default: False).

    Raises:
        ValueError: If the input tensor dimensions are not 3D or 4D.

    Examples:
        >>> batch_norm = BatchNorm(num_features=64)
        >>> input_tensor = torch.randn(10, 64, 32, 32)  # 4D input
        >>> output_tensor = batch_norm(input_tensor)

    Note:
        The input tensor must have a shape of (N, C, H, W) or (N, C, L)
        where N is the batch size, C is the number of channels, H and W are
        the height and width of the input for 4D tensors, or L is the length
        for 3D tensors.
    """

    def _check_input_dim(self, input):
        if input.dim() < 2 or input.dim() > 4:
            raise ValueError(
                "expected 4D or 3D input (got {}D input)".format(input.dim())
            )


class OnReIm(nn.Module):
    """
    Module to apply a real-valued operation to the real and imaginary parts.

    This class allows for applying two different modules to the real and
    imaginary components of a complex-valued input tensor. It is useful for
    operations that need to be performed separately on the real and
    imaginary parts of complex numbers, and then combine the results back
    into a complex representation.

    Attributes:
        re_module (nn.Module): Module applied to the real part of the input.
        im_module (nn.Module): Module applied to the imaginary part of the input.

    Args:
        module_cls (callable): A class or function that returns a Torch
            module/functional. It is called twice with `*args`, `**kwargs`
            to create two separate modules for real and imaginary parts.
        *args: Variable length argument list passed to the module_cls.
        **kwargs: Arbitrary keyword arguments passed to the module_cls.

    Examples:
        >>> real_module = nn.Linear(10, 5)
        >>> imag_module = nn.Linear(10, 5)
        >>> model = OnReIm(lambda: real_module, lambda: imag_module)
        >>> complex_input = torch.randn(2, 10) + 1j * torch.randn(2, 10)
        >>> output = model(complex_input)
        >>> output.shape
        torch.Size([2, 5])

    Raises:
        ValueError: If the input tensor does not have a complex type.
    """

    def __init__(self, module_cls, *args, **kwargs):
        super().__init__()
        self.re_module = module_cls(*args, **kwargs)
        self.im_module = module_cls(*args, **kwargs)

    def forward(self, x):
        """
        DCUNet model for complex spectrogram processing.

        This implementation is based on the architecture described in
        the paper "Speech Enhancement with Score-Based Generative Models
        in the Complex STFT Domain". It leverages deep learning techniques
        for audio signal processing, specifically focusing on
        complex-valued input tensors.

        Args:
            dcunet_architecture (str): The architecture to use for the
                DCUNet model. Default is "DilDCUNet-v2".
            dcunet_time_embedding (str): The type of time embedding to use.
                Options are "gfp" for Gaussian Fourier Projection or "ds"
                for Diffusion Step embedding. Default is "gfp".
            dcunet_temb_layers_global (int): Number of global time embedding
                layers. Default is 2.
            dcunet_temb_layers_local (int): Number of local time embedding
                layers. Default is 1.
            dcunet_temb_activation (str): Activation function for time
                embedding layers. Default is "silu".
            dcunet_time_embedding_complex (bool): Whether to use complex
                time embedding. Default is False.
            dcunet_fix_length (str): Method to handle input length. Options
                are "pad" or "trim". Default is "pad".
            dcunet_mask_bound (str): Mask bounding option. Default is "none".
            dcunet_norm_type (str): Type of normalization to apply.
                Default is "bN".
            dcunet_activation (str): Activation function for the model.
                Default is "relu".
            embed_dim (int): Dimensionality of the embedding. Default is 128.

        Returns:
            None

        Examples:
            >>> model = DCUNet()
            >>> input_tensor = torch.randn(4, 2, 257, 256) + \
                1j * torch.randn(4, 2, 257, 256)
            >>> output = model(input_tensor, torch.randn(4))
            >>> print(output.shape)
            torch.Size([4, 1, 257, 256])

        Note:
            This model expects the input tensor shape to be
            (batch, nfreqs, time), where `nfreqs - 1` is divisible
            by the frequency strides of the encoders and
            `time - 1` is divisible by the time strides of the encoders.

        Raises:
            TypeError: If input tensor dimensions are incompatible.
            NotImplementedError: If an unsupported architecture or
                mask bounding is specified.
        """
        return torch_complex_from_reim(self.re_module(x.real), self.im_module(x.imag))


# Code for DCUNet largely copied from Danilo's `informedenh` repo, cheers!


def unet_decoder_args(encoders, *, skip_connections):
    """
    Get list of decoder arguments for upsampling (right) side of a symmetric u-net.

    This function generates the arguments needed to construct the decoder layers
    of a U-Net architecture based on the encoder parameters. The output can
    include skip connections if specified.

    Args:
        encoders (tuple of length `N` of tuples of
            (in_chan, out_chan, kernel_size, stride, padding)):
            List of arguments used to construct the encoders.
        skip_connections (bool): Whether to include skip connections in the
            calculation of decoder input channels.

    Returns:
        tuple of length `N` of tuples of
            (in_chan, out_chan, kernel_size, stride, padding):
            Arguments to be used to construct decoders.

    Examples:
        >>> encoder_args = (
        ...     (1, 32, (7, 5), (2, 2), "auto"),
        ...     (32, 64, (7, 5), (2, 2), "auto"),
        ... )
        >>> decoder_args = unet_decoder_args(encoder_args, skip_connections=True)
        >>> print(decoder_args)
        ((64, 32, (7, 5), (2, 2), 'auto'), (64, 1, (7, 5), (2, 2), 'auto'))

    Note:
        The decoder input channels are calculated based on the output channels
        of the corresponding encoder layers and the specified skip connection
        policy.
    """
    decoder_args = []
    for (
        enc_in_chan,
        enc_out_chan,
        enc_kernel_size,
        enc_stride,
        enc_padding,
        enc_dilation,
    ) in reversed(encoders):
        if skip_connections and decoder_args:
            skip_in_chan = enc_out_chan
        else:
            skip_in_chan = 0
        decoder_args.append(
            (
                enc_out_chan + skip_in_chan,
                enc_in_chan,
                enc_kernel_size,
                enc_stride,
                enc_padding,
                enc_dilation,
            )
        )
    return tuple(decoder_args)


def make_unet_encoder_decoder_args(encoder_args, decoder_args):
    """
    Construct encoder and decoder arguments for a U-Net architecture.

    This function processes the encoder arguments and prepares the decoder
    arguments based on the specified configurations. If `decoder_args` is set
    to "auto", the decoder arguments will be automatically derived from the
    encoder arguments using skip connections.

    Args:
        encoder_args (tuple): A tuple containing tuples of encoder parameters,
            where each inner tuple consists of:
                - in_chan (int): Number of input channels.
                - out_chan (int): Number of output channels.
                - kernel_size (tuple): Size of the convolutional kernel.
                - stride (tuple): Stride of the convolution.
                - padding (tuple or str): Padding applied to the input.
                - dilation (tuple): Dilation of the convolution.
        decoder_args (tuple or str): A tuple containing tuples of decoder
            parameters, similar to `encoder_args`, or "auto" to automatically
            derive decoder arguments.

    Returns:
        tuple: A tuple containing two tuples:
            - The first tuple contains processed encoder arguments.
            - The second tuple contains processed decoder arguments.

    Examples:
        >>> encoder_args = (
        ...     (1, 32, (7, 5), (2, 2), "auto", (1, 1)),
        ...     (32, 64, (7, 5), (2, 2), "auto", (1, 1)),
        ... )
        >>> decoder_args = "auto"
        >>> make_unet_encoder_decoder_args(encoder_args, decoder_args)
        (
            ((1, 32, (7, 5), (2, 2), (3, 2), (1, 1)), ...),
            ((32, 1, (7, 5), (2, 2), (3, 2), (1, 1)), ...)
        )

    Note:
        Padding can be specified as "auto" to automatically calculate the
        appropriate padding based on the kernel size.

    Raises:
        ValueError: If `encoder_args` or `decoder_args` are not in the
            expected format.
    """
    encoder_args = tuple(
        (
            in_chan,
            out_chan,
            tuple(kernel_size),
            tuple(stride),
            (
                tuple([n // 2 for n in kernel_size])
                if padding == "auto"
                else tuple(padding)
            ),
            tuple(dilation),
        )
        for in_chan, out_chan, kernel_size, stride, padding, dilation in encoder_args
    )

    if decoder_args == "auto":
        decoder_args = unet_decoder_args(
            encoder_args,
            skip_connections=True,
        )
    else:
        decoder_args = tuple(
            (
                in_ch,
                out_ch,
                tuple(ks),
                tuple(stride),
                tuple([n // 2 for n in ks]) if pad == "auto" else pad,
                tuple(dilation),
                out_pad,
            )
            for in_ch, out_ch, ks, stride, pad, dilation, out_pad in decoder_args
        )

    return encoder_args, decoder_args


DCUNET_ARCHITECTURES = {
    "DCUNet-10": make_unet_encoder_decoder_args(
        # Encoders:
        # (in_chan, out_chan, kernel_size, stride, padding, dilation)
        (
            (1, 32, (7, 5), (2, 2), "auto", (1, 1)),
            (32, 64, (7, 5), (2, 2), "auto", (1, 1)),
            (64, 64, (5, 3), (2, 2), "auto", (1, 1)),
            (64, 64, (5, 3), (2, 2), "auto", (1, 1)),
            (64, 64, (5, 3), (2, 1), "auto", (1, 1)),
        ),
        # Decoders: automatic inverse
        "auto",
    ),
    "DCUNet-16": make_unet_encoder_decoder_args(
        # Encoders:
        # (in_chan, out_chan, kernel_size, stride, padding, dilation)
        (
            (1, 32, (7, 5), (2, 2), "auto", (1, 1)),
            (32, 32, (7, 5), (2, 1), "auto", (1, 1)),
            (32, 64, (7, 5), (2, 2), "auto", (1, 1)),
            (64, 64, (5, 3), (2, 1), "auto", (1, 1)),
            (64, 64, (5, 3), (2, 2), "auto", (1, 1)),
            (64, 64, (5, 3), (2, 1), "auto", (1, 1)),
            (64, 64, (5, 3), (2, 2), "auto", (1, 1)),
            (64, 64, (5, 3), (2, 1), "auto", (1, 1)),
        ),
        # Decoders: automatic inverse
        "auto",
    ),
    "DCUNet-20": make_unet_encoder_decoder_args(
        # Encoders:
        # (in_chan, out_chan, kernel_size, stride, padding, dilation)
        (
            (1, 32, (7, 1), (1, 1), "auto", (1, 1)),
            (32, 32, (1, 7), (1, 1), "auto", (1, 1)),
            (32, 64, (7, 5), (2, 2), "auto", (1, 1)),
            (64, 64, (7, 5), (2, 1), "auto", (1, 1)),
            (64, 64, (5, 3), (2, 2), "auto", (1, 1)),
            (64, 64, (5, 3), (2, 1), "auto", (1, 1)),
            (64, 64, (5, 3), (2, 2), "auto", (1, 1)),
            (64, 64, (5, 3), (2, 1), "auto", (1, 1)),
            (64, 64, (5, 3), (2, 2), "auto", (1, 1)),
            (64, 90, (5, 3), (2, 1), "auto", (1, 1)),
        ),
        # Decoders: automatic inverse
        "auto",
    ),
    "DilDCUNet-v2": make_unet_encoder_decoder_args(
        # architecture used in SGMSE / Interspeech paper
        # Encoders:
        # (in_chan, out_chan, kernel_size, stride, padding, dilation)
        (
            (1, 32, (4, 4), (1, 1), "auto", (1, 1)),
            (32, 32, (4, 4), (1, 1), "auto", (1, 1)),
            (32, 32, (4, 4), (1, 1), "auto", (1, 1)),
            (32, 64, (4, 4), (2, 1), "auto", (2, 1)),
            (64, 128, (4, 4), (2, 2), "auto", (4, 1)),
            (128, 256, (4, 4), (2, 2), "auto", (8, 1)),
        ),
        # Decoders: automatic inverse
        "auto",
    ),
}


class DCUNet(nn.Module):
    """
    DCUNet model for speech enhancement.

    This class implements the DCUNet architecture as proposed in
    S. Welker et al., "Speech Enhancement with Score-Based
    Generative Models in the Complex STFT Domain". The architecture
    is designed for enhancing speech signals in the complex domain
    using deep learning techniques.

    Attributes:
        architecture (str): Name of the DCUNet architecture to use.
        fix_length_mode (str): Method to handle input length; can be 'pad',
            'trim', or None.
        norm_type (str): Type of normalization to apply; can be 'bN' or 'CbN'.
        activation (str): Activation function to use in the model.
        input_channels (int): Number of input channels, typically 2 for
            complex inputs.
        time_embedding (str): Type of time embedding to use, such as 'gfp'
            or 'ds'.
        time_embedding_complex (bool): Indicates if the time embedding is
            complex-valued.
        temb_layers_global (int): Number of global time embedding layers.
        temb_layers_local (int): Number of local time embedding layers.
        temb_activation (str): Activation function for time embedding layers.
        embed (nn.Sequential): Sequential container for time embedding layers.
        encoders (nn.ModuleList): List of encoder blocks.
        decoders (nn.ModuleList): List of decoder blocks.
        output_layer (nn.Module): Output layer of the network.

    Args:
        dcunet_architecture (str): The architecture of the DCUNet to use.
        dcunet_time_embedding (str): The type of time embedding to use.
        dcunet_temb_layers_global (int): Number of global time embedding layers.
        dcunet_temb_layers_local (int): Number of local time embedding layers.
        dcunet_temb_activation (str): Activation function for time embedding.
        dcunet_time_embedding_complex (bool): Whether to use complex time
            embedding.
        dcunet_fix_length (str): Method to handle input length ('pad',
            'trim', or 'none').
        dcunet_mask_bound (str): Mask bounding strategy ('none' or others).
        dcunet_norm_type (str): Normalization type ('bN' or 'CbN').
        dcunet_activation (str): Activation function for the network.
        embed_dim (int): Dimension of the embedding layer.
        **kwargs: Additional keyword arguments for customization.

    Returns:
        Tensor: Output tensor of the model.

    Examples:
        >>> net = DCUNet()
        >>> dnn_input = torch.randn(4, 2, 257, 256) + 1j * torch.randn(4, 2, 257, 256)
        >>> score = net(dnn_input, torch.randn(4))
        >>> print(score.shape)
        torch.Size([4, 1, n_fft, frames])

    Note:
        Input shape is expected to be (batch, nfreqs, time), where
        nfreqs - 1 is divisible by the frequency strides of the encoders,
        and time - 1 is divisible by the time strides of the encoders.

    Raises:
        NotImplementedError: If the mask bounding method is not implemented.
    """

    def __init__(
        self,
        dcunet_architecture: str = "DilDCUNet-v2",
        dcunet_time_embedding: str = "gfp",
        dcunet_temb_layers_global: int = 2,
        dcunet_temb_layers_local: int = 1,
        dcunet_temb_activation: str = "silu",
        dcunet_time_embedding_complex: bool = False,
        dcunet_fix_length: str = "pad",
        dcunet_mask_bound: str = "none",
        dcunet_norm_type: str = "bN",
        dcunet_activation: str = "relu",
        embed_dim: int = 128,
        **kwargs,
    ):
        super().__init__()

        self.architecture = dcunet_architecture
        self.fix_length_mode = (
            dcunet_fix_length if dcunet_fix_length != "none" else None
        )
        self.norm_type = dcunet_norm_type
        self.activation = dcunet_activation
        self.input_channels = 2
        # for x_t and y -- note that this is 2 rather than 4,
        # because we directly treat complex channels in this DNN
        self.time_embedding = (
            dcunet_time_embedding if dcunet_time_embedding != "none" else None
        )
        self.time_embedding_complex = dcunet_time_embedding_complex
        self.temb_layers_global = dcunet_temb_layers_global
        self.temb_layers_local = dcunet_temb_layers_local
        self.temb_activation = dcunet_temb_activation
        conf_encoders, conf_decoders = DCUNET_ARCHITECTURES[dcunet_architecture]

        # Replace `input_channels` in encoders config
        _replaced_input_channels, *rest = conf_encoders[0]
        encoders = ((self.input_channels, *rest), *conf_encoders[1:])
        decoders = conf_decoders
        self.encoders_stride_product = np.prod(
            [enc_stride for _, _, _, enc_stride, _, _ in encoders], axis=0
        )

        # Prepare kwargs for encoder and decoder
        # (to potentially be modified before layer instantiation)
        encoder_decoder_kwargs = dict(
            norm_type=self.norm_type,
            activation=self.activation,
            temb_layers=self.temb_layers_local,
            temb_activation=self.temb_activation,
        )

        # Instantiate (global) time embedding layer
        embed_ops = []
        if self.time_embedding is not None:
            complex_valued = self.time_embedding_complex
            if self.time_embedding == "gfp":
                embed_ops += [
                    GaussianFourierProjection(
                        embed_dim=embed_dim, complex_valued=complex_valued
                    )
                ]
                encoder_decoder_kwargs["embed_dim"] = embed_dim
            elif self.time_embedding == "ds":
                embed_ops += [
                    DiffusionStepEmbedding(
                        embed_dim=embed_dim, complex_valued=complex_valued
                    )
                ]
                encoder_decoder_kwargs["embed_dim"] = embed_dim

            if self.time_embedding_complex:
                assert self.time_embedding in (
                    "gfp",
                    "ds",
                ), "Complex timestep embedding only available for gfp and ds"
                encoder_decoder_kwargs["complex_time_embedding"] = True
            for _ in range(self.temb_layers_global):
                embed_ops += [
                    ComplexLinear(embed_dim, embed_dim, complex_valued=True),
                    OnReIm(get_activation(dcunet_temb_activation)),
                ]
        self.embed = nn.Sequential(*embed_ops)

        # Instantiate DCUNet layers #
        output_layer = ComplexConvTranspose2d(*decoders[-1])
        encoders = [
            DCUNetComplexEncoderBlock(*args, **encoder_decoder_kwargs)
            for args in encoders
        ]
        decoders = [
            DCUNetComplexDecoderBlock(*args, **encoder_decoder_kwargs)
            for args in decoders[:-1]
        ]

        self.mask_bound = dcunet_mask_bound if dcunet_mask_bound != "none" else None
        if self.mask_bound is not None:
            raise NotImplementedError(
                "sorry, mask bounding not implemented at the moment"
            )
        # TODO(gituser) we can't use nn.Sequential since the ComplexConvTranspose2d
        # needs a second `output_size` argument
        # operations = (output_layer, complex_nn.BoundComplexMask(self.mask_bound))
        # output_layer = nn.Sequential(*[x for x in operations if x is not None])

        assert len(encoders) == len(decoders) + 1
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.output_layer = output_layer or nn.Identity()

    def forward(self, spec, t) -> Tensor:
        """
            Process input through the DCUNet architecture.

        The input shape is expected to be $(batch, nfreqs, time)$, where $nfreqs - 1$
        must be divisible by the product of frequency strides from the encoders,
        and $time - 1$ must be divisible by the product of time strides from the
        encoders.

        Args:
            spec (Tensor): A complex spectrogram tensor. This can be a 1D, 2D, or
                3D tensor, with the time dimension last.

        Returns:
            Tensor: The output tensor, which has a shape of (batch, time) or (time).

        Examples:
            >>> net = DCUNet()
            >>> dnn_input = torch.randn(4, 2, 257, 256) + 1j * torch.randn(4, 2, 257, 256)
            >>> score = net(dnn_input, torch.randn(4))
            >>> print(score.shape)
            torch.Size([4, 2, 257, 256])

        Note:
            The input tensor must have a valid shape that complies with the model's
            stride requirements. If the shape is invalid, a TypeError will be raised.
        """
        # TF-rep shape: (batch, self.input_channels, n_fft, frames)
        # Estimate mask from time-frequency representation.
        x_in = self.fix_input_dims(spec)
        x = x_in
        t_embed = self.embed(t + 0j) if self.time_embedding is not None else None

        enc_outs = []
        for idx, enc in enumerate(self.encoders):
            x = enc(x, t_embed)
            # UNet skip connection
            enc_outs.append(x)
        for enc_out, dec in zip(reversed(enc_outs[:-1]), self.decoders):
            x = dec(x, t_embed, output_size=enc_out.shape)
            x = torch.cat([x, enc_out], dim=1)

        output = self.output_layer(x, output_size=x_in.shape)
        # output shape: (batch, 1, n_fft, frames)
        output = self.fix_output_dims(output, spec)
        return output

    def fix_input_dims(self, x):
        """
            Adjust the input dimensions to be compatible with DCUNet.

        This method pads or trims the input tensor `x` to ensure its shape
        is compatible with the architecture of DCUNet. Specifically, it checks
        that the frequency and time dimensions are divisible by the stride
        products of the encoder layers. The method operates based on the
        `fix_length_mode` attribute, which determines whether to pad or trim
        the input tensor.

        Args:
            x (Tensor): The input tensor of shape (batch, channels, freq, time).

        Returns:
            Tensor: The adjusted input tensor with compatible dimensions.

        Raises:
            TypeError: If the input shape is not compatible with the expected
                frequency and time dimensions.
            ValueError: If an unknown `fix_length_mode` is specified.

        Examples:
            >>> import torch
            >>> dcu_net = DCUNet()
            >>> input_tensor = torch.randn(4, 2, 258, 256)  # Example shape
            >>> adjusted_tensor = dcu_net.fix_input_dims(input_tensor)
            >>> print(adjusted_tensor.shape)  # Output shape will be adjusted
        """
        return _fix_dcu_input_dims(
            self.fix_length_mode, x, torch.from_numpy(self.encoders_stride_product)
        )

    def fix_output_dims(self, out, x):
        """
            Adjusts the output dimensions to match the original input shape.

        This method fixes the shape of the output tensor `out` to the original
        shape of the input tensor `x` by padding or cropping, based on the
        specified length mode. It ensures that the output dimensions are
        compatible with the expected output shape for further processing.

        Args:
            out (Tensor): The output tensor from the model. It is expected to be
                a tensor with dimensions that may differ from the input tensor.
            x (Tensor): The original input tensor whose shape will be used to
                adjust the output tensor.

        Returns:
            Tensor: The adjusted output tensor, with its shape modified to match
            the original input tensor's shape.

        Note:
            The method uses padding if the output is shorter than the input and
            crops the output if it is longer. The specific behavior is controlled
            by the `fix_length_mode` attribute of the class.

        Examples:
            >>> input_tensor = torch.randn(4, 2, 257, 256)
            >>> output_tensor = torch.randn(4, 2, 255, 256)
            >>> fixed_output = self.fix_output_dims(output_tensor, input_tensor)
            >>> fixed_output.shape
            torch.Size([4, 2, 257, 256])  # Shape matches the input tensor
        """
        return _fix_dcu_output_dims(self.fix_length_mode, out, x)


def _fix_dcu_input_dims(fix_length_mode, x, encoders_stride_product):
    """Pad or trim `x` to a length compatible with DCUNet."""
    freq_prod = int(encoders_stride_product[0])
    time_prod = int(encoders_stride_product[1])
    if (x.shape[2] - 1) % freq_prod:
        raise TypeError(
            f"Input shape must be [batch, ch, freq + 1, time + 1] "
            f"with freq divisible by "
            f"{freq_prod}, got {x.shape} instead"
        )
    time_remainder = (x.shape[3] - 1) % time_prod
    if time_remainder:
        if fix_length_mode is None:
            raise TypeError(
                f"Input shape must be [batch, ch, freq + 1, time + 1] with time "
                f"divisible by {time_prod}, got {x.shape} instead."
                f" Set the 'fix_length_mode' argument "
                f"in 'DCUNet' to 'pad' or 'trim' to fix shapes automatically."
            )
        elif fix_length_mode == "pad":
            pad_shape = [0, time_prod - time_remainder]
            x = nn.functional.pad(x, pad_shape, mode="constant")
        elif fix_length_mode == "trim":
            pad_shape = [0, -time_remainder]
            x = nn.functional.pad(x, pad_shape, mode="constant")
        else:
            raise ValueError(f"Unknown fix_length mode '{fix_length_mode}'")
    return x


def _fix_dcu_output_dims(fix_length_mode, out, x):
    """Fix shape of `out` to the original shape of `x` by padding/cropping."""
    inp_len = x.shape[-1]
    output_len = out.shape[-1]
    return nn.functional.pad(out, [0, inp_len - output_len])


def _get_norm(norm_type):
    if norm_type == "CbN":
        return ComplexBatchNorm
    elif norm_type == "bN":
        return partial(OnReIm, BatchNorm)
    else:
        raise NotImplementedError(f"Unknown norm type: {norm_type}")


class DCUNetComplexEncoderBlock(nn.Module):
    """
    DCUNet Complex Encoder Block.

    This block is a key component of the DCUNet architecture, which is used
    for speech enhancement tasks. It performs complex-valued convolutional
    operations, normalization, and activation functions. The encoder block
    processes the input features and incorporates time embedding if specified.

    Attributes:
        in_chan (int): Number of input channels.
        out_chan (int): Number of output channels.
        kernel_size (tuple): Size of the convolutional kernel.
        stride (tuple): Stride of the convolution.
        padding (tuple): Padding applied to the input.
        dilation (tuple): Dilation applied to the convolution.
        temb_layers (int): Number of time embedding layers.
        temb_activation (str): Activation function for time embedding.
        complex_time_embedding (bool): Whether to use complex time embedding.
        conv (ComplexConv2d): Complex convolutional layer.
        norm (nn.Module): Normalization layer.
        activation (nn.Module): Activation function.
        embed_dim (int): Dimension of the embedding space.
        embed_layer (nn.Sequential): Sequential layer for embedding.

    Args:
        in_chan (int): Number of input channels.
        out_chan (int): Number of output channels.
        kernel_size (tuple): Size of the convolutional kernel.
        stride (tuple): Stride of the convolution.
        padding (tuple): Padding applied to the input.
        dilation (tuple): Dilation applied to the convolution.
        norm_type (str): Type of normalization to use (default: "bN").
        activation (str): Activation function to use (default: "leaky_relu").
        embed_dim (int, optional): Dimension of the embedding space.
        complex_time_embedding (bool, optional): Whether to use complex
            time embedding (default: False).
        temb_layers (int, optional): Number of time embedding layers
            (default: 1).
        temb_activation (str, optional): Activation function for
            time embedding (default: "silu").

    Returns:
        None

    Examples:
        >>> encoder_block = DCUNetComplexEncoderBlock(1, 32, (3, 3), (1, 1),
        ...                                             (1, 1), (1, 1))
        >>> input_tensor = torch.randn(4, 1, 64, 64) + 1j * torch.randn(4, 1, 64, 64)
        >>> output = encoder_block(input_tensor, None)
        >>> print(output.shape)
        torch.Size([4, 32, 64, 64])

    Note:
        The input tensor should be complex-valued and have the shape
        (batch_size, channels, height, width).
    """

    def __init__(
        self,
        in_chan,
        out_chan,
        kernel_size,
        stride,
        padding,
        dilation,
        norm_type="bN",
        activation="leaky_relu",
        embed_dim=None,
        complex_time_embedding=False,
        temb_layers=1,
        temb_activation="silu",
    ):
        super().__init__()

        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.temb_layers = temb_layers
        self.temb_activation = temb_activation
        self.complex_time_embedding = complex_time_embedding

        self.conv = ComplexConv2d(
            in_chan,
            out_chan,
            kernel_size,
            stride,
            padding,
            bias=norm_type is None,
            dilation=dilation,
        )
        self.norm = _get_norm(norm_type)(out_chan)
        self.activation = OnReIm(get_activation(activation))
        self.embed_dim = embed_dim
        if self.embed_dim is not None:
            ops = []
            for _ in range(max(0, self.temb_layers - 1)):
                ops += [
                    ComplexLinear(self.embed_dim, self.embed_dim, complex_valued=True),
                    OnReIm(get_activation(self.temb_activation)),
                ]
            ops += [
                FeatureMapDense(self.embed_dim, self.out_chan, complex_valued=True),
                OnReIm(get_activation(self.temb_activation)),
            ]
            self.embed_layer = nn.Sequential(*ops)

    def forward(self, x, t_embed):
        """
            Processes the input complex spectrogram and time embedding through the
        DCUNet architecture.

        The input shape is expected to be $(batch, nfreqs, time)$, where $nfreqs - 1$
        is divisible by $f_0 * f_1 * ... * f_N$ where $f_k$ are the frequency strides
        of the encoders, and $time - 1$ is divisible by $t_0 * t_1 * ... * t_N$
        where $t_N$ are the time strides of the encoders.

        Args:
            spec (Tensor): Complex spectrogram tensor. It can be a 1D, 2D, or 3D
                tensor, with time being the last dimension.
            t (Tensor): Time embedding tensor to provide additional context for the
                processing.

        Returns:
            Tensor: Output tensor, of shape (batch, time) or (time) after processing
                through the network.

        Examples:
            >>> net = DCUNet()
            >>> dnn_input = torch.randn(4, 2, 257, 256) + 1j * torch.randn(4, 2, 257, 256)
            >>> time_embedding = torch.randn(4)
            >>> output = net(dnn_input, time_embedding)
            >>> print(output.shape)
            torch.Size([4, 1, n_fft, frames])  # Shape depends on input dimensions.

        Note:
            The input tensor `spec` should have the last dimension representing the time
            frames and the preceding dimensions corresponding to the batch size and
            frequency channels.
        """
        y = self.conv(x)
        if self.embed_dim is not None:
            y = y + self.embed_layer(t_embed)
        return self.activation(self.norm(y))


class DCUNetComplexDecoderBlock(nn.Module):
    """
    A complex-valued decoder block for the DCUNet architecture.

    This block performs upsampling using transposed convolutions and includes
    normalization and activation functions. It can also integrate time
    embeddings to enhance the network's capacity to learn from temporal
    features.

    Attributes:
        in_chan (int): Number of input channels.
        out_chan (int): Number of output channels.
        kernel_size (tuple): Size of the convolution kernel.
        stride (tuple): Stride of the convolution.
        padding (tuple): Padding added to both sides of the input.
        dilation (tuple): Dilation rate for the convolution.
        output_padding (tuple): Additional size added to the output.
        complex_time_embedding (bool): Flag indicating if complex time
            embedding is used.
        temb_layers (int): Number of layers for the time embedding.
        temb_activation (str): Activation function for the time embedding.
        embed_dim (int, optional): Dimension of the embedding.
        deconv (nn.Module): The transposed convolution layer.
        norm (nn.Module): Normalization layer.
        activation (nn.Module): Activation layer.
        embed_layer (nn.Sequential, optional): Layer for processing time
            embeddings.

    Args:
        in_chan (int): Number of input channels.
        out_chan (int): Number of output channels.
        kernel_size (tuple): Size of the convolution kernel.
        stride (tuple): Stride of the convolution.
        padding (tuple): Padding added to both sides of the input.
        dilation (tuple): Dilation rate for the convolution.
        output_padding (tuple, optional): Additional size added to the output.
        norm_type (str, optional): Type of normalization to use. Default is "bN".
        activation (str, optional): Activation function to use. Default is
            "leaky_relu".
        embed_dim (int, optional): Dimension of the embedding. Default is None.
        temb_layers (int, optional): Number of layers for the time embedding.
            Default is 1.
        temb_activation (str, optional): Activation function for the time
            embedding. Default is "swish".
        complex_time_embedding (bool, optional): Flag indicating if complex
            time embedding is used. Default is False.

    Examples:
        >>> decoder_block = DCUNetComplexDecoderBlock(
        ...     in_chan=64,
        ...     out_chan=32,
        ...     kernel_size=(3, 3),
        ...     stride=(2, 2),
        ...     padding=(1, 1)
        ... )
        >>> input_tensor = torch.randn(4, 64, 128, 128) + 1j * torch.randn(4, 64, 128, 128)
        >>> output_tensor = decoder_block(input_tensor, t_embed=None)
        >>> print(output_tensor.shape)
        torch.Size([4, 32, 256, 256])

    Note:
        The input tensor should be a complex-valued tensor where the real
        and imaginary parts are represented separately.

    Raises:
        ValueError: If the input tensor dimensions do not match the expected
        shape or if the specified normalization type is not supported.
    """

    def __init__(
        self,
        in_chan,
        out_chan,
        kernel_size,
        stride,
        padding,
        dilation,
        output_padding=(0, 0),
        norm_type="bN",
        activation="leaky_relu",
        embed_dim=None,
        temb_layers=1,
        temb_activation="swish",
        complex_time_embedding=False,
    ):
        super().__init__()

        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.complex_time_embedding = complex_time_embedding
        self.temb_layers = temb_layers
        self.temb_activation = temb_activation

        self.deconv = ComplexConvTranspose2d(
            in_chan,
            out_chan,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation=dilation,
            bias=norm_type is None,
        )
        self.norm = _get_norm(norm_type)(out_chan)
        self.activation = OnReIm(get_activation(activation))
        self.embed_dim = embed_dim
        if self.embed_dim is not None:
            ops = []
            for _ in range(max(0, self.temb_layers - 1)):
                ops += [
                    ComplexLinear(self.embed_dim, self.embed_dim, complex_valued=True),
                    OnReIm(get_activation(self.temb_activation)),
                ]
            ops += [
                FeatureMapDense(self.embed_dim, self.out_chan, complex_valued=True),
                OnReIm(get_activation(self.temb_activation)),
            ]
            self.embed_layer = nn.Sequential(*ops)

    def forward(self, x, t_embed, output_size=None):
        """
            Performs the forward pass of the DCUNetComplexDecoderBlock.

        This method takes an input tensor `x` and a time embedding tensor
        `t_embed`, processes them through a series of operations including
        complex transposed convolution, normalization, and activation, and
        returns the resulting tensor.

        Args:
            x (Tensor): Input tensor, expected shape is (batch, in_chan,
                height, width) where in_chan is the number of input channels.
            t_embed (Tensor): Time embedding tensor, shape is expected to
                match the embedding dimensions used during initialization.
            output_size (tuple, optional): If provided, specifies the target
                output size for the transposed convolution. The shape should
                be (batch, out_chan, target_height, target_width). Defaults to
                None, in which case the output size is determined automatically.

        Returns:
            Tensor: Output tensor after applying the decoder block, shape will
            be (batch, out_chan, height, width) or adjusted to match the
            provided output_size.

        Examples:
            >>> decoder_block = DCUNetComplexDecoderBlock(
            ...     in_chan=64, out_chan=32, kernel_size=(3, 3), stride=(2, 2),
            ...     padding=(1, 1), output_padding=(1, 1))
            >>> x = torch.randn(4, 64, 16, 16)  # Example input tensor
            >>> t_embed = torch.randn(4, 128)   # Example time embedding
            >>> output = decoder_block(x, t_embed, output_size=(4, 32, 33, 33))
            >>> print(output.shape)
            torch.Size([4, 32, 33, 33])
        """
        y = self.deconv(x, output_size=output_size)
        if self.embed_dim is not None:
            y = y + self.embed_layer(t_embed)
        return self.activation(self.norm(y))


# From https://github.com/chanil1218/DCUnet.pytorch/blob/
# 2dcdd30804be47a866fde6435cbb7e2f81585213/models/layers/complexnn.py
class ComplexBatchNorm(torch.nn.Module):
    """
    Complex Batch Normalization layer.

    This layer normalizes complex-valued inputs by applying batch normalization
    independently to the real and imaginary parts. It can be used to improve
    training stability and convergence in deep learning models that handle
    complex data.

    Attributes:
        num_features (int): Number of features (channels) in the input.
        eps (float): A small value added for numerical stability during
            division.
        momentum (float): The momentum for the moving average of the
            running statistics.
        affine (bool): If True, this layer has learnable affine parameters.
        track_running_stats (bool): If True, this layer tracks running
            statistics (mean and variance) during training.

    Args:
        num_features (int): Number of features (channels) in the input.
        eps (float, optional): A small value added for numerical stability
            (default: 1e-5).
        momentum (float, optional): Momentum for running statistics
            (default: 0.1).
        affine (bool, optional): If True, add learnable parameters to the
            layer (default: True).
        track_running_stats (bool, optional): If True, track running
            statistics (default: False).

    Raises:
        AssertionError: If the dimensions of the input tensors are not as
            expected.

    Examples:
        >>> batch_norm = ComplexBatchNorm(num_features=64)
        >>> input_tensor = torch.randn(32, 64, 128) + 1j * torch.randn(32, 64, 128)
        >>> output_tensor = batch_norm(input_tensor)

    Note:
        The layer expects the input to be a complex tensor with separate
        real and imaginary parts. The output will also be a complex tensor.
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=False,
    ):
        super(ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.Wrr = torch.nn.Parameter(torch.Tensor(num_features))
            self.Wri = torch.nn.Parameter(torch.Tensor(num_features))
            self.Wii = torch.nn.Parameter(torch.Tensor(num_features))
            self.Br = torch.nn.Parameter(torch.Tensor(num_features))
            self.Bi = torch.nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter("Wrr", None)
            self.register_parameter("Wri", None)
            self.register_parameter("Wii", None)
            self.register_parameter("Br", None)
            self.register_parameter("Bi", None)
        if self.track_running_stats:
            self.register_buffer("RMr", torch.zeros(num_features))
            self.register_buffer("RMi", torch.zeros(num_features))
            self.register_buffer("RVrr", torch.ones(num_features))
            self.register_buffer("RVri", torch.zeros(num_features))
            self.register_buffer("RVii", torch.ones(num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("RMr", None)
            self.register_parameter("RMi", None)
            self.register_parameter("RVrr", None)
            self.register_parameter("RVri", None)
            self.register_parameter("RVii", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        """
            Applies Batch Normalization over complex-valued inputs.

        This class implements batch normalization for complex-valued
        inputs, allowing for affine transformations. The forward method
        normalizes the input using batch statistics during training and
        running statistics during evaluation.

        Attributes:
            num_features (int): Number of features (channels) in the input.
            eps (float): A small value added for numerical stability.
            momentum (float): Momentum for the running statistics.
            affine (bool): Whether to include learnable parameters.
            track_running_stats (bool): Whether to track running statistics.
            Wrr (torch.nn.Parameter): Weight for the real-real part.
            Wri (torch.nn.Parameter): Weight for the real-imaginary part.
            Wii (torch.nn.Parameter): Weight for the imaginary-imaginary part.
            Br (torch.nn.Parameter): Bias for the real part.
            Bi (torch.nn.Parameter): Bias for the imaginary part.
            RMr (torch.Tensor): Running mean for the real part.
            RMi (torch.Tensor): Running mean for the imaginary part.
            RVrr (torch.Tensor): Running variance for the real-real part.
            RVri (torch.Tensor): Running variance for the real-imaginary part.
            RVii (torch.Tensor): Running variance for the imaginary-imaginary part.
            num_batches_tracked (torch.Tensor): Number of batches processed.

        Args:
            num_features (int): Number of features (channels) in the input.
            eps (float): A small value added for numerical stability (default: 1e-5).
            momentum (float): Momentum for the running statistics (default: 0.1).
            affine (bool): Whether to include learnable parameters (default: True).
            track_running_stats (bool): Whether to track running statistics (default: False).

        Examples:
            >>> import torch
            >>> layer = ComplexBatchNorm(num_features=3)
            >>> input_tensor = torch.randn(10, 3, 5) + 1j * torch.randn(10, 3, 5)
            >>> output = layer(input_tensor)

        Note:
            The input tensor should have a shape of (batch_size, num_features, ...).

        Raises:
            ValueError: If the input dimensions are incorrect.

        Methods:
            reset_running_stats: Resets the running mean and variance.
            reset_parameters: Resets learnable parameters to their initial state.
        """
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        """
            Applies Batch Normalization over a complex input.

        This layer normalizes the input across the batch dimension and
        is designed specifically for complex-valued inputs. The
        normalization is performed separately for the real and imaginary
        parts of the input.

        Attributes:
            num_features (int): Number of features or channels.
            eps (float): A value added to the denominator for numerical stability.
            momentum (float): Momentum for the moving average.
            affine (bool): If True, this layer has learnable parameters.
            track_running_stats (bool): If True, this layer tracks the running
                mean and variance.

        Args:
            num_features (int): Number of features or channels in the input.
            eps (float, optional): Default is 1e-5.
            momentum (float, optional): Default is 0.1.
            affine (bool, optional): Default is True.
            track_running_stats (bool, optional): Default is False.

        Examples:
            >>> layer = ComplexBatchNorm(num_features=32)
            >>> input_tensor = torch.randn(64, 32, 10) + 1j * torch.randn(64, 32, 10)
            >>> output = layer(input_tensor)

        Note:
            The layer can be used in both training and evaluation modes. In
            training mode, it normalizes the input using the current batch
            statistics and updates the running statistics. In evaluation
            mode, it uses the running statistics for normalization.

        Todo:
            Implement functionality for different normalization strategies
            if required in future versions.
        """
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-0.9, +0.9)  # W will be positive-definite
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert xr.shape == xi.shape
        assert xr.size(1) == self.num_features

    def forward(self, x):
        """
            Passes the input through the DCUNet architecture.

        This method processes the input complex spectrogram tensor and
        embeds the time information before passing it through the
        encoder-decoder architecture. The expected input shape is
        $(batch, nfreqs, time)$, where $nfreqs - 1$ must be divisible by
        the frequency strides of the encoders, and $time - 1$ must be
        divisible by the time strides of the encoders.

        Args:
            spec (Tensor): A complex spectrogram tensor of shape
                (batch, nfreqs, time). The last dimension should
                represent time.

        Returns:
            Tensor: The output tensor of shape (batch, time) or (time)
                depending on the architecture and input dimensions.

        Raises:
            TypeError: If the input shape is not compatible with the
                expected dimensions.

        Examples:
            >>> net = DCUNet()
            >>> dnn_input = torch.randn(4, 2, 257, 256) + 1j * torch.randn(4, 2, 257, 256)
            >>> score = net(dnn_input, torch.randn(4))
            >>> print(score.shape)  # Output shape will depend on the architecture

        Note:
            Ensure that the input tensor dimensions conform to the
            requirements outlined in the method description. The method
            will perform checks and raise errors if the dimensions do
            not match the expected format.
        """
        xr, xi = x.real, x.imag
        self._check_input_dim(xr, xi)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        #
        # NOTE: The precise meaning of the "training flag" is:
        #       True:  Normalize using batch   statistics, update running statistics
        #              if they are being collected.
        #       False: Normalize using running statistics, ignore batch   statistics.
        #
        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i != 1]
        vdim = [1] * xr.dim()
        vdim[1] = xr.size(1)

        #
        # Mean M Computation and Centering
        #
        # Includes running mean update if training and running.
        #
        if training:
            Mr, Mi = xr, xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)
        xr, xi = xr - Mr, xi - Mi

        #
        # Variance Matrix V Computation
        #
        # Includes epsilon numerical stabilizer/Tikhonov regularizer.
        # Includes running variance update if training and running.
        #
        if training:
            Vrr = xr * xr
            Vri = xr * xi
            Vii = xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)
        Vrr = Vrr + self.eps
        Vri = Vri
        Vii = Vii + self.eps

        #
        # Matrix Inverse Square Root U = V^-0.5
        #
        # sqrt of a 2x2 matrix,
        # - https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        tau = Vrr + Vii
        delta = torch.addcmul(Vrr * Vii, Vri, Vri, value=-1)
        s = delta.sqrt()
        t = (tau + 2 * s).sqrt()

        # matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html
        rst = (s * t).reciprocal()
        Urr = (s + Vii) * rst
        Uii = (s + Vrr) * rst
        Uri = (-Vri) * rst

        #
        # Optionally left-multiply U by affine weights W to produce combined
        # weights Z, left-multiply the inputs by Z, then optionally bias them.
        #
        # y = Zx + B
        # y = WUx + B
        # y = [Wrr Wri][Urr Uri] [xr] + [Br]
        #     [Wir Wii][Uir Uii] [xi]   [Bi]
        #
        if self.affine:
            Wrr, Wri, Wii = (
                self.Wrr.view(vdim),
                self.Wri.view(vdim),
                self.Wii.view(vdim),
            )
            Zrr = (Wrr * Urr) + (Wri * Uri)
            Zri = (Wrr * Uri) + (Wri * Uii)
            Zir = (Wri * Urr) + (Wii * Uri)
            Zii = (Wri * Uri) + (Wii * Uii)
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr = (Zrr * xr) + (Zri * xi)
        yi = (Zir * xr) + (Zii * xi)

        if self.affine:
            yr = yr + self.Br.view(vdim)
            yi = yi + self.Bi.view(vdim)

        return torch.view_as_complex(torch.stack([yr, yi], dim=-1))

    def extra_repr(self):
        """
            Applies Batch Normalization for complex-valued inputs.

        This layer normalizes the input using the mean and variance
        of each feature across the batch. It can optionally apply
        affine transformation and track running statistics.

        Attributes:
            num_features (int): Number of features (channels).
            eps (float): A value added to the denominator for numerical stability.
            momentum (float): Momentum for the moving average.
            affine (bool): If True, this layer has learnable parameters.
            track_running_stats (bool): If True, tracks the running mean and
                variance during training.

        Args:
            num_features (int): Number of features (channels) in the input.
            eps (float, optional): A value added to the denominator for numerical
                stability (default: 1e-5).
            momentum (float, optional): Momentum for the moving average (default: 0.1).
            affine (bool, optional): If True, this layer has learnable parameters
                (default: True).
            track_running_stats (bool, optional): If True, tracks the running mean
                and variance during training (default: False).

        Example:
            >>> layer = ComplexBatchNorm(num_features=64)
            >>> input_tensor = torch.randn(10, 64, 32, 32) + 1j * torch.randn(10, 64, 32, 32)
            >>> output_tensor = layer(input_tensor)

        Raises:
            AssertionError: If the real and imaginary parts of the input do not
                have the same shape.
        """
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )


if __name__ == "__main__":
    net = DCUNet()
    dnn_input = torch.randn(4, 2, 257, 256) + 1j * torch.randn(4, 2, 257, 256)

    score = net(dnn_input, torch.randn(4))
    print(score.shape)
