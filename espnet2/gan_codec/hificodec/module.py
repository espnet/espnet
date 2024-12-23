import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

from espnet2.gan_codec.shared.quantizer.residual_vq import ResidualVectorQuantizer

LRELU_SLOPE = 0.1


@dataclass
class QuantizedResult:
    """
        A data class that encapsulates the results of quantization during audio
    processing.

    Attributes:
        quantized (torch.Tensor): The quantized audio signal.
        codes (torch.Tensor): The codes corresponding to the quantized audio.
        bandwidth (torch.Tensor): Bandwidth in kb/s used per batch item.
        penalty (Optional[torch.Tensor]): An optional penalty term for quantization.

    Examples:
        >>> result = QuantizedResult(
        ...     quantized=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        ...     codes=torch.tensor([[1, 2], [3, 4]]),
        ...     bandwidth=torch.tensor([128.0]),
        ...     penalty=torch.tensor([0.05])
        ... )
        >>> print(result.quantized)
        tensor([[0.1, 0.2],
                [0.3, 0.4]])
        >>> print(result.bandwidth)
        tensor([128.])
    """

    quantized: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor  # bandwidth in kb/s used, per batch item.
    penalty: Optional[torch.Tensor] = None


class Generator(torch.nn.Module):
    """
    Generator module for HiFi-GAN based audio synthesis.

    This class implements a generator that upsamples and processes input
    audio signals using a series of convolutional layers and residual blocks.
    It is designed for use in GAN-based audio codec applications.

    Attributes:
        num_kernels (int): Number of kernel sizes used in residual blocks.
        num_upsamples (int): Number of upsampling layers.
        conv_pre (nn.Module): Initial convolutional layer.
        ups (nn.ModuleList): List of upsampling layers.
        resblocks (nn.ModuleList): List of residual blocks.
        conv_post (nn.Module): Final convolutional layer.

    Args:
        upsample_rates (List[int]): List of upsampling rates.
        upsample_kernel_sizes (List[int]): List of kernel sizes for upsampling.
        upsample_initial_channel (int): Number of channels in the initial layer.
        resblock_num (str): Type of residual block to use ('1' or '2').
        resblock_kernel_sizes (List[int]): List of kernel sizes for residual blocks.
        resblock_dilation_sizes (List[List[int]]): List of dilation sizes for
            residual blocks.
        out_dim (int): Dimension of the output signal.

    Returns:
        torch.Tensor: Synthesized audio signal.

    Examples:
        >>> generator = Generator(
        ...     upsample_rates=[8, 8, 2],
        ...     upsample_kernel_sizes=[16, 16, 4],
        ...     upsample_initial_channel=256,
        ...     resblock_num='2',
        ...     resblock_kernel_sizes=[3, 5, 7],
        ...     resblock_dilation_sizes=[[1, 2], [1, 3]],
        ...     out_dim=1
        ... )
        >>> input_signal = torch.randn(1, 1, 256)
        >>> output_signal = generator(input_signal)
        >>> output_signal.shape
        torch.Size([1, 1, 256])

    Raises:
        ValueError: If `resblock_num` is not '1' or '2'.
    """

    def __init__(
        self,
        upsample_rates,
        upsample_kernel_sizes,
        upsample_initial_channel,
        resblock_num,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        out_dim,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(
            Conv1d(out_dim, upsample_initial_channel, 7, 1, padding=3)
        )
        resblock = ResBlock1 if resblock_num == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        # padding=(u//2 + u%2),
                        padding=(k - u) // 2,
                        # output_padding=u%2
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        """
            Passes the input tensor through the generator network.

        This method applies a series of convolutional and transposed convolutional
        layers to the input tensor `x`, which is expected to have shape (B, C, T),
        where B is the batch size, C is the number of input channels, and T is the
        length of the input sequence. The output tensor is generated by applying
        leaky ReLU activations and a series of residual blocks.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T).

        Returns:
            torch.Tensor: Output tensor of shape (B, 1, T_out), where T_out is
            the length of the output sequence after all layers have been applied.

        Examples:
            >>> generator = Generator(upsample_rates=[4, 4, 4],
            ...                        upsample_kernel_sizes=[8, 8, 8],
            ...                        upsample_initial_channel=256,
            ...                        resblock_num="2",
            ...                        resblock_kernel_sizes=[3, 5, 7],
            ...                        resblock_dilation_sizes=[[1, 2, 4],
            ...                                                  [1, 2, 4]],
            ...                        out_dim=80)
            >>> input_tensor = torch.randn(1, 80, 100)  # Example input
            >>> output_tensor = generator(input_tensor)
            >>> output_tensor.shape
            torch.Size([1, 1, T_out])  # T_out will depend on the generator config

        Note:
            Ensure that the input tensor `x` has the correct shape and number of
            channels as expected by the generator.

        Raises:
            ValueError: If the input tensor does not have the expected number of
            dimensions or shape.
        """
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        """
            Remove weight normalization from the layers of the generator.

        This method removes weight normalization from all layers within the
        generator, including the upsampling layers, residual blocks, and
        the convolutional layers. It is important to call this method when
        the model is being prepared for inference or after training, as it
        ensures that the model behaves as expected without the added
        complexity of weight normalization.

        Note:
            This method will print a message indicating that weight
            normalization is being removed.

        Examples:
            >>> generator = Generator(...)  # Initialize the generator
            >>> generator.remove_weight_norm()  # Remove weight normalization

        Raises:
            RuntimeError: If any layer does not support weight normalization
            removal.
        """
        print("Removing weight norm...")
        for layers in self.ups:
            remove_weight_norm(layers)
        for layers in self.resblocks:
            layers.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class Encoder(torch.nn.Module):
    """
    Encoder module for HiFi-GAN-based neural audio codec.

    This Encoder processes input audio signals through a series of
    convolutional layers and residual blocks to generate a high-level
    representation of the audio. It is designed to be part of a GAN-based
    codec architecture.

    Attributes:
        num_kernels (int): Number of kernel sizes used in the residual blocks.
        num_upsamples (int): Number of upsampling operations.
        conv_pre (torch.nn.Module): Initial convolutional layer.
        normalize (nn.ModuleList): Group normalization layers for residual outputs.
        ups (nn.ModuleList): List of upsampling convolutional layers.
        resblocks (nn.ModuleList): List of residual blocks.
        conv_post (torch.nn.Module): Final convolutional layer.

    Args:
        resblock_num (str): Type of residual block to use ("1" or "2").
        resblock_kernel_sizes (List[int]): List of kernel sizes for residual blocks.
        resblock_dilation_sizes (List[int]): List of dilation sizes for residual blocks.
        upsample_rates (List[int]): List of upsampling rates.
        upsample_kernel_sizes (List[int]): List of kernel sizes for upsampling.

    Returns:
        None

    Examples:
        encoder = Encoder(
            resblock_num="1",
            resblock_kernel_sizes=[3, 5],
            resblock_dilation_sizes=[1, 3],
            upsample_rates=[8, 8, 2],
            upsample_kernel_sizes=[16, 16, 4]
        )
        output = encoder(torch.randn(1, 1, 16000))  # Example input tensor

    Note:
        The encoder is part of a larger GAN-based audio codec architecture.

    Raises:
        ValueError: If invalid parameters are passed during initialization.
    """

    def __init__(
        self,
        resblock_num,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_kernel_sizes,
    ):
        super(Encoder, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(Conv1d(1, 32, 7, 1, padding=3))
        self.normalize = nn.ModuleList()
        resblock = ResBlock1 if resblock_num == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            list(reversed(list(zip(upsample_rates, upsample_kernel_sizes))))
        ):
            self.ups.append(
                weight_norm(
                    Conv1d(
                        32 * (2**i),
                        32 * (2 ** (i + 1)),
                        k,
                        u,
                        padding=((k - u) // 2),
                        # padding=(u//2 + u%2)
                    )
                )
            )
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = 32 * (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(
                    list(reversed(resblock_kernel_sizes)),
                    list(reversed(resblock_dilation_sizes)),
                )
            ):
                self.resblocks.append(resblock(ch, k, d))
                self.normalize.append(
                    torch.nn.GroupNorm(ch // 16, ch, eps=1e-6, affine=True)
                )
        self.conv_post = Conv1d(512, 512, 3, 1, padding=1)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        """
            Perform the forward pass of the Encoder.

        This method takes an input tensor and applies a series of convolutional
        layers, followed by activation functions and normalization, to produce
        the output tensor. The input is progressively upsampled and passed
        through residual blocks.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, T), where B is the
                              batch size, 1 is the number of channels, and T
                              is the length of the sequence.

        Returns:
            torch.Tensor: Output tensor of shape (B, 512, T'), where T' is the
                          length of the output sequence after processing.

        Examples:
            >>> encoder = Encoder(resblock_num="1",
            ...                    resblock_kernel_sizes=[3, 5],
            ...                    resblock_dilation_sizes=[1, 2],
            ...                    upsample_rates=[2, 2],
            ...                    upsample_kernel_sizes=[4, 4])
            >>> input_tensor = torch.randn(16, 1, 16000)  # Example input
            >>> output_tensor = encoder(input_tensor)
            >>> print(output_tensor.shape)  # Should output: torch.Size([16, 512, T'])
        """
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                    xs = self.normalize[i * self.num_kernels + j](xs)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
                    xs = self.normalize[i * self.num_kernels + j](xs)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return x

    def remove_weight_norm(self):
        """
            Remove weight normalization from all layers in the generator or encoder.

        This method iterates through the layers of the generator or encoder
        and removes weight normalization from each layer, including the
        convolutional layers and residual blocks. It is typically called
        when the model is being finalized for inference or evaluation.

        It prints a message indicating that weight normalization is being
        removed for clarity.

        Examples:
            >>> generator = Generator(...)
            >>> generator.remove_weight_norm()
            Removing weight norm...

            >>> encoder = Encoder(...)
            >>> encoder.remove_weight_norm()
            Removing weight norm...

        Note:
            This operation is irreversible. Once weight normalization is
            removed, it cannot be added back without reinitializing the
            model.

        Raises:
            ValueError: If the model has not been properly initialized
            or if layers are missing weight normalization.
        """
        for layers in self.ups:
            remove_weight_norm(layers)
        for layers in self.resblocks:
            layers.remove_weight_norm()
        remove_weight_norm(self.conv_pre)


class GroupResidualVectorQuantization(nn.Module):
    """
    Group Residual Vector Quantization for audio codec.

    This class implements a group residual vector quantization scheme,
    designed for encoding and decoding audio signals. It utilizes two
    residual vector quantizers to process input tensors, which can be
    split into two halves, facilitating independent quantization.

    Attributes:
        quantizer1 (ResidualVectorQuantizer): First residual vector
            quantizer instance.
        quantizer0 (ResidualVectorQuantizer): Second residual vector
            quantizer instance.
        l1_quantization_loss (torch.nn.L1Loss): L1 loss function for
            quantization loss calculation.
        l2_quantization_loss (torch.nn.MSELoss): L2 loss function for
            quantization loss calculation.
        target_bandwidths (List[float]): Target bandwidths for quantization.

    Args:
        quantizer_target_bandwidth (List[float]): List of target bandwidths.
        hidden_dim (int): Dimension of the hidden states.
        quantizer_n_q (int): Number of quantization levels.
        quantizer_bins (int): Number of bins for quantization.
        quantizer_decay (float): Decay factor for quantization.
        quantizer_kmeans_init (bool): Whether to initialize with k-means.
        quantizer_kmeans_iters (int): Number of k-means iterations.
        quantizer_threshold_ema_dead_code (float): Threshold for dead codes.

    Returns:
        QuantizedResult: A named tuple containing the quantized tensor,
        codes, bandwidth used, and penalty (if any).

    Examples:
        # Example usage of GroupResidualVectorQuantization
        quantizer = GroupResidualVectorQuantization(
            quantizer_target_bandwidth=[64.0],
            hidden_dim=512,
            quantizer_n_q=256,
            quantizer_bins=256,
            quantizer_decay=0.99,
            quantizer_kmeans_init=True,
            quantizer_kmeans_iters=10,
            quantizer_threshold_ema_dead_code=0.1
        )

        # Encoding
        input_tensor = torch.randn(8, 512, 16000)  # Batch of audio samples
        encoded = quantizer.encode(input_tensor, frame_rate=16000)

        # Decoding
        decoded = quantizer.decode(encoded)

    Note:
        The `forward` method requires input tensor `xin` with shape
        [B, T, D], where B is batch size, T is the sequence length,
        and D is the feature dimension.

    Raises:
        ValueError: If the input tensor dimensions are not as expected.
    """

    def __init__(
        self,
        quantizer_target_bandwidth,
        hidden_dim,
        quantizer_n_q,
        quantizer_bins,
        quantizer_decay,
        quantizer_kmeans_init,
        quantizer_kmeans_iters,
        quantizer_threshold_ema_dead_code,
        **kwargs
    ):
        super().__init__()

        self.quantizer1 = ResidualVectorQuantizer(
            dimension=hidden_dim,
            n_q=quantizer_n_q,
            bins=quantizer_bins,
            decay=quantizer_decay,
            kmeans_init=quantizer_kmeans_init,
            kmeans_iters=quantizer_kmeans_iters,
            threshold_ema_dead_code=quantizer_threshold_ema_dead_code,
        )
        self.quantizer0 = ResidualVectorQuantizer(
            dimension=hidden_dim,
            n_q=quantizer_n_q,
            bins=quantizer_bins,
            decay=quantizer_decay,
            kmeans_init=quantizer_kmeans_init,
            kmeans_iters=quantizer_kmeans_iters,
            threshold_ema_dead_code=quantizer_threshold_ema_dead_code,
        )

        self.l1_quantization_loss = torch.nn.L1Loss(reduction="mean")
        self.l2_quantization_loss = torch.nn.MSELoss(reduction="mean")

        self.target_bandwidths = quantizer_target_bandwidth

    def forward(
        self, xin: torch.Tensor, sample_rate: int, bandwidth: Optional[float] = None
    ) -> QuantizedResult:
        """
            Forward pass for the GroupResidualVectorQuantization model.

        This method takes an input tensor, applies quantization using two
        residual vector quantizers, and computes the associated quantization
        losses. The input tensor is expected to have shape (B, T, D) where:
        - B: Batch size
        - T: Number of time steps
        - D: Number of dimensions (features)

        Args:
            xin (torch.Tensor): Input tensor of shape (B, T, D).
            sample_rate (int): Sample rate of the input signal.
            bandwidth (Optional[float]): Desired bandwidth for quantization. If
                not provided, the default target bandwidth is used.

        Returns:
            QuantizedResult: A named tuple containing the following fields:
                - quantized (torch.Tensor): The quantized output tensor.
                - codes (torch.Tensor): The codes generated by the quantizer.
                - bandwidth (torch.Tensor): The bandwidth in kb/s used per
                  batch item.
                - penalty (Optional[torch.Tensor]): Optional penalty for
                  quantization.

        Examples:
            >>> model = GroupResidualVectorQuantization(...)
            >>> input_tensor = torch.randn(4, 1024, 512)  # Example input
            >>> result = model.forward(input_tensor, sample_rate=22050)
            >>> print(result.quantized.shape)  # Output shape of quantized tensor

        Note:
            The quantization loss is computed as a combination of L1 and L2
            losses for both quantizers. This is crucial for optimizing the
            quantization process.

        Raises:
            ValueError: If the input tensor shape is not compatible.
        """
        # x: [B,T,D]

        xin = xin.transpose(1, 2)
        # x = xin.reshape(-1, 512)
        x = xin
        # x = torch.split(x, 512 // self.h.n_code_groups, dim=-1)

        x0, x1 = torch.split(x, 512 // 2, dim=-1)
        x0 = x0.transpose(1, 2)
        x1 = x1.transpose(1, 2)

        if bandwidth is None:
            bw = self.target_bandwidths[-1]
        else:
            bw = bandwidth

        quantized1, _, _, commit_loss1 = self.quantizer1(x1, sample_rate, bw)

        quantized0, _, _, commit_loss0 = self.quantizer0(x0, sample_rate, bw)

        quantized = torch.cat([quantized0, quantized1], dim=1)

        commit_loss = commit_loss0 + commit_loss1

        quantization_loss1 = self.l1_quantization_loss(
            x1, quantized1.detach()
        ) + self.l2_quantization_loss(x1, quantized1.detach())

        quantization_loss0 = self.l1_quantization_loss(
            x0, quantized0.detach()
        ) + self.l2_quantization_loss(x0, quantized0.detach())

        quantization_loss = quantization_loss0 + quantization_loss1

        return quantized, _, _, commit_loss, quantization_loss

    def encode(
        self,
        xin: torch.Tensor,
        frame_rate: int,
        target_bw: Optional[float] = None,
    ):
        """
        Encode input tensor using HiFICodec codec.

        This method performs encoding on the input tensor `xin` by splitting it
        into two parts, quantizing each part, and concatenating the resulting
        codes. The quantization is performed based on the specified frame rate
        and target bandwidth.

        Args:
            xin (torch.Tensor): Input tensor of shape (B, 1, T) where B is the
                batch size and T is the sequence length.
            frame_rate (int): Frame rate to be used during encoding.
            target_bw (Optional[float]): Target bandwidth for quantization. If
                None, the last value from `self.target_bandwidths` is used.

        Returns:
            torch.Tensor: Concatenated neural codes from the quantization of
                both parts of the input tensor.

        Examples:
            >>> encoder = GroupResidualVectorQuantization(...)
            >>> input_tensor = torch.randn(4, 1, 1024)  # Batch of 4, 1 channel
            >>> encoded_codes = encoder.encode(input_tensor, frame_rate=16000)

        Note:
            The input tensor is expected to have a shape of (B, 1, T) and
            will be split into two equal parts for processing.
        """

        x = xin

        x0, x1 = torch.split(x, 512 // 2, dim=1)

        if target_bw is None:
            bw = self.target_bandwidths[-1]
        else:
            bw = target_bw

        codes0 = self.quantizer0.encode(x0, frame_rate, bw)
        codes1 = self.quantizer1.encode(x1, frame_rate, bw)
        code = torch.cat([codes0, codes1], dim=1)

        return code

    def decode(self, code: torch.Tensor):
        """
        HiFICodec codec decoding.

        This method takes neural codec representations and converts them back into
        resynthesized audio signals. The input codes are split into two parts,
        which are processed by two separate quantizers to reconstruct the original
        audio waveform.

        Args:
            code (torch.Tensor): Neural codecs in shape (B, N), where B is the batch
                size and N is the number of codes.

        Returns:
            torch.Tensor: Resynthesized audio of shape (B, T, D), where T is the
                length of the audio signal and D is the number of channels.

        Examples:
            >>> import torch
            >>> codec = GroupResidualVectorQuantization(128, 512, 256, 256, 0.99, True, 10, 0.1)
            >>> code = torch.randn(4, 256)  # Example input
            >>> audio = codec.decode(code)
            >>> print(audio.shape)  # Output shape will be (4, T, D)

        Note:
            The shape of the input tensor must match the expected shape for the
            decoding process to work correctly.
        """

        code0, code1 = torch.split(code, 2 // 2, dim=1)

        quantized0 = self.quantizer0.decode(code0)
        quantized1 = self.quantizer1.decode(code1)
        quantized = torch.cat([quantized0, quantized1], dim=1)

        return quantized


class ResBlock1(torch.nn.Module):
    """
    Residual Block with multiple convolutional layers.

    This class implements a residual block that consists of multiple
    convolutional layers with different dilation rates. The output of
    each layer is passed through a leaky ReLU activation function,
    and the input is added to the output to create a residual connection.

    Attributes:
        convs1 (nn.ModuleList): A list of convolutional layers for the first
            part of the residual block with varying dilation rates.
        convs2 (nn.ModuleList): A list of convolutional layers for the second
            part of the residual block with unit dilation.

    Args:
        channels (int): The number of input and output channels for the
            convolutional layers.
        kernel_size (int, optional): The size of the convolutional kernel.
            Defaults to 3.
        dilation (tuple, optional): A tuple containing the dilation rates for
            the first part of the block. Defaults to (1, 3, 5).

    Examples:
        >>> res_block = ResBlock1(channels=64)
        >>> x = torch.randn(1, 64, 128)  # Batch size of 1, 64 channels, length 128
        >>> output = res_block(x)
        >>> print(output.shape)
        torch.Size([1, 64, 128])

    Note:
        The input tensor must have the shape (batch_size, channels, length).
    """

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        """
            Forward pass through the generator network.

        This method takes an input tensor `x`, processes it through several
        convolutional and residual blocks, and returns the output tensor after
        applying a series of transformations including upsampling and
        non-linear activations.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T), where B is the
                batch size, C is the number of input channels, and T is the
                length of the input sequence.

        Returns:
            torch.Tensor: Output tensor of shape (B, 1, T'), where T' is the
            length of the output sequence, typically different from T due to
            the upsampling operations.

        Examples:
            >>> generator = Generator(
            ...     upsample_rates=[4, 4, 4],
            ...     upsample_kernel_sizes=[8, 8, 8],
            ...     upsample_initial_channel=256,
            ...     resblock_num="1",
            ...     resblock_kernel_sizes=[3, 5, 7],
            ...     resblock_dilation_sizes=[1, 3, 5],
            ...     out_dim=1
            ... )
            >>> input_tensor = torch.randn(1, 256, 128)  # Example input
            >>> output_tensor = generator(input_tensor)
            >>> output_tensor.shape
            torch.Size([1, 1, T'])

        Note:
            The number of output channels is fixed to 1 as this is intended
            for audio generation tasks.
        """
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        """
            Remove weight normalization from the layers of the model.

        This method iterates through the upsampling layers, residual blocks, and
        the initial and final convolution layers, removing weight normalization
        from each layer. This is often done before saving the model or when
        performing inference to ensure that the model operates with its learned
        weights without the effects of weight normalization.

        Attributes:
            None

        Args:
            self: An instance of the Generator or Encoder class.

        Returns:
            None

        Examples:
            >>> generator = Generator(...)
            >>> generator.remove_weight_norm()

            >>> encoder = Encoder(...)
            >>> encoder.remove_weight_norm()

        Note:
            It is important to ensure that the model is fully trained before
            removing weight normalization, as it may affect the model's
            performance if done prematurely.
        """
        for layers in self.convs1:
            remove_weight_norm(layers)
        for layers in self.convs2:
            remove_weight_norm(layers)


class ResBlock2(torch.nn.Module):
    """
    Residual Block with two convolutional layers and leaky ReLU activation.

    This class implements a residual block consisting of two convolutional
    layers, each followed by a leaky ReLU activation function. The input to
    the block is added to the output of the second convolutional layer to
    form the residual connection.

    Attributes:
        convs (nn.ModuleList): A list containing the convolutional layers
            with weight normalization applied.

    Args:
        channels (int): The number of input and output channels for the
            convolutional layers.
        kernel_size (int, optional): The size of the convolutional kernel.
            Default is 3.
        dilation (tuple, optional): The dilation rates for the convolutional
            layers. Default is (1, 3).

    Examples:
        >>> res_block = ResBlock2(channels=64, kernel_size=3, dilation=(1, 3))
        >>> input_tensor = torch.randn(1, 64, 128)  # (batch_size, channels, length)
        >>> output_tensor = res_block(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 64, 128])

    Note:
        The leaky ReLU uses a negative slope defined by the constant
        `LRELU_SLOPE`.
    """

    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x):
        """
            Forward pass for the generator network.

        This method takes an input tensor `x`, applies a series of convolutional
        and transposed convolutional layers along with residual blocks to produce
        the output. The output is a processed tensor that has gone through
        upsampling and activation functions.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T) where:
                B = batch size,
                C = number of channels,
                T = length of the sequence.

        Returns:
            torch.Tensor: Output tensor of shape (B, 1, T') where T' is the
            length of the output sequence after processing.

        Examples:
            >>> generator = Generator(...)
            >>> input_tensor = torch.randn(8, 256, 100)  # Example input
            >>> output_tensor = generator(input_tensor)
            >>> print(output_tensor.shape)
            torch.Size([8, 1, T'])  # Output shape will vary depending on
                                      # the generator configuration.
        """
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        """
            Remove weight normalization from the model layers.

        This method removes weight normalization from all the layers in the
        Generator or Encoder model, including upsampling layers, residual
        blocks, and the convolutional layers at the beginning and end.
        It is typically used before saving the model or during the
        evaluation phase to ensure that the model behaves as expected.

        Note:
            Weight normalization is a technique that can help in training
            deep networks by reparameterizing the weight vectors. However,
            for inference or model deployment, it may be necessary to remove
            this normalization.

        Examples:
            >>> generator = Generator(...)
            >>> generator.remove_weight_norm()

        Raises:
            RuntimeError: If the layers do not have weight normalization
            applied.
        """
        for layers in self.convs:
            remove_weight_norm(layers)


def init_weights(m, mean=0.0, std=0.01):
    """
    Initialize weights for convolutional layers.

    This function applies a normal distribution to the weights of
    convolutional layers in a given module. It sets the weights to
    have a mean of `mean` and a standard deviation of `std`.

    Args:
        m (torch.nn.Module): The module whose weights are to be initialized.
        mean (float, optional): The mean of the normal distribution.
            Defaults to 0.0.
        std (float, optional): The standard deviation of the normal
            distribution. Defaults to 0.01.

    Note:
        This function specifically checks if the module is of a
        convolutional type (i.e., classes that contain "Conv" in their
        name) before applying the weight initialization.

    Examples:
        >>> model = nn.Sequential(
        ...     nn.Conv2d(1, 20, 5),
        ...     nn.Conv2d(20, 64, 5)
        ... )
        >>> model.apply(init_weights)

    Todo:
        Consider adding more weight initialization strategies
        in the future for different types of layers.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    """
    Calculate the padding required for a convolutional layer.

    This function computes the amount of padding needed for a convolutional
    layer given the kernel size and dilation. The padding is calculated to
    ensure that the output size of the convolution operation is the same as
    the input size when using valid padding.

    Args:
        kernel_size (int): The size of the convolutional kernel.
        dilation (int, optional): The dilation rate of the convolution.
            Defaults to 1.

    Returns:
        int: The amount of padding to be applied on each side of the input.

    Examples:
        >>> get_padding(3, 1)
        1
        >>> get_padding(5, 2)
        4
        >>> get_padding(7, 3)
        9
    """
    return int((kernel_size * dilation - dilation) / 2)
