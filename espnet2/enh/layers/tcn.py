# Implementation of the TCN proposed in
# Luo. et al.  "Conv-tasnet: Surpassing ideal time–frequency
# magnitude masking for speech separation."
#
# The code is based on:
# https://github.com/kaituoxu/Conv-TasNet/blob/master/src/conv_tasnet.py
# Licensed under MIT.
#


import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.enh.layers.adapt_layers import make_adapt_layer

EPS = torch.finfo(torch.get_default_dtype()).eps


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network for speech separation.

    This class implements the Temporal Convolutional Network (TCN) as proposed
    in Luo et al. "Conv-tasnet: Surpassing ideal time–frequency magnitude 
    masking for speech separation." The architecture is designed to perform 
    speech separation by estimating masks for different speakers from a 
    mixture of audio signals.

    Attributes:
        C: Number of speakers.
        mask_nonlinear: Non-linear function used to generate masks.
        skip_connection: Boolean indicating if skip connections are used.
        out_channel: Number of output channels.

    Args:
        N: Number of filters in the autoencoder.
        B: Number of channels in the bottleneck 1x1-conv block.
        H: Number of channels in convolutional blocks.
        P: Kernel size in convolutional blocks.
        X: Number of convolutional blocks in each repeat.
        R: Number of repeats.
        C: Number of speakers.
        Sc: Number of channels in skip-connection paths' 1x1-conv blocks.
        out_channel: Number of output channels; if None, `N` is used.
        norm_type: Type of normalization; options include "BN", "gLN", "cLN".
        causal: Boolean indicating if the model is causal.
        pre_mask_nonlinear: Non-linear function before mask generation.
        mask_nonlinear: Non-linear function to generate the mask.

    Returns:
        est_mask: Estimated masks for the speakers.

    Raises:
        ValueError: If an unsupported mask non-linear function is provided.

    Examples:
        >>> model = TemporalConvNet(N=64, B=32, H=128, P=3, X=4, R=2, C=2)
        >>> mixture_w = torch.randn(10, 64, 100)  # Batch size 10, 64 channels, 100 length
        >>> est_mask = model(mixture_w)
        >>> print(est_mask.shape)  # Should output: torch.Size([10, 2, 64, 100])
    """
    def __init__(
        self,
        N,
        B,
        H,
        P,
        X,
        R,
        C,
        Sc=None,
        out_channel=None,
        norm_type="gLN",
        causal=False,
        pre_mask_nonlinear="linear",
        mask_nonlinear="relu",
    ):
        """Basic Module of tasnet.

        Args:
            N: Number of filters in autoencoder
            B: Number of channels in bottleneck 1 * 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            Sc: Number of channels in skip-connection paths' 1x1-conv blocks
            out_channel: Number of output channels
                if it is None, `N` will be used instead.
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            pre_mask_nonlinear: the non-linear function before masknet
            mask_nonlinear: use which non-linear function to generate mask
        """
        super().__init__()
        # Hyper-parameter
        self.C = C
        self.mask_nonlinear = mask_nonlinear
        self.skip_connection = Sc is not None
        self.out_channel = N if out_channel is None else out_channel
        if self.skip_connection:
            assert Sc == B, (Sc, B)
        # Components
        # [M, N, K] -> [M, N, K]
        layer_norm = ChannelwiseLayerNorm(N)
        # [M, N, K] -> [M, B, K]
        bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)
        # [M, B, K] -> [M, B, K]
        repeats = []

        self.receptive_field = 0
        for r in range(R):
            blocks = []
            for x in range(X):
                dilation = 2**x
                if r == 0 and x == 0:
                    self.receptive_field += P
                else:
                    self.receptive_field += (P - 1) * dilation
                padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
                blocks += [
                    TemporalBlock(
                        B,
                        H,
                        Sc,
                        P,
                        stride=1,
                        padding=padding,
                        dilation=dilation,
                        norm_type=norm_type,
                        causal=causal,
                    )
                ]
            repeats += [nn.Sequential(*blocks)]
        temporal_conv_net = nn.Sequential(*repeats)
        # [M, B, K] -> [M, C*N, K]
        mask_conv1x1 = nn.Conv1d(B, C * self.out_channel, 1, bias=False)
        # Put together (for compatibility with older versions)
        if pre_mask_nonlinear == "linear":
            self.network = nn.Sequential(
                layer_norm, bottleneck_conv1x1, temporal_conv_net, mask_conv1x1
            )
        else:
            activ = {
                "prelu": nn.PReLU(),
                "relu": nn.ReLU(),
                "tanh": nn.Tanh(),
                "sigmoid": nn.Sigmoid(),
            }[pre_mask_nonlinear]
            self.network = nn.Sequential(
                layer_norm, bottleneck_conv1x1, temporal_conv_net, activ, mask_conv1x1
            )

    def forward(self, mixture_w):
        """
        Perform forward pass of the Temporal Convolutional Network.

        This method processes the input mixture of audio signals and estimates 
        the masks for each speaker using the temporal convolutional network. 
        The input is expected to be a tensor of shape [M, N, K], where:
        - M is the batch size,
        - N is the number of input channels (filters), and
        - K is the sequence length.

        Args:
            mixture_w (torch.Tensor): A tensor of shape [M, N, K], where M is 
            the batch size, N is the number of input channels, and K is the 
            sequence length.

        Returns:
            torch.Tensor: A tensor of shape [M, C, N, K] representing the estimated 
            masks for each speaker, where C is the number of speakers.

        Raises:
            ValueError: If an unsupported mask non-linear function is specified.

        Examples:
            >>> model = TemporalConvNet(N=64, B=32, H=64, P=3, X=4, R=2, C=2)
            >>> mixture = torch.randn(10, 64, 100)  # Example input
            >>> estimated_masks = model.forward(mixture)
            >>> print(estimated_masks.shape)  # Output: torch.Size([10, 2, 64, 100])
        """
        M, N, K = mixture_w.size()
        bottleneck = self.network[:2]
        tcns = self.network[2]
        masknet = self.network[3:]
        output = bottleneck(mixture_w)
        skip_conn = 0.0
        for block in tcns:
            for layer in block:
                tcn_out = layer(output)
                if self.skip_connection:
                    residual, skip = tcn_out
                    skip_conn = skip_conn + skip
                else:
                    residual = tcn_out
                output = output + residual
        # Use residual output when no skip connection
        if self.skip_connection:
            score = masknet(skip_conn)
        else:
            score = masknet(output)

        # [M, C*self.out_channel, K] -> [M, C, self.out_channel, K]
        score = score.view(M, self.C, self.out_channel, K)
        if self.mask_nonlinear == "softmax":
            est_mask = torch.softmax(score, dim=1)
        elif self.mask_nonlinear == "relu":
            est_mask = torch.relu(score)
        elif self.mask_nonlinear == "sigmoid":
            est_mask = torch.sigmoid(score)
        elif self.mask_nonlinear == "tanh":
            est_mask = torch.tanh(score)
        elif self.mask_nonlinear == "linear":
            est_mask = score
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask


class TemporalConvNetInformed(TemporalConvNet):
    """
    Basic Module of TasNet with adaptation layers.

        This class extends the basic TemporalConvNet to include
        adaptation layers that modify the output based on the
        speaker embedding provided. It is designed for speech
        separation tasks.

        Args:
            N: Number of filters in autoencoder.
            B: Number of channels in bottleneck 1 * 1-conv block.
            H: Number of channels in convolutional blocks.
            P: Kernel size in convolutional blocks.
            X: Number of convolutional blocks in each repeat.
            R: Number of repeats.
            Sc: Number of channels in skip-connection paths' 1x1-conv blocks.
            out_channel: Number of output channels. If None, `N` will be used.
            norm_type: Normalization type; options are BN, gLN, cLN.
            causal: If True, the model will be causal.
            pre_mask_nonlinear: Non-linear function before masknet.
            mask_nonlinear: Non-linear function to generate mask.
            i_adapt_layer: Index of the adaptation layer.
            adapt_layer_type: Type of adaptation layer. 
                See espnet2.enh.layers.adapt_layers for options.
            adapt_enroll_dim: Dimensionality of the speaker embedding.
        
        Raises:
            ValueError: If an unsupported mask non-linear function is specified.

        Examples:
            # Create an instance of TemporalConvNetInformed
            model = TemporalConvNetInformed(
                N=256,
                B=128,
                H=256,
                P=3,
                X=8,
                R=3,
                Sc=64,
                out_channel=256,
                norm_type="gLN",
                causal=True,
                pre_mask_nonlinear="prelu",
                mask_nonlinear="relu",
                i_adapt_layer=7,
                adapt_layer_type="mul",
                adapt_enroll_dim=128
            )

            # Forward pass with mixture and enrollment embedding
            mixture = torch.randn(32, 256, 100)  # [M, N, K]
            enroll_emb = torch.randn(32, 256)    # [M, adapt_enroll_dim]
            output_mask = model(mixture, enroll_emb)
    """
    def __init__(
        self,
        N,
        B,
        H,
        P,
        X,
        R,
        Sc=None,
        out_channel=None,
        norm_type="gLN",
        causal=False,
        pre_mask_nonlinear="prelu",
        mask_nonlinear="relu",
        i_adapt_layer: int = 7,
        adapt_layer_type: str = "mul",
        adapt_enroll_dim: int = 128,
        **adapt_layer_kwargs
    ):
        """Basic Module of TasNet with adaptation layers.

        Args:
            N: Number of filters in autoencoder
            B: Number of channels in bottleneck 1 * 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            Sc: Number of channels in skip-connection paths' 1x1-conv blocks
            out_channel: Number of output channels
                if it is None, `N` will be used instead.
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            pre_mask_nonlinear: the non-linear function before masknet
            mask_nonlinear: use which non-linear function to generate mask
            i_adapt_layer: int, index of the adaptation layer
            adapt_layer_type: str, type of adaptation layer
                see espnet2.enh.layers.adapt_layers for options
            adapt_enroll_dim: int, dimensionality of the speaker embedding
        """
        super().__init__(
            N,
            B,
            H,
            P,
            X,
            R,
            1,
            Sc=Sc,
            out_channel=out_channel,
            norm_type=norm_type,
            causal=causal,
            pre_mask_nonlinear=pre_mask_nonlinear,
            mask_nonlinear=mask_nonlinear,
        )
        self.i_adapt_layer = i_adapt_layer
        self.adapt_enroll_dim = adapt_enroll_dim
        self.adapt_layer_type = adapt_layer_type
        self.adapt_layer = make_adapt_layer(
            adapt_layer_type,
            indim=B,
            enrolldim=adapt_enroll_dim,
            ninputs=2 if self.skip_connection else 1,
            **adapt_layer_kwargs
        )

    def forward(self, mixture_w, enroll_emb):
        """
        TasNet forward with adaptation layers.

        This method processes the input mixture of waveforms and produces an 
        estimated mask using a temporal convolutional network. It is designed 
        to support adaptation layers that allow the model to better handle 
        variations in the input data based on speaker embeddings.

        Args:
            mixture_w: A tensor of shape [M, N, K], where M is the batch size, 
                N is the number of input channels, and K is the sequence length.
            enroll_emb: A tensor that represents the speaker embedding, with 
                shape [M, 2*adapt_enroll_dim] if skip connections are used, 
                or [M, adapt_enroll_dim] if not.

        Returns:
            est_mask: A tensor of shape [M, N, K], representing the estimated 
                mask for the input mixture.

        Raises:
            ValueError: If an unsupported mask non-linear function is specified.

        Examples:
            >>> model = TemporalConvNetInformed(N=64, B=32, H=16, P=3, X=4, R=2)
            >>> mixture_w = torch.randn(8, 64, 100)  # Example input
            >>> enroll_emb = torch.randn(8, 128)  # Example embedding
            >>> estimated_mask = model(mixture_w, enroll_emb)
            >>> print(estimated_mask.shape)  # Output: torch.Size([8, 64, 100])
        """
        M, N, K = mixture_w.size()

        bottleneck = self.network[:2]
        tcns = self.network[2]
        masknet = self.network[3:]
        output = bottleneck(mixture_w)
        skip_conn = 0.0
        for i, block in enumerate(tcns):
            for j, layer in enumerate(block):
                idx = i * len(block) + j
                is_adapt_layer = idx == self.i_adapt_layer
                tcn_out = layer(output)
                if self.skip_connection:
                    residual, skip = tcn_out
                    if is_adapt_layer:
                        residual, skip = self.adapt_layer(
                            (residual, skip), torch.chunk(enroll_emb, 2, dim=1)
                        )
                    skip_conn = skip_conn + skip
                else:
                    residual = tcn_out
                    if is_adapt_layer:
                        residual = self.adapt_layer(residual, enroll_emb)
                output = output + residual

        # Use residual output when no skip connection
        if self.skip_connection:
            score = masknet(skip_conn)
        else:
            score = masknet(output)

        # [M, self.out_channel, K]
        if self.mask_nonlinear == "softmax":
            est_mask = F.softmax(score, dim=1)
        elif self.mask_nonlinear == "relu":
            est_mask = F.relu(score)
        elif self.mask_nonlinear == "sigmoid":
            est_mask = F.sigmoid(score)
        elif self.mask_nonlinear == "tanh":
            est_mask = F.tanh(score)
        elif self.mask_nonlinear == "linear":
            est_mask = score
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask


class TemporalBlock(nn.Module):
    """
    Temporal Block for Temporal Convolutional Network.

    This class implements a temporal block that applies a sequence of operations 
    including a 1x1 convolution, activation function, normalization, and a 
    depthwise separable convolution. The block supports skip connections for 
    improved feature extraction.

    Attributes:
        skip_connection (bool): Indicates if skip connections are used.
        net (nn.Sequential): Sequential container for the convolutional layers 
            and activation functions.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        skip_channels (int or None): Number of channels for the skip connection.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride for the convolution.
        padding (int): Padding for the convolution.
        dilation (int): Dilation for the convolution.
        norm_type (str): Type of normalization to use ('gLN', 'cLN', 'BN').
        causal (bool): If True, applies causal convolution.

    Returns:
        res_out (Tensor): Output tensor after passing through the block.
        skip_out (Tensor): Output tensor for skip connections, if used.

    Examples:
        >>> temporal_block = TemporalBlock(64, 128, 32, kernel_size=3, 
        ...                                  stride=1, padding=1, 
        ...                                  dilation=1, norm_type='gLN', 
        ...                                  causal=False)
        >>> x = torch.randn(10, 64, 100)  # Batch of 10, 64 channels, length 100
        >>> output = temporal_block(x)
        >>> output.shape
        torch.Size([10, 128, 100])  # Output shape after convolution

    Note:
        The output length is the same as the input length if padding is 
        appropriately set.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        super().__init__()
        self.skip_connection = skip_channels is not None
        # [M, B, K] -> [M, H, K]
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = choose_norm(norm_type, out_channels)
        # [M, H, K] -> [M, B, K]
        dsconv = DepthwiseSeparableConv(
            out_channels,
            in_channels,
            skip_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            norm_type,
            causal,
        )
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):
        """Forward.

        Args:
            x: [M, B, K]

        Returns:
            [M, B, K]
        """
        if self.skip_connection:
            res_out, skip_out = self.net(x)
            return res_out, skip_out
        else:
            res_out = self.net(x)
            return res_out


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution Layer.

    This module implements depthwise separable convolution, which consists
    of a depthwise convolution followed by a pointwise convolution. The
    depthwise convolution applies a single filter to each input channel,
    while the pointwise convolution combines the outputs of the depthwise
    convolution.

    Attributes:
        skip_conv (nn.Conv1d or None): A convolutional layer for skip 
            connections if skip_channels is not None.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        skip_channels (int or None): Number of channels for skip connection.
        kernel_size (int): Size of the convolution kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding added to both sides of the input.
        dilation (int): Dilation factor for the convolution.
        norm_type (str): Type of normalization to apply (e.g., 'gLN', 'cLN').
        causal (bool): Whether to use causal convolution.

    Returns:
        res_out (torch.Tensor): Output tensor after pointwise convolution.
        skip_out (torch.Tensor or None): Output tensor for skip connection
            if skip_channels is not None.

    Examples:
        >>> conv = DepthwiseSeparableConv(
        ...     in_channels=16,
        ...     out_channels=32,
        ...     skip_channels=8,
        ...     kernel_size=3,
        ...     stride=1,
        ...     padding=1,
        ...     dilation=1,
        ...     norm_type='gLN',
        ...     causal=False
        ... )
        >>> x = torch.randn(10, 16, 50)  # Batch size of 10, 16 channels, length 50
        >>> res_out, skip_out = conv(x)
        >>> res_out.shape
        torch.Size([10, 32, 50])
        >>> skip_out.shape
        torch.Size([10, 8, 50])
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        super().__init__()
        # Use `groups` option to implement depthwise convolution
        # [M, H, K] -> [M, H, K]
        depthwise_conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        if causal:
            chomp = Chomp1d(padding)
        prelu = nn.PReLU()
        norm = choose_norm(norm_type, in_channels)
        # [M, H, K] -> [M, B, K]
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        # Put together
        if causal:
            self.net = nn.Sequential(depthwise_conv, chomp, prelu, norm, pointwise_conv)
        else:
            self.net = nn.Sequential(depthwise_conv, prelu, norm, pointwise_conv)

        # skip connection
        if skip_channels is not None:
            self.skip_conv = nn.Conv1d(in_channels, skip_channels, 1, bias=False)
        else:
            self.skip_conv = None

    def forward(self, x):
        """
        Perform the forward pass of the temporal convolutional network.

        This method takes an input tensor representing a mixture of audio signals
        and processes it through the network to estimate masks for separating the
        individual audio sources.

        Args:
            mixture_w: A tensor of shape [M, N, K], where:
                M (int): The batch size.
                N (int): The number of input channels (filters).
                K (int): The length of the input sequences.

        Returns:
            est_mask: A tensor of shape [M, C, N, K], representing the estimated
            masks for the C speakers, where:
                C (int): The number of speakers.

        Raises:
            ValueError: If an unsupported mask non-linear function is specified.

        Examples:
            >>> model = TemporalConvNet(N=64, B=32, H=128, P=3, X=4, R=2, C=2)
            >>> mixture = torch.randn(10, 64, 100)  # Example input tensor
            >>> masks = model.forward(mixture)
            >>> print(masks.shape)  # Output: torch.Size([10, 2, 64, 100])
        """
        shared_block = self.net[:-1]
        shared = shared_block(x)
        res_out = self.net[-1](shared)
        if self.skip_conv is None:
            return res_out
        skip_out = self.skip_conv(shared)
        return res_out, skip_out


class Chomp1d(nn.Module):
    """
    To ensure the output length is the same as the input.

    This module is designed to remove a specific number of elements 
    from the end of the input tensor along the last dimension, ensuring 
    that the output length matches the input length minus the specified 
    chomp size.

    Attributes:
        chomp_size (int): The number of elements to remove from the end 
            of the input tensor.

    Args:
        chomp_size (int): The number of elements to be removed from the 
            end of the input tensor.

    Returns:
        torch.Tensor: A tensor of shape [M, H, K] where K is the original 
            length minus chomp_size.

    Examples:
        >>> chomp = Chomp1d(chomp_size=2)
        >>> input_tensor = torch.randn(1, 3, 10)  # [M, H, Kpad]
        >>> output_tensor = chomp(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 3, 8])  # [M, H, K] after chomp

    Note:
        The input tensor is expected to have at least chomp_size 
        elements in the last dimension. If the input tensor's last 
        dimension is smaller than chomp_size, it will raise an 
        error.

    Raises:
        IndexError: If the input tensor's last dimension is less than 
            chomp_size.
    """

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Keep this API same with TasNet.

    This method processes the input mixture of audio signals and estimates
    the masks for each speaker. It performs the forward pass through the 
    temporal convolutional network defined in the `TemporalConvNet` class.

    Args:
        mixture_w: A tensor of shape [M, N, K], where:
            M (int): Batch size.
            N (int): Number of input channels (filters).
            K (int): Length of the input sequence.

    Returns:
        est_mask: A tensor of shape [M, C, N, K], where:
            C (int): Number of speakers.
            N (int): Number of output channels (filters).
            K (int): Length of the output sequence.

    Raises:
        ValueError: If the mask non-linear function specified is unsupported.

    Examples:
        >>> model = TemporalConvNet(N=16, B=4, H=8, P=3, X=2, R=2, C=2)
        >>> mixture = torch.randn(8, 16, 100)  # Example input
        >>> estimated_mask = model.forward(mixture)
        >>> print(estimated_mask.shape)
        torch.Size([8, 2, 16, 100])  # Output shape

    Note:
        The forward pass includes the bottleneck layer, the temporal
        convolutional blocks, and the mask generation layers. The output 
        mask is generated using the specified non-linear activation function.
        """
        return x[:, :, : -self.chomp_size].contiguous()


def check_nonlinear(nolinear_type):
    """
    Check if the specified nonlinear type is supported.

    This function validates the given nonlinear type against a set of
    predefined options. If the type is not supported, a ValueError is raised.

    Args:
        nonlinear_type (str): The type of nonlinear function to check. It
            must be one of the following: "softmax", "relu".

    Raises:
        ValueError: If the provided nonlinear_type is not supported.

    Examples:
        >>> check_nonlinear("relu")  # No exception raised
        >>> check_nonlinear("tanh")   # Raises ValueError
        ValueError: Unsupported nonlinear type
    """
    if nolinear_type not in ["softmax", "relu"]:
        raise ValueError("Unsupported nonlinear type")


def choose_norm(norm_type, channel_size, shape="BDT"):
    """
    Choose the appropriate normalization layer based on the norm type.

    The input of normalization will be (M, C, K), where M is the batch size.
    C is the channel size and K is the sequence length.

    Args:
        norm_type (str): The type of normalization to use. Options include:
            "gLN" for Global Layer Normalization,
            "cLN" for Channel-wise Layer Normalization,
            "BN" for Batch Normalization,
            "GN" for Group Normalization.
        channel_size (int): The number of channels to normalize.
        shape (str, optional): The shape of the input tensor, either "BDT" 
            (Batch, Depth, Time) or "BTD" (Batch, Time, Depth). Defaults to "BDT".

    Returns:
        nn.Module: The selected normalization layer.

    Raises:
        ValueError: If an unsupported normalization type is provided.

    Examples:
        >>> norm_layer = choose_norm("gLN", 64)
        >>> input_tensor = torch.randn(10, 64, 50)  # (M, C, K)
        >>> output_tensor = norm_layer(input_tensor)
    """
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size, shape=shape)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size, shape=shape)
    elif norm_type == "BN":
        # Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
        # along M and K, so this BN usage is right.
        return nn.BatchNorm1d(channel_size)
    elif norm_type == "GN":
        return nn.GroupNorm(1, channel_size, eps=1e-8)
    else:
        raise ValueError("Unsupported normalization type")


class ChannelwiseLayerNorm(nn.Module):
    """
    Channel-wise Layer Normalization (cLN).

    This module applies layer normalization across the channel dimension
    for each instance in the batch, which helps stabilize the learning
    process and can improve convergence.

    Attributes:
        gamma: Scale parameter for normalization.
        beta: Shift parameter for normalization.
        shape: Specifies the input shape format. It can be either "BDT"
            (Batch, Depth, Time) or "BTD" (Batch, Time, Depth).

    Args:
        channel_size (int): Number of channels for normalization.
        shape (str, optional): Input shape format. Default is "BDT".
            Acceptable values are "BDT" and "BTD".

    Methods:
        reset_parameters: Resets the parameters gamma and beta.
        forward: Applies the channel-wise layer normalization.

    Examples:
        >>> layer_norm = ChannelwiseLayerNorm(channel_size=64)
        >>> input_tensor = torch.randn(10, 64, 100)  # [M, N, K]
        >>> output_tensor = layer_norm(input_tensor)
        >>> output_tensor.shape
        torch.Size([10, 64, 100])

    Note:
        The input tensor should have three dimensions: [M, N, K], where
        M is the batch size, N is the number of channels, and K is the
        length of the sequence.

    Raises:
        AssertionError: If the input tensor does not have 3 dimensions.
    """

    def __init__(self, channel_size, shape="BDT"):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()
        assert shape in ["BDT", "BTD"]
        self.shape = shape

    def reset_parameters(self):
        """
        Reset the parameters of the ChannelwiseLayerNorm.

        This method initializes the learnable parameters `gamma` and `beta` of the
        channel-wise layer normalization to their default values. Specifically, `gamma`
        is set to 1 and `beta` is set to 0. This is typically called when creating
        an instance of the class or when fine-tuning the model.

        Attributes:
            gamma (torch.Tensor): Scaling parameter for the normalization.
            beta (torch.Tensor): Shifting parameter for the normalization.

        Examples:
            >>> layer_norm = ChannelwiseLayerNorm(channel_size=10)
            >>> layer_norm.gamma  # Initially filled with 1
            tensor([[[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]])
            >>> layer_norm.beta  # Initially filled with 0
            tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]])

        Note:
            It is important to reset the parameters when the model is being
            re-initialized or if you want to start training from scratch.
        """
        self.beta.data.zero_()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, y):
        """
        Forward pass of the TemporalConvNet.

    This method takes an input tensor and processes it through the
    network layers to estimate the masks for the input mixture.

    Args:
        mixture_w: A tensor of shape [M, N, K], where M is the batch size,
                   N is the number of input channels, and K is the length
                   of the input sequence.

    Returns:
        est_mask: A tensor of shape [M, C, N, K], where C is the number of
                  speakers, representing the estimated masks for the input
                  mixture.

    Raises:
        ValueError: If an unsupported mask non-linear function is specified.

    Examples:
        >>> model = TemporalConvNet(N=64, B=32, H=128, P=3, X=8, R=3, C=2)
        >>> mixture = torch.randn(10, 64, 100)  # Batch of 10, 64 channels, length 100
        >>> masks = model(mixture)
        >>> print(masks.shape)  # Should output: torch.Size([10, 2, 64, 100])
        """

        assert y.dim() == 3

        if self.shape == "BTD":
            y = y.transpose(1, 2).contiguous()

        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta

        if self.shape == "BTD":
            cLN_y = cLN_y.transpose(1, 2).contiguous()

        return cLN_y


class GlobalLayerNorm(nn.Module):
    """
    Global Layer Normalization (gLN).

    This class implements global layer normalization, which normalizes
    the input across all dimensions, helping to stabilize the learning
    process and improve convergence. It scales and shifts the normalized
    output using learnable parameters gamma and beta.

    Attributes:
        gamma: Learnable parameter for scaling the normalized output.
        beta: Learnable parameter for shifting the normalized output.
        shape: Defines the shape of the input tensor. 
               Options are "BDT" (Batch, Depth, Time) or "BTD" (Batch, Time, Depth).

    Args:
        channel_size: Number of channels to normalize.
        shape: Shape of the input tensor, either "BDT" or "BTD". 

    Raises:
        AssertionError: If the shape is not "BDT" or "BTD".

    Examples:
        >>> gLN = GlobalLayerNorm(channel_size=64, shape="BDT")
        >>> input_tensor = torch.randn(32, 64, 100)  # [M, N, K]
        >>> output_tensor = gLN(input_tensor)
        >>> output_tensor.shape
        torch.Size([32, 64, 100])
    """

    def __init__(self, channel_size, shape="BDT"):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()
        assert shape in ["BDT", "BTD"]
        self.shape = shape

    def reset_parameters(self):
        """
        Global Layer Normalization (gLN).

    This class implements global layer normalization, which normalizes the 
    input across both spatial dimensions and batches, making it suitable for 
    tasks where the input distribution varies across these dimensions.

    Attributes:
        gamma: Learnable scale parameter of shape (1, channel_size, 1).
        beta: Learnable shift parameter of shape (1, channel_size, 1).
        shape: The shape of the input tensor, either "BDT" or "BTD".

    Args:
        channel_size: The number of channels in the input tensor.
        shape: The shape of the input tensor, either "BDT" or "BTD".

    Raises:
        AssertionError: If the shape is not one of "BDT" or "BTD".

    Examples:
        >>> import torch
        >>> gLN = GlobalLayerNorm(channel_size=10)
        >>> input_tensor = torch.randn(32, 10, 100)  # [M, N, K]
        >>> output_tensor = gLN(input_tensor)
        >>> output_tensor.shape
        torch.Size([32, 10, 100])

    Note:
        The forward pass computes the mean and variance of the input tensor 
        and normalizes it using the learnable parameters gamma and beta.

    Todo:
        Add support for different shapes in the forward method.
        """
        self.beta.data.zero_()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, y):
        """
        Global Layer Normalization (gLN).

    This layer normalizes the input across all dimensions except the batch
    dimension, applying a learnable scale and shift to the normalized values.
    
    Attributes:
        gamma (torch.nn.Parameter): Learnable scale parameter.
        beta (torch.nn.Parameter): Learnable shift parameter.
        shape (str): Defines the input shape arrangement, either 'BDT' or 'BTD'.
    
    Args:
        channel_size (int): The number of channels in the input.
        shape (str): The shape of the input, either 'BDT' (Batch, Channel, Time)
                     or 'BTD' (Batch, Time, Channel). Defaults to 'BDT'.
    
    Returns:
        gLN_y: Normalized output tensor with the same shape as the input.

    Examples:
        >>> layer_norm = GlobalLayerNorm(channel_size=64)
        >>> input_tensor = torch.randn(32, 64, 100)  # [M, N, K]
        >>> output_tensor = layer_norm(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([32, 64, 100])  # Output shape remains the same

    Note:
        The layer applies the normalization based on the mean and variance
        calculated across the specified dimensions, ensuring stability and
        faster convergence during training.

    Raises:
        AssertionError: If the shape is not 'BDT' or 'BTD'.
        """
        if self.shape == "BTD":
            y = y.transpose(1, 2).contiguous()

        mean = y.mean(dim=(1, 2), keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=(1, 2), keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta

        if self.shape == "BTD":
            gLN_y = gLN_y.transpose(1, 2).contiguous()
        return gLN_y
