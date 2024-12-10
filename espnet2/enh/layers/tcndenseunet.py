import torch
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.torch_utils.get_layer_from_string import get_layer

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class Conv2DActNorm(torch.nn.Module):
    """
    Conv2DActNorm is a building block for a convolutional layer followed by an 
activation function and instance normalization.

This module combines a 2D convolution operation with an activation function 
and group normalization to form a reusable component in neural networks, 
particularly for tasks involving image or spectrogram data.

Attributes:
    layer (torch.nn.Sequential): A sequential container that holds the 
        convolution, activation, and normalization layers.

Args:
    in_channels (int): Number of input channels.
    out_channels (int): Number of output channels.
    ksz (tuple): Kernel size for the convolution. Default is (3, 3).
    stride (tuple): Stride for the convolution. Default is (1, 2).
    padding (tuple): Padding for the convolution. Default is (1, 0).
    upsample (bool): If True, uses transposed convolution for upsampling. 
        Default is False.
    activation (callable): Activation function to use. Default is 
        torch.nn.ELU.

Returns:
    torch.Tensor: The output tensor after applying convolution, activation, 
    and normalization.

Examples:
    >>> conv_layer = Conv2DActNorm(1, 16)
    >>> input_tensor = torch.randn(1, 1, 64, 32)  # (batch, channels, height, width)
    >>> output_tensor = conv_layer(input_tensor)
    >>> output_tensor.shape
    torch.Size([1, 16, 32, 16])  # After convolution and downsampling

Raises:
    ValueError: If input parameters are invalid (e.g., negative channel 
    sizes).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        ksz=(3, 3),
        stride=(1, 2),
        padding=(1, 0),
        upsample=False,
        activation=torch.nn.ELU,
    ):
        super(Conv2DActNorm, self).__init__()

        if upsample:
            conv = torch.nn.ConvTranspose2d(
                in_channels, out_channels, ksz, stride, padding
            )
        else:
            conv = torch.nn.Conv2d(
                in_channels, out_channels, ksz, stride, padding, padding_mode="reflect"
            )
        act = get_layer(activation)()
        norm = torch.nn.GroupNorm(out_channels, out_channels, eps=1e-8)
        self.layer = torch.nn.Sequential(conv, act, norm)

    def forward(self, inp):
        """
        Forward pass of the TCNDenseUNet.

        Args:
            tf_rep (torch.Tensor): 4D tensor (multi-channel complex STFT of 
                mixture) of shape [B, T, C, F] where B is batch size, T is 
                number of frames, C is the number of microphones, and F is 
                the number of frequencies.

        Returns:
            out (torch.Tensor): Complex 3D tensor representing the monaural STFT 
                of the targets, with shape [B, T, F] where B is batch size, 
                T is number of frames, and F is number of frequencies.

        Examples:
            >>> model = TCNDenseUNet(n_spk=2, in_freqs=257, mic_channels=1)
            >>> input_tensor = torch.randn(4, 10, 1, 257)  # Example input
            >>> output = model(input_tensor)
            >>> print(output.shape)  # Should be [4, 10, 2, 257] for 2 speakers

        Note:
            The input tensor should be in the format expected by the model, 
            which is a multi-channel complex STFT representation.

        Raises:
            AssertionError: If the number of microphone channels in the input 
            tensor does not match the expected number of microphone channels 
            specified during model initialization.
        """
        return self.layer(inp)


class FreqWiseBlock(torch.nn.Module):
    """
    FreqWiseBlock, see iNeuBe paper.

Block that applies pointwise 2D convolution over
STFT-like image tensor on frequency axis.
The input is assumed to be [batch, image_channels, frames, freq].

Attributes:
    bottleneck (Conv2DActNorm): A convolutional layer with activation and
        normalization for processing input channels.
    freq_proc (Conv2DActNorm): A convolutional layer with activation and
        normalization for processing frequency channels.

Args:
    in_channels (int): Number of input channels (image axis).
    num_freqs (int): Number of complex frequencies in the input STFT
        complex image-like tensor.
    out_channels (int): Number of output channels (image axis).
    activation (callable): Activation function to use, default is
        torch.nn.ELU.

Returns:
    torch.Tensor: The output tensor after applying the frequency-wise
        processing.

Examples:
    >>> import torch
    >>> block = FreqWiseBlock(in_channels=64, num_freqs=128, out_channels=32)
    >>> input_tensor = torch.randn(10, 64, 100, 128)  # [batch, channels, frames, freq]
    >>> output_tensor = block(input_tensor)
    >>> output_tensor.shape
    torch.Size([10, 32, 100, 128])

Note:
    This block is designed to operate on STFT-like tensors, where the
    frequency axis is processed independently.
    """

    def __init__(self, in_channels, num_freqs, out_channels, activation=torch.nn.ELU):
        super(FreqWiseBlock, self).__init__()

        self.bottleneck = Conv2DActNorm(
            in_channels, out_channels, (1, 1), (1, 1), (0, 0), activation=activation
        )
        self.freq_proc = Conv2DActNorm(
            num_freqs, num_freqs, (1, 1), (1, 1), (0, 0), activation=activation
        )

    def forward(self, inp):
        """
        Forward pass for the TCNDenseUNet model.

    This method processes a 4D tensor representing the multi-channel complex 
    Short-Time Fourier Transform (STFT) of a mixture signal. The output is 
    a complex 3D tensor representing the monaural STFT of the targets.

    Args:
        tf_rep (torch.Tensor): 4D tensor (multi-channel complex STFT of mixture)
            of shape [B, T, C, F] where B is the batch size, T is the number 
            of frames, C is the number of microphone channels, and F is the 
            number of frequencies.

    Returns:
        out (torch.Tensor): Complex 3D tensor representing the monaural STFT 
            of the targets with shape [B, T, F] where B is the batch size, 
            T is the number of frames, and F is the number of frequencies.

    Examples:
        >>> model = TCNDenseUNet(n_spk=2, in_freqs=257, mic_channels=1)
        >>> mixture = torch.randn(8, 100, 1, 257)  # Batch of 8, 100 frames
        >>> output = model.forward(mixture)
        >>> print(output.shape)  # Output shape should be [8, 100, 257]

    Note:
        The input tensor is expected to be in the shape [B, T, C, F]. The 
        function will permute and reshape it accordingly to match the 
        expected input format of the model.

    Raises:
        AssertionError: If the number of microphone channels in the input 
        tensor does not match the expected number of microphone channels 
        for the model.
        """
        # bsz, chans, x, y
        out = self.freq_proc(self.bottleneck(inp).permute(0, 3, 2, 1)).permute(
            0, 3, 2, 1
        )

        return out


class DenseBlock(torch.nn.Module):
    """
    Single DenseNet block as used in iNeuBe model.

    This class implements a DenseNet block that consists of multiple 
    convolutional layers. It processes input tensors assumed to be in 
    the format [batch, image_channels, frames, freq] and is designed 
    for use in the iNeuBe model.

    Args:
        in_channels (int): Number of input channels (image axis).
        out_channels (int): Number of output channels (image axis).
        num_freqs (int): Number of complex frequencies in the input STFT 
            complex image-like tensor. The input is batch, image_channels, 
            frames, freqs.
        pre_blocks (int): Number of dense blocks before point-wise convolution 
            block over frequency axis (default: 2).
        freq_proc_blocks (int): Number of frequency axis processing blocks 
            (default: 1).
        post_blocks (int): Number of dense blocks after point-wise 
            convolution block over frequency axis (default: 2).
        ksz (tuple): Kernel size used in DenseNet Conv2D layers (default: (3, 3)).
        activation (callable): Activation function to use in the whole 
            iNeuBe model. You can use any torch supported activation 
            (e.g., 'relu' or 'elu') (default: torch.nn.ELU).
        hid_chans (int): Number of hidden channels in DenseNet Conv2D 
            (default: 32).

    Examples:
        >>> dense_block = DenseBlock(
        ...     in_channels=64,
        ...     out_channels=128,
        ...     num_freqs=257,
        ...     pre_blocks=2,
        ...     freq_proc_blocks=1,
        ...     post_blocks=2,
        ...     ksz=(3, 3),
        ...     activation=torch.nn.ReLU,
        ...     hid_chans=32
        ... )
        >>> input_tensor = torch.randn(8, 64, 100, 257)  # [batch, channels, frames, freqs]
        >>> output = dense_block(input_tensor)
        >>> output.shape
        torch.Size([8, 128, 100, 257])

    Raises:
        AssertionError: If post_blocks or pre_blocks is less than 1.

    Note:
        The output of this block can be further processed in a network 
        designed for tasks such as speech enhancement or audio signal 
        processing.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_freqs,
        pre_blocks=2,
        freq_proc_blocks=1,
        post_blocks=2,
        ksz=(3, 3),
        activation=torch.nn.ELU,
        hid_chans=32,
    ):
        super(DenseBlock, self).__init__()

        assert post_blocks >= 1
        assert pre_blocks >= 1

        self.pre_blocks = torch.nn.ModuleList([])
        tot_layers = 0
        for indx in range(pre_blocks):
            c_layer = Conv2DActNorm(
                in_channels + hid_chans * tot_layers,
                hid_chans,
                ksz,
                (1, 1),
                (1, 1),
                activation=activation,
            )
            self.pre_blocks.append(c_layer)
            tot_layers += 1

        self.freq_proc_blocks = torch.nn.ModuleList([])
        for indx in range(freq_proc_blocks):
            c_layer = FreqWiseBlock(
                in_channels + hid_chans * tot_layers,
                num_freqs,
                hid_chans,
                activation=activation,
            )
            self.freq_proc_blocks.append(c_layer)
            tot_layers += 1

        self.post_blocks = torch.nn.ModuleList([])
        for indx in range(post_blocks - 1):
            c_layer = Conv2DActNorm(
                in_channels + hid_chans * tot_layers,
                hid_chans,
                ksz,
                (1, 1),
                (1, 1),
                activation=activation,
            )
            self.post_blocks.append(c_layer)
            tot_layers += 1

        last = Conv2DActNorm(
            in_channels + hid_chans * tot_layers,
            out_channels,
            ksz,
            (1, 1),
            (1, 1),
            activation=activation,
        )
        self.post_blocks.append(last)

    def forward(self, input):
        """
        Forward pass through the TCNDenseUNet.

        This method processes the input tensor representing a multi-channel 
        complex Short-Time Fourier Transform (STFT) of a mixture and produces 
        a monaural STFT of the target signals.

        Args:
            tf_rep (torch.Tensor): A 4D tensor representing the multi-channel 
                complex STFT of the mixture. The expected shape is 
                [B, T, C, F] where:
                    B = batch size,
                    T = number of frames,
                    C = number of microphones,
                    F = number of frequencies.

        Returns:
            out (torch.Tensor): A complex 3D tensor representing the 
                monaural STFT of the targets. The shape is [B, T, F] where:
                    B = batch size,
                    T = number of frames,
                    F = number of frequencies.

        Examples:
            >>> model = TCNDenseUNet(n_spk=2, in_freqs=257, mic_channels=1)
            >>> input_tensor = torch.randn(4, 100, 1, 257)  # Example input
            >>> output = model.forward(input_tensor)
            >>> print(output.shape)
            torch.Size([4, 100, 257])  # Output shape

        Note:
            The input tensor should contain complex values as separate real 
            and imaginary parts. This function concatenates the real and 
            imaginary parts and reshapes them for processing.

        Raises:
            AssertionError: If the number of microphones in the input does 
            not match the expected number of microphone channels.
        """
        # batch, channels, frames, freq

        out = [input]
        for pre_block in self.pre_blocks:
            c_out = pre_block(torch.cat(out, 1))
            out.append(c_out)

        for freq_block in self.freq_proc_blocks:
            c_out = freq_block(torch.cat(out, 1))
            out.append(c_out)

        for post_block in self.post_blocks:
            c_out = post_block(torch.cat(out, 1))
            out.append(c_out)

        return c_out


class TCNResBlock(torch.nn.Module):
    """
    Single depth-wise separable TCN block as used in iNeuBe TCN.

    This block implements a depth-wise separable convolution followed by a 
    point-wise convolution. It applies group normalization and an activation 
    function to the input features, allowing for efficient processing of 
    temporal data.

    Args:
        in_chan (int): Number of input feature channels.
        out_chan (int): Number of output feature channels.
        ksz (int, optional): Kernel size. Defaults to 3.
        stride (int, optional): Stride in depth-wise convolution. Defaults to 1.
        dilation (int, optional): Dilation in depth-wise convolution. Defaults to 1.
        activation (callable, optional): Activation function to use in the whole 
            iNeuBe model, you can use any torch supported activation 
            e.g. 'relu' or 'elu'. Defaults to `torch.nn.ELU`.

    Examples:
        >>> tcn_block = TCNResBlock(in_chan=64, out_chan=128)
        >>> input_tensor = torch.randn(32, 64, 100)  # [batch_size, channels, frames]
        >>> output_tensor = tcn_block(input_tensor)
        >>> output_tensor.shape
        torch.Size([32, 128, 100])  # Output will have the shape of [B, C, F]

    Returns:
        torch.Tensor: Output tensor of shape [B, out_chan, F] where B is the 
        batch size, out_chan is the number of output channels, and F is the 
        number of frames.

    Note:
        The input tensor should be 3D with shape [B, C, F] where B is the 
        batch size, C is the number of channels, and F is the number of frames.
    """

    def __init__(
        self, in_chan, out_chan, ksz=3, stride=1, dilation=1, activation=torch.nn.ELU
    ):
        super(TCNResBlock, self).__init__()
        padding = dilation
        dconv = torch.nn.Conv1d(
            in_chan,
            in_chan,
            ksz,
            stride,
            padding=padding,
            dilation=dilation,
            padding_mode="reflect",
            groups=in_chan,
        )
        point_conv = torch.nn.Conv1d(in_chan, out_chan, 1)

        self.layer = torch.nn.Sequential(
            torch.nn.GroupNorm(in_chan, in_chan, eps=1e-8),
            get_layer(activation)(),
            dconv,
            point_conv,
        )

    def forward(self, inp):
        """
        forward.

    Args:
        tf_rep (torch.Tensor): 4D tensor (multi-channel complex STFT of mixture)
            of shape [B, T, C, F], where B is the batch size, T is the number 
            of frames, C is the number of microphone channels, and F is the 
            number of frequencies.

    Returns:
        out (torch.Tensor): complex 3D tensor representing the monaural STFT 
            of the targets, with shape [B, T, F], where B is the batch size, 
            T is the number of frames, and F is the number of frequencies.

    Examples:
        >>> import torch
        >>> model = TCNDenseUNet()
        >>> tf_rep = torch.randn(8, 64, 2, 257)  # 8 samples, 64 frames, 2 mics, 257 freqs
        >>> output = model(tf_rep)
        >>> print(output.shape)  # Expected output shape: [8, 2, 257]

    Note:
        The input tensor must be permuted to match the expected shape before 
        being passed to this method. The input tensor is assumed to be a 
        multi-channel complex STFT representation.
        """
        # [B, C, F] batch, channels, frames
        return self.layer(inp) + inp


class TCNDenseUNet(torch.nn.Module):
    """
    TCNDenseNet block from iNeuBe.

    Reference:
    Lu, Y. J., Cornell, S., Chang, X., Zhang, W., Li, C., Ni, Z., ... & Watanabe, S.
    Towards Low-Distortion Multi-Channel Speech Enhancement:
    The ESPNET-Se Submission to the L3DAS22 Challenge. ICASSP 2022 p. 9201-9205.

    Args:
        n_spk (int): Number of output sources/speakers.
        in_freqs (int): Number of complex STFT frequencies.
        mic_channels (int): Number of microphones channels
            (only fixed-array geometry supported).
        hid_chans (int): Number of channels in the subsampling/upsampling conv layers.
        hid_chans_dense (int): Number of channels in the densenet layers
            (reduce this to reduce VRAM requirements).
        ksz_dense (tuple): Kernel size in the densenet layers through iNeuBe.
        ksz_tcn (int): Kernel size in the TCN submodule.
        tcn_repeats (int): Number of repetitions of blocks in the TCN submodule.
        tcn_blocks (int): Number of blocks in the TCN submodule.
        tcn_channels (int): Number of channels in the TCN submodule.
        activation (callable): Activation function to use in the whole iNeuBe model,
            you can use any torch supported activation e.g. 'relu' or 'elu'.

    Attributes:
        n_spk (int): Number of output sources/speakers.
        in_channels (int): Number of input frequencies.
        mic_channels (int): Number of microphone channels.
        encoder (torch.nn.ModuleList): List of encoder layers.
        tcn (torch.nn.Sequential): TCN block composed of multiple TCNResBlocks.
        decoder (torch.nn.ModuleList): List of decoder layers.

    Examples:
        >>> model = TCNDenseUNet(n_spk=2, in_freqs=257, mic_channels=1)
        >>> input_tensor = torch.randn(8, 2, 1, 257)  # Batch size 8
        >>> output = model(input_tensor)
        >>> print(output.shape)  # Should output a shape of [8, 2, F] where F is the output frequency
    """

    def __init__(
        self,
        n_spk=1,
        in_freqs=257,
        mic_channels=1,
        hid_chans=32,
        hid_chans_dense=32,
        ksz_dense=(3, 3),
        ksz_tcn=3,
        tcn_repeats=4,
        tcn_blocks=7,
        tcn_channels=384,
        activation=torch.nn.ELU,
    ):
        super(TCNDenseUNet, self).__init__()
        self.n_spk = n_spk
        self.in_channels = in_freqs
        self.mic_channels = mic_channels

        num_freqs = in_freqs - 2
        first = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.mic_channels * 2,
                hid_chans,
                (3, 3),
                (1, 1),
                (1, 0),
                padding_mode="reflect",
            ),
            DenseBlock(
                hid_chans,
                hid_chans,
                num_freqs,
                ksz=ksz_dense,
                activation=activation,
                hid_chans=hid_chans_dense,
            ),
        )

        freq_axis_dims = self._get_depth(num_freqs)
        self.encoder = torch.nn.ModuleList([])
        self.encoder.append(first)

        for layer_indx in range(len(freq_axis_dims)):
            downsample = Conv2DActNorm(
                hid_chans, hid_chans, (3, 3), (1, 2), (1, 0), activation=activation
            )
            denseblocks = DenseBlock(
                hid_chans,
                hid_chans,
                freq_axis_dims[layer_indx],
                ksz=ksz_dense,
                activation=activation,
                hid_chans=hid_chans_dense,
            )
            c_layer = torch.nn.Sequential(downsample, denseblocks)
            self.encoder.append(c_layer)

        self.encoder.append(
            Conv2DActNorm(
                hid_chans, hid_chans * 2, (3, 3), (1, 2), (1, 0), activation=activation
            )
        )
        self.encoder.append(
            Conv2DActNorm(
                hid_chans * 2,
                hid_chans * 4,
                (3, 3),
                (1, 2),
                (1, 0),
                activation=activation,
            )
        )
        self.encoder.append(
            Conv2DActNorm(
                hid_chans * 4,
                tcn_channels,
                (3, 3),
                (1, 1),
                (1, 0),
                activation=activation,
            )
        )

        self.tcn = []
        for r in range(tcn_repeats):
            for x in range(tcn_blocks):
                self.tcn.append(
                    TCNResBlock(
                        tcn_channels,
                        tcn_channels,
                        ksz_tcn,
                        dilation=2**x,
                        activation=activation,
                    )
                )

        self.tcn = torch.nn.Sequential(*self.tcn)
        self.decoder = torch.nn.ModuleList([])
        self.decoder.append(
            Conv2DActNorm(
                tcn_channels * 2,
                hid_chans * 4,
                (3, 3),
                (1, 1),
                (1, 0),
                activation=activation,
                upsample=True,
            )
        )
        self.decoder.append(
            Conv2DActNorm(
                hid_chans * 8,
                hid_chans * 2,
                (3, 3),
                (1, 2),
                (1, 0),
                activation=activation,
                upsample=True,
            )
        )
        self.decoder.append(
            Conv2DActNorm(
                hid_chans * 4,
                hid_chans,
                (3, 3),
                (1, 2),
                (1, 0),
                activation=activation,
                upsample=True,
            )
        )

        for dec_indx in range(len(freq_axis_dims)):
            c_num_freqs = freq_axis_dims[len(freq_axis_dims) - dec_indx - 1]
            denseblocks = DenseBlock(
                hid_chans * 2,
                hid_chans * 2,
                c_num_freqs,
                ksz=ksz_dense,
                activation=activation,
                hid_chans=hid_chans_dense,
            )
            upsample = Conv2DActNorm(
                hid_chans * 2,
                hid_chans,
                (3, 3),
                (1, 2),
                (1, 0),
                activation=activation,
                upsample=True,
            )
            c_layer = torch.nn.Sequential(denseblocks, upsample)
            self.decoder.append(c_layer)

        last = torch.nn.Sequential(
            DenseBlock(
                hid_chans * 2,
                hid_chans * 2,
                self.in_channels - 2,
                ksz=ksz_dense,
                activation=activation,
                hid_chans=hid_chans_dense,
            ),
            torch.nn.ConvTranspose2d(
                hid_chans * 2, 2 * self.n_spk, (3, 3), (1, 1), (1, 0)
            ),
        )
        self.decoder.append(last)

    def _get_depth(self, num_freq):
        n_layers = 0
        freqs = []
        while num_freq > 15:
            num_freq = int(num_freq / 2)
            freqs.append(num_freq)
            n_layers += 1
        return freqs

    def forward(self, tf_rep):
        """
        Forward pass through the TCNDenseUNet model.

        Args:
            tf_rep (torch.Tensor): A 4D tensor representing the multi-channel
                complex Short-Time Fourier Transform (STFT) of the mixture.
                The shape of the tensor should be [B, T, C, F], where:
                - B is the batch size,
                - T is the number of frames,
                - C is the number of microphone channels,
                - F is the number of frequencies.

        Returns:
            out (torch.Tensor): A complex 3D tensor representing the monaural
                STFT of the targets. The shape of the output tensor is 
                [B, T, F], where:
                - B is the batch size,
                - T is the number of frames,
                - F is the number of frequencies.

        Examples:
            >>> model = TCNDenseUNet(n_spk=2, in_freqs=257, mic_channels=1)
            >>> input_tensor = torch.randn(8, 100, 1, 257)  # Example input
            >>> output = model(input_tensor)
            >>> print(output.shape)  # Output shape: [8, 100, 257]

        Note:
            The input tensor should be formatted correctly to ensure proper
            functioning of the model. Ensure that the number of microphone
            channels matches the expected input shape.

        Raises:
            AssertionError: If the number of microphone channels in the input
            tensor does not match the expected number of microphone channels.
        """
        # B, T, C, F
        tf_rep = tf_rep.permute(0, 2, 3, 1)
        bsz, mics, _, frames = tf_rep.shape
        assert mics == self.mic_channels

        inp_feats = torch.cat((tf_rep.real, tf_rep.imag), 1)
        inp_feats = inp_feats.transpose(-1, -2)
        inp_feats = inp_feats.reshape(
            bsz, self.mic_channels * 2, frames, self.in_channels
        )

        enc_out = []
        buffer = inp_feats
        for enc_layer in self.encoder:
            buffer = enc_layer(buffer)
            enc_out.append(buffer)

        assert buffer.shape[-1] == 1
        tcn_out = self.tcn(buffer.squeeze(-1)).unsqueeze(-1)

        buffer = tcn_out
        for indx, dec_layer in enumerate(self.decoder):
            c_input = torch.cat((buffer, enc_out[-(indx + 1)]), 1)
            buffer = dec_layer(c_input)

        buffer = buffer.reshape(bsz, 2, self.n_spk, -1, self.in_channels)

        if is_torch_1_9_plus:
            out = torch.complex(buffer[:, 0], buffer[:, 1])
        else:
            out = ComplexTensor(buffer[:, 0], buffer[:, 1])
        # bsz, complex_chans, frames or bsz, spk, complex_chans, frames
        return out  # bsz, spk, time, freq -> bsz, time, spk, freq
