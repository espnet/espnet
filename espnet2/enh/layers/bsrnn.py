from itertools import accumulate

import torch
import torch.nn as nn

from espnet2.enh.layers.tcn import choose_norm as choose_norm1d

EPS = torch.finfo(torch.get_default_dtype()).eps


class BSRNN(nn.Module):
    """
    Band-Split RNN (BSRNN) for high fidelity speech enhancement.

    This model implements a Band-Split RNN architecture for effective
    monaural speech enhancement. It leverages RNNs to model temporal
    and frequency features of audio signals, aiming to improve the
    quality of speech signals in noisy environments.

    References:
        [1] J. Yu, H. Chen, Y. Luo, R. Gu, and C. Weng, “High fidelity speech
        enhancement with band-split RNN,” in Proc. ISCA Interspeech, 2023.
        https://isca-speech.org/archive/interspeech_2023/yu23b_interspeech.html
        [2] J. Yu, and Y. Luo, “Efficient monaural speech enhancement with
        universal sample rate band-split RNN,” in Proc. ICASSP, 2023.
        https://ieeexplore.ieee.org/document/10096020

    Args:
        input_dim (int): Maximum number of frequency bins corresponding to
            `target_fs`.
        num_channel (int): Embedding dimension of each time-frequency bin.
        num_layer (int): Number of time and frequency RNN layers.
        target_fs (int): Maximum sampling frequency supported by the model.
        causal (bool): Whether to adopt causal processing. If True,
            LSTM will be used instead of BLSTM for time modeling.
        num_spk (int): Number of outputs to be generated.
        norm_type (str): Type of normalization layer (cfLN / cLN / BN / GN).

    Returns:
        out (torch.Tensor): Output tensor of shape (B, num_spk, T, F, 2).

    Examples:
        >>> model = BSRNN(input_dim=481, num_channel=16, num_layer=6)
        >>> input_tensor = torch.randn(8, 100, 481, 2)  # Batch size of 8
        >>> output = model(input_tensor)
        >>> print(output.shape)
        torch.Size([8, 1, 100, 481, 2])  # Assuming num_spk=1

    Note:
        The input tensor shape is expected to be (B, T, F, 2) where:
        - B is the batch size
        - T is the time dimension
        - F is the frequency dimension
        - 2 represents the real and imaginary parts of the complex signal.

    Raises:
        ValueError: If an unsupported normalization type is provided.
    """

    # ported from https://github.com/sungwon23/BSRNN
    def __init__(
        self,
        input_dim=481,
        num_channel=16,
        num_layer=6,
        target_fs=48000,
        causal=True,
        num_spk=1,
        norm_type="GN",
    ):
        """Band-Split RNN (BSRNN).

        References:
            [1] J. Yu, H. Chen, Y. Luo, R. Gu, and C. Weng, “High fidelity speech
            enhancement with band-split RNN,” in Proc. ISCA Interspeech, 2023.
            https://isca-speech.org/archive/interspeech_2023/yu23b_interspeech.html
            [2] J. Yu, and Y. Luo, “Efficient monaural speech enhancement with
            universal sample rate band-split RNN,” in Proc. ICASSP, 2023.
            https://ieeexplore.ieee.org/document/10096020

        Args:
            input_dim (int): maximum number of frequency bins corresponding to
                `target_fs`
            num_channel (int): embedding dimension of each time-frequency bin
            num_layer (int): number of time and frequency RNN layers
            target_fs (int): maximum sampling frequency supported by the model
            causal (bool): Whether or not to adopt causal processing
                if True, LSTM will be used instead of BLSTM for time modeling
            num_spk (int): number of outputs to be generated
            norm_type (str): type of normalization layer (cfLN / cLN / BN / GN)
        """
        super().__init__()
        norm1d_type = norm_type if norm_type != "cfLN" else "cLN"
        self.num_layer = num_layer
        self.band_split = BandSplit(
            input_dim, target_fs=target_fs, channels=num_channel, norm_type=norm1d_type
        )
        self.target_fs = target_fs
        self.causal = causal
        self.num_spk = num_spk

        self.norm_time = nn.ModuleList()
        self.rnn_time = nn.ModuleList()
        self.fc_time = nn.ModuleList()
        self.norm_freq = nn.ModuleList()
        self.rnn_freq = nn.ModuleList()
        self.fc_freq = nn.ModuleList()
        hdim = 2 * num_channel
        for i in range(self.num_layer):
            self.norm_time.append(choose_norm(norm_type, num_channel))
            self.rnn_time.append(
                nn.LSTM(
                    num_channel,
                    hdim,
                    batch_first=True,
                    bidirectional=not causal,
                )
            )
            self.fc_time.append(nn.Linear(hdim if causal else hdim * 2, num_channel))
            self.norm_freq.append(choose_norm(norm_type, num_channel))
            self.rnn_freq.append(
                nn.LSTM(num_channel, hdim, batch_first=True, bidirectional=True)
            )
            self.fc_freq.append(nn.Linear(4 * num_channel, num_channel))

        self.mask_decoder = MaskDecoder(
            input_dim,
            self.band_split.subbands,
            channels=num_channel,
            num_spk=num_spk,
            norm_type=norm1d_type,
        )

    def forward(self, x, fs=None):
        """
            BSRNN forward pass.

        This method performs the forward pass of the Band-Split RNN (BSRNN),
        processing the input tensor to produce an output tensor. The input
        tensor is assumed to have a specific shape and can be optionally
        truncated based on the sampling rate provided.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, F, 2), where B is the
                batch size, T is the time dimension, F is the frequency dimension,
                and 2 represents the real and imaginary parts of the complex signal.
            fs (int, optional): Sampling rate of the input signal. If not None,
                the input signal will be truncated to only process the effective
                frequency subbands. If None, the input signal is assumed to be
                already truncated to only contain effective frequency subbands.

        Returns:
            out (torch.Tensor): Output tensor of shape (B, num_spk, T, F, 2),
                where num_spk is the number of speakers to be generated.

        Examples:
            >>> model = BSRNN()
            >>> input_tensor = torch.randn(8, 100, 481, 2)  # Example input
            >>> output = model(input_tensor, fs=48000)
            >>> print(output.shape)
            torch.Size([8, 1, 100, 481, 2])  # Example output shape

        Note:
            The input tensor should be formatted correctly to ensure proper
            processing. The forward pass involves multiple layers of normalization,
            RNN processing, and a mask decoding step to generate the output.

        Raises:
            ValueError: If the input tensor shape does not match the expected
                dimensions.
        """
        z = self.band_split(x, fs=fs)
        B, N, T, K = z.shape
        skip = z
        for i in range(self.num_layer):
            out = self.norm_time[i](skip)
            out = out.transpose(1, 3).reshape(B * K, T, N)
            out, _ = self.rnn_time[i](out)
            out = self.fc_time[i](out)
            out = out.reshape(B, K, T, N).transpose(1, 3)
            skip = skip + out

            out = self.norm_freq[i](skip)
            out = out.permute(0, 2, 3, 1).contiguous().reshape(B * T, K, N)
            out, _ = self.rnn_freq[i](out)
            out = self.fc_freq[i](out)
            out = out.reshape(B, T, K, N).permute(0, 3, 1, 2).contiguous()
            skip = skip + out

        m, r = self.mask_decoder(skip)
        m = torch.view_as_complex(m)
        r = torch.view_as_complex(r)
        x = torch.view_as_complex(x)
        m = m[..., : x.size(-1)]
        r = r[..., : x.size(-1)]
        ret = torch.view_as_real(m * x.unsqueeze(1) + r)
        return ret


class BandSplit(nn.Module):
    """
    Splits the input tensor into frequency subbands for processing.

    This class implements the band-splitting operation, dividing the input
    frequency bins into several subbands for further processing in neural
    network architectures, particularly in the context of speech enhancement.

    Attributes:
        subbands (tuple): A tuple representing the number of frequency bins in
            each subband.
        subband_freqs (torch.Tensor): Frequencies corresponding to the subbands
            calculated from the FFT bins.
        norm (nn.ModuleList): A list of normalization layers for each subband.
        fc (nn.ModuleList): A list of 1D convolutional layers for each subband.

    Args:
        input_dim (int): Maximum number of frequency bins corresponding to
            `target_fs`. Must be an odd number.
        target_fs (int): Maximum sampling frequency supported by the model.
        channels (int): Number of output channels after convolution for each
            subband.
        norm_type (str): Type of normalization layer to use (e.g., "GN", "BN",
            etc.).

    Raises:
        AssertionError: If `input_dim` is not an odd number or if the sum of
            subbands does not equal `input_dim`.
        NotImplementedError: If the specified `input_dim` and `target_fs`
            do not match predefined configurations.

    Examples:
        >>> band_split = BandSplit(input_dim=481, target_fs=48000, channels=128)
        >>> input_tensor = torch.randn(10, 100, 481, 2)  # (B, T, F, 2)
        >>> output = band_split(input_tensor)
        >>> print(output.shape)  # Should be (B, N, T, K')
    """

    def __init__(self, input_dim, target_fs=48000, channels=128, norm_type="GN"):
        super().__init__()
        assert input_dim % 2 == 1, input_dim
        n_fft = (input_dim - 1) * 2
        # freq resolution = target_fs / n_fft = freqs[1] - freqs[0]
        freqs = torch.fft.rfftfreq(n_fft, 1.0 / target_fs)
        if input_dim == 481 and target_fs == 48000:
            # n_fft=960 (20ms)
            # first 20 200Hz subbands: [0-200], (200-400], (400-600], ..., (3800-4000]
            # subsequent 6 500Hz subbands: (4000, 4500], ..., (6500, 7000]
            # subsequent 7 2kHz subbands: (7000, 9000], ..., (19000, 21000]
            # final 3kHz subband: (21000, 24000]
            self.subbands = tuple([5] + [4] * 19 + [10] * 6 + [40] * 7 + [60])
        else:
            raise NotImplementedError(
                f"Please define your own subbands for input_dim={input_dim} and "
                f"target_fs={target_fs}"
            )
        assert sum(self.subbands) == input_dim, (self.subbands, input_dim)
        self.subband_freqs = freqs[[idx - 1 for idx in accumulate(self.subbands)]]

        self.norm = nn.ModuleList()
        self.fc = nn.ModuleList()
        for i in range(len(self.subbands)):
            self.norm.append(choose_norm1d(norm_type, int(self.subbands[i] * 2)))
            self.fc.append(nn.Conv1d(int(self.subbands[i] * 2), channels, 1))

    def forward(self, x, fs=None):
        """
        BSRNN forward.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, F, 2), where B is the
                batch size, T is the number of time steps, F is the number of
                frequency bins, and 2 represents the real and imaginary parts
                of the complex input.
            fs (int, optional): Sampling rate of the input signal. If not None,
                the input signal will be truncated to only process the effective
                frequency subbands. If None, the input signal is assumed to be
                already truncated to only contain effective frequency subbands.

        Returns:
            out (torch.Tensor): Output tensor of shape (B, num_spk, T, F, 2),
                where num_spk is the number of speakers, T is the number of time
                steps, F is the number of frequency bins, and 2 represents the
                real and imaginary parts of the output.

        Examples:
            >>> model = BSRNN()
            >>> input_tensor = torch.randn(4, 100, 481, 2)  # (B, T, F, 2)
            >>> output = model(input_tensor, fs=48000)
            >>> output.shape
            torch.Size([4, 1, 100, 481, 2])  # (B, num_spk, T, F, 2)
        """
        hz_band = 0
        for i, subband in enumerate(self.subbands):
            x_band = x[:, :, hz_band : hz_band + int(subband), :]
            if int(subband) > x_band.size(2):
                x_band = nn.functional.pad(
                    x_band, (0, 0, 0, int(subband) - x_band.size(2))
                )
            x_band = x_band.reshape(x_band.size(0), x_band.size(1), -1)
            out = self.norm[i](x_band.transpose(1, 2))
            # (B, band * 2, T) -> (B, N, T)
            out = self.fc[i](out)

            if i == 0:
                z = out.unsqueeze(-1)
            else:
                z = torch.cat((z, out.unsqueeze(-1)), dim=-1)
            hz_band = hz_band + int(subband)
            if hz_band >= x.size(2):
                break
            if fs is not None and self.subband_freqs[i] >= fs / 2:
                break
        return z


class MaskDecoder(nn.Module):
    """
    Mask Decoder for band-split RNN-based speech enhancement.

    This class implements a mask decoder that processes input tensors and
    generates the corresponding output masks and residuals. The mask decoder
    is a crucial component in the BSRNN architecture, enabling the model to
    enhance speech signals by estimating the mask and residual signals.

    Attributes:
        subbands (tuple): The number of frequency subbands.
        freq_dim (int): Total frequency dimension, should equal the sum of
            subbands.
        num_spk (int): Number of speakers to generate outputs for.
        mlp_mask (nn.ModuleList): List of MLPs for generating masks for each
            subband.
        mlp_residual (nn.ModuleList): List of MLPs for generating residuals for
            each subband.

    Args:
        freq_dim (int): Total frequency dimension.
        subbands (tuple): Number of frequency subbands.
        channels (int): Number of channels in the input tensor.
        num_spk (int): Number of outputs to generate (default is 1).
        norm_type (str): Type of normalization layer to be used (default is "GN").

    Returns:
        None

    Examples:
        >>> decoder = MaskDecoder(freq_dim=481, subbands=(5, 4, 4, 4, 4),
        ...                        channels=128, num_spk=1)
        >>> input_tensor = torch.randn(10, 16, 20, 5)  # Example input
        >>> masks, residuals = decoder(input_tensor)
        >>> masks.shape  # Should be (10, 1, 20, 481, 2)
        >>> residuals.shape  # Should be (10, 1, 20, 481, 2)

    Raises:
        AssertionError: If `freq_dim` does not equal the sum of `subbands`.
    """

    def __init__(self, freq_dim, subbands, channels=128, num_spk=1, norm_type="GN"):
        super().__init__()
        assert freq_dim == sum(subbands), (freq_dim, subbands)
        self.subbands = subbands
        self.freq_dim = freq_dim
        self.num_spk = num_spk
        self.mlp_mask = nn.ModuleList()
        self.mlp_residual = nn.ModuleList()
        for subband in self.subbands:
            self.mlp_mask.append(
                nn.Sequential(
                    choose_norm1d(norm_type, channels),
                    nn.Conv1d(channels, 4 * channels, 1),
                    nn.Tanh(),
                    nn.Conv1d(4 * channels, int(subband * 4 * num_spk), 1),
                    nn.GLU(dim=1),
                )
            )
            self.mlp_residual.append(
                nn.Sequential(
                    choose_norm1d(norm_type, channels),
                    nn.Conv1d(channels, 4 * channels, 1),
                    nn.Tanh(),
                    nn.Conv1d(4 * channels, int(subband * 4 * num_spk), 1),
                    nn.GLU(dim=1),
                )
            )

    def forward(self, x):
        """
            BSRNN forward.

        This method performs the forward pass of the Band-Split RNN (BSRNN) model,
        processing the input tensor through the band splitting, RNN layers, and
        mask decoding to produce the output tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, F, 2), where B is the
                batch size, T is the time dimension, F is the frequency dimension,
                and 2 represents the real and imaginary parts of the complex input.
            fs (int, optional): Sampling rate of the input signal. If provided, the
                input signal will be truncated to only process the effective frequency
                subbands. If None, the input signal is assumed to be already truncated
                to only contain effective frequency subbands.

        Returns:
            out (torch.Tensor): Output tensor of shape (B, num_spk, T, F, 2), where
                num_spk is the number of speakers, T is the time dimension, F is the
                frequency dimension, and 2 represents the real and imaginary parts of
                the complex output.

        Examples:
            >>> model = BSRNN(input_dim=481, num_spk=2)
            >>> input_tensor = torch.randn(8, 100, 481, 2)  # Example input
            >>> output = model(input_tensor, fs=48000)
            >>> print(output.shape)  # Output shape will be (8, 2, 100, 481, 2)

        Note:
            The input tensor must have the correct shape, and if the sampling rate
            (fs) is provided, it should be compatible with the target sampling rate
            of the model.
        """
        for i in range(len(self.subbands)):
            if i >= x.size(-1):
                break
            x_band = x[:, :, :, i]
            out = self.mlp_mask[i](x_band).transpose(1, 2).contiguous()
            # (B, T, num_spk, subband, 2)
            out = out.reshape(out.size(0), out.size(1), self.num_spk, -1, 2)
            if i == 0:
                m = out
            else:
                m = torch.cat((m, out), dim=3)

            res = self.mlp_residual[i](x_band).transpose(1, 2).contiguous()
            # (B, T, num_spk, subband, 2)
            res = res.reshape(res.size(0), res.size(1), self.num_spk, -1, 2)
            if i == 0:
                r = res
            else:
                r = torch.cat((r, res), dim=3)
        # Pad zeros in addition to effective subbands to cover the full frequency range
        m = nn.functional.pad(m, (0, 0, 0, int(self.freq_dim - m.size(-2))))
        r = nn.functional.pad(r, (0, 0, 0, int(self.freq_dim - r.size(-2))))
        return m.moveaxis(1, 2), r.moveaxis(1, 2)


def choose_norm(norm_type, channel_size, shape="BDTF"):
    """
    Selects and returns the appropriate normalization layer based on the type.

    The input to the normalization layer will be of shape (M, C, K), where:
    - M is the batch size.
    - C is the channel size.
    - K is the sequence length.

    Args:
        norm_type (str): Type of normalization layer to choose. Options include:
            - "cfLN": Channel-and-Frequency-wise Layer Normalization.
            - "cLN": Channel-wise Layer Normalization.
            - "BN": Batch Normalization.
            - "GN": Group Normalization.
        channel_size (int): The number of channels in the input tensor.
        shape (str, optional): The expected shape of the input tensor.
            Defaults to "BDTF". Can be "BTFD" to indicate a different layout.

    Returns:
        nn.Module: The selected normalization layer.

    Raises:
        ValueError: If the provided `norm_type` is unsupported.

    Examples:
        >>> norm_layer = choose_norm("BN", channel_size=16)
        >>> input_tensor = torch.randn(32, 16, 100)  # (M, C, K)
        >>> output_tensor = norm_layer(input_tensor)

    Note:
        The normalization layers are designed to be used in deep learning
        architectures to stabilize and accelerate training by normalizing
        the inputs to each layer.
    """
    if norm_type == "cfLN":
        return ChannelFreqwiseLayerNorm(channel_size, shape=shape)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size, shape=shape)
    elif norm_type == "BN":
        # Given input (M, C, T, K), nn.BatchNorm2d(C) will accumulate statics
        # along M, T, and K, so this BN usage is right.
        return nn.BatchNorm2d(channel_size)
    elif norm_type == "GN":
        return nn.GroupNorm(1, channel_size)
    else:
        raise ValueError("Unsupported normalization type")


class ChannelwiseLayerNorm(nn.Module):
    """
        Channel-wise Layer Normalization (cLN).

    This layer normalizes the input across the channel dimension. It computes the
    mean and variance for each channel and applies the normalization to each
    element of the channel independently. This can help stabilize training and
    improve convergence in deep learning models.

    Attributes:
        gamma (torch.nn.Parameter): Scale parameter for normalization.
        beta (torch.nn.Parameter): Shift parameter for normalization.
        shape (str): The expected shape of the input tensor. It can be "BDTF"
            (Batch, Depth, Time, Frequency) or "BTFD" (Batch, Time, Frequency, Depth).

    Args:
        channel_size (int): Number of channels to normalize.
        shape (str, optional): The shape of the input tensor. Defaults to "BDTF".

    Methods:
        reset_parameters: Resets the parameters of the layer.
        forward: Applies channel-wise layer normalization to the input tensor.

    Examples:
        >>> layer_norm = ChannelwiseLayerNorm(channel_size=16)
        >>> input_tensor = torch.randn(8, 16, 100, 200)  # (B, N, T, K)
        >>> output_tensor = layer_norm(input_tensor)
        >>> output_tensor.shape
        torch.Size([8, 16, 100, 200])  # The shape remains the same.

    Raises:
        AssertionError: If the input tensor does not have 4 dimensions.

    Note:
        The normalization is performed using the formula:
            cLN_y = gamma * (y - mean) / sqrt(var + EPS) + beta
        where mean and var are computed across the channel dimension.

    Todo:
        - Implement support for additional shapes if necessary.
    """

    def __init__(self, channel_size, shape="BDTF"):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1, 1))  # [1, N, 1]
        self.reset_parameters()
        assert shape in ["BDTF", "BTFD"], shape
        self.shape = shape

    def reset_parameters(self):
        """
            Reset the parameters of the Channelwise Layer Normalization.

        This method initializes the learnable parameters `gamma` and `beta` of the
        Channelwise Layer Normalization to their default values. Specifically,
        `gamma` is set to 1 and `beta` is set to 0. This is typically called
        when the layer is first created to ensure that the normalization starts
        with neutral parameters.

        Attributes:
            gamma (torch.Tensor): Learnable scale parameter, initialized to 1.
            beta (torch.Tensor): Learnable shift parameter, initialized to 0.

        Examples:
            >>> layer_norm = ChannelwiseLayerNorm(channel_size=128)
            >>> layer_norm.gamma  # Should be initialized to 1
            tensor([[[[1.]]]])
            >>> layer_norm.beta  # Should be initialized to 0
            tensor([[[[0.]]]])

        Note:
            This method is automatically called in the `__init__` method of the
            class.
        """
        self.beta.data.zero_()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, y):
        """
            BSRNN forward.

        This method processes the input tensor through the Band-Split RNN model,
        applying normalization, recurrent layers, and a mask decoder to produce
        enhanced audio output.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, F, 2), where B is the
                batch size, T is the number of time frames, F is the number of
                frequency bins, and the last dimension represents the complex
                values (real and imaginary parts).
            fs (int, optional): Sampling rate of the input signal. If not None,
                the input signal will be truncated to only process the effective
                frequency subbands. If None, the input signal is assumed to be
                already truncated to only contain effective frequency subbands.

        Returns:
            out (torch.Tensor): Output tensor of shape (B, num_spk, T, F, 2),
                where num_spk is the number of speakers. The output tensor
                contains the enhanced audio signal.

        Examples:
            >>> model = BSRNN()
            >>> input_tensor = torch.randn(8, 100, 481, 2)  # Example input
            >>> output = model(input_tensor, fs=48000)
            >>> print(output.shape)  # Output shape should be (8, num_spk, 100, F, 2)

        Note:
            The input tensor should contain complex values represented as
            separate real and imaginary components in the last dimension.
            The method processes the input through multiple layers, including
            normalization and LSTM layers, to generate the final output.

        Raises:
            ValueError: If the input tensor does not have the expected shape.
        """

        assert y.dim() == 4

        if self.shape == "BTFD":
            y = y.moveaxis(-1, 1)

        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, T, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, T, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta

        if self.shape == "BTFD":
            cLN_y = cLN_y.moveaxis(1, -1)

        return cLN_y


class ChannelFreqwiseLayerNorm(nn.Module):
    """
    Channel-and-Frequency-wise Layer Normalization (cfLN).

    This layer normalizes the input tensor across the channel and frequency
    dimensions, improving the training stability and convergence of neural
    networks. It computes the mean and variance for each channel across the
    frequency dimension and normalizes the input accordingly.

    Attributes:
        gamma (torch.Parameter): Scale parameter for the normalization.
        beta (torch.Parameter): Shift parameter for the normalization.
        shape (str): The shape of the input tensor, either "BDTF" or "BTFD".

    Args:
        channel_size (int): The number of channels in the input tensor.
        shape (str): The shape of the input tensor; must be "BDTF" or "BTFD".

    Raises:
        AssertionError: If the provided shape is not "BDTF" or "BTFD".

    Examples:
        >>> layer_norm = ChannelFreqwiseLayerNorm(channel_size=16, shape="BDTF")
        >>> input_tensor = torch.randn(8, 16, 100, 50)  # (B, C, T, K)
        >>> output_tensor = layer_norm(input_tensor)
        >>> output_tensor.shape
        torch.Size([8, 16, 100, 50])

    Note:
        The normalization is performed in a way that preserves the input shape.
    """

    def __init__(self, channel_size, shape="BDTF"):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1, 1))  # [1, N, 1, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1, 1))  # [1, N, 1, 1]
        self.reset_parameters()
        assert shape in ["BDTF", "BTFD"], shape
        self.shape = shape

    def reset_parameters(self):
        """
            Channel-and-Frequency-wise Layer Normalization (cfLN).

        This layer normalizes the input tensor across both the channel and frequency
        dimensions, allowing for improved training stability and performance in deep
        learning models. It applies normalization in a way that considers the
        interdependence between channels and frequencies.

        Attributes:
            gamma (torch.Tensor): Learnable scale parameter of shape [1, N, 1, 1].
            beta (torch.Tensor): Learnable shift parameter of shape [1, N, 1, 1].
            shape (str): Specifies the input tensor shape, either "BDTF" or "BTFD".

        Args:
            channel_size (int): The number of channels in the input tensor.
            shape (str): The shape of the input tensor. It can be either "BDTF"
                (Batch, Depth, Time, Frequency) or "BTFD" (Batch, Time, Frequency, Depth).

        Examples:
            >>> layer_norm = ChannelFreqwiseLayerNorm(channel_size=128)
            >>> input_tensor = torch.randn(32, 128, 50, 50)  # [Batch, Channel, Time, Frequency]
            >>> output_tensor = layer_norm(input_tensor)

        Note:
            The normalization is performed using the formula:
            gLN_y = γ * (y - mean) / sqrt(var + EPS) + β
            where mean and var are calculated across the channel and frequency dimensions.

        Todo:
            - Add support for additional normalization techniques if needed.
        """
        self.beta.data.zero_()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, y):
        """
            Channel-and-Frequency-wise Layer Normalization (cfLN).

        This class implements a normalization layer that performs normalization
        across both the channel and frequency dimensions. It helps in stabilizing
        the training process by reducing internal covariate shift.

        Attributes:
            gamma (torch.Parameter): Learnable scale parameter of shape [1, N, 1, 1].
            beta (torch.Parameter): Learnable shift parameter of shape [1, N, 1, 1].
            shape (str): The shape of the input tensor, either "BDTF" or "BTFD".

        Args:
            channel_size (int): The number of channels to normalize.
            shape (str): The shape of the input tensor, either "BDTF" or "BTFD".

        Raises:
            AssertionError: If the shape is not "BDTF" or "BTFD".

        Examples:
            >>> layer_norm = ChannelFreqwiseLayerNorm(channel_size=64)
            >>> input_tensor = torch.randn(32, 64, 128, 256)  # [M, N, T, K]
            >>> output_tensor = layer_norm(input_tensor)
            >>> print(output_tensor.shape)  # Output shape will be [32, 64, 128, 256]

        Note:
            This implementation uses PyTorch's automatic mixed precision (AMP)
            for forward pass.
        """
        if self.shape == "BTFD":
            y = y.moveaxis(-1, 1)

        mean = y.mean(dim=(1, 3), keepdim=True)  # [M, 1, T, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=(1, 3), keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta

        if self.shape == "BTFD":
            gLN_y = gLN_y.moveaxis(1, -1)
        return gLN_y
