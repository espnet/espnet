import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

from espnet2.enh.decoder.stft_decoder import STFTDecoder
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.layers.complex_utils import new_complex_like
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.torch_utils.get_layer_from_string import get_layer


class TFGridNetV2(AbsSeparator):
    """
    Offline TFGridNetV2 for speech separation.

    Compared to TFGridNet, TFGridNetV2 enhances performance by vectorizing 
    multiple heads in self-attention and improving the handling of Deconv1D 
    in each intra- and inter-block when `emb_ks` equals `emb_hs`.

    References:
        [1] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
        "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech 
        Separation", in TASLP, 2023.
        
        [2] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
        "TF-GridNet: Making Time-Frequency Domain Models Great Again for 
        Monaural Speaker Separation", in ICASSP, 2023.

    Note:
        This model performs optimally when trained with variance-normalized 
        mixture inputs and targets. For a mixture tensor of shape 
        [batch, samples, microphones], normalize it by dividing with 
        `torch.std(mixture, (1, 2))`. Apply the same normalization to 
        the target signals. This is particularly important when not using 
        scale-invariant loss functions such as SI-SDR. Specifically, use:
        
            std_ = std(mix)
            mix = mix / std_
            tgt = tgt / std_

    Args:
        input_dim (int): Placeholder, not used.
        n_srcs (int): Number of output sources/speakers.
        n_fft (int): STFT window size.
        stride (int): STFT stride.
        window (str): STFT window type; choose between 'hamming', 'hanning', or None.
        n_imics (int): Number of microphone channels (only fixed-array geometry supported).
        n_layers (int): Number of TFGridNetV2 blocks.
        lstm_hidden_units (int): Number of hidden units in LSTM.
        attn_n_head (int): Number of heads in self-attention.
        attn_approx_qk_dim (int): Approximate dimension of frame-level key and value tensors.
        emb_dim (int): Embedding dimension.
        emb_ks (int): Kernel size for unfolding and Deconv1D.
        emb_hs (int): Hop size for unfolding and Deconv1D.
        activation (str): Activation function to use in the entire TFGridNetV2 model.
            Can use any torch-supported activation (e.g., 'relu' or 'elu').
        eps (float): Small epsilon for normalization layers.
        use_builtin_complex (bool): Whether to use built-in complex type or not.

    Examples:
        >>> model = TFGridNetV2(n_srcs=2, n_fft=256)
        >>> input_tensor = torch.randn(8, 512, 1)  # [B, N, M]
        >>> ilens = torch.tensor([512] * 8)  # input lengths
        >>> enhanced, lengths, _ = model(input_tensor, ilens)
    """

    def __init__(
        self,
        input_dim,
        n_srcs=2,
        n_fft=128,
        stride=64,
        window="hann",
        n_imics=1,
        n_layers=6,
        lstm_hidden_units=192,
        attn_n_head=4,
        attn_approx_qk_dim=512,
        emb_dim=48,
        emb_ks=4,
        emb_hs=1,
        activation="prelu",
        eps=1.0e-5,
        use_builtin_complex=False,
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_imics = n_imics
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1

        self.enc = STFTEncoder(
            n_fft, n_fft, stride, window=window, use_builtin_complex=use_builtin_complex
        )
        self.dec = STFTDecoder(n_fft, n_fft, stride, window=window)

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * n_imics, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )

        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                GridNetV2Block(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    n_freqs,
                    lstm_hidden_units,
                    n_head=attn_n_head,
                    approx_qk_dim=attn_approx_qk_dim,
                    activation=activation,
                    eps=eps,
                )
            )

        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=padding)

    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
        """
        Offline TFGridNetV2 for speech separation.

    This model improves upon TFGridNet by vectorizing multiple heads in 
    self-attention and enhancing the handling of Deconv1D operations when 
    `emb_ks` equals `emb_hs`.

    References:
        [1] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and 
            S. Watanabe, "TF-GridNet: Integrating Full- and Sub-Band 
            Modeling for Speech Separation", in TASLP, 2023.
        [2] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and 
            S. Watanabe, "TF-GridNet: Making Time-Frequency Domain Models 
            Great Again for Monaural Speaker Separation", in ICASSP, 2023.

    Notes:
        For optimal performance, train this model with variance normalized 
        mixture inputs and targets. For a mixture of shape [batch, samples, 
        microphones], normalize it by dividing by 
        `torch.std(mixture, (1, 2))`. This normalization should also be 
        applied to target signals. It is particularly recommended when not 
        using scale-invariant loss functions such as SI-SDR. The 
        normalization steps are as follows:
            std_ = std(mix)
            mix = mix / std_
            tgt = tgt / std_

    Args:
        input_dim (int): Placeholder, not used.
        n_srcs (int): Number of output sources/speakers.
        n_fft (int): STFT window size.
        stride (int): STFT stride.
        window (str): STFT window type; options are 'hamming', 'hanning', or None.
        n_imics (int): Number of microphone channels (only fixed-array geometry 
            supported).
        n_layers (int): Number of TFGridNetV2 blocks.
        lstm_hidden_units (int): Number of hidden units in LSTM.
        attn_n_head (int): Number of heads in self-attention.
        attn_approx_qk_dim (int): Approximate dimension of frame-level key and 
            value tensors.
        emb_dim (int): Embedding dimension.
        emb_ks (int): Kernel size for unfolding and Deconv1D.
        emb_hs (int): Hop size for unfolding and Deconv1D.
        activation (str): Activation function to use in the model; can be any 
            torch-supported activation, e.g., 'relu' or 'elu'.
        eps (float): Small epsilon for normalization layers.
        use_builtin_complex (bool): Whether to use the built-in complex type.

    Examples:
        model = TFGridNetV2(n_srcs=2, n_fft=256)
        input_tensor = torch.randn(10, 256, 1)  # Batch of 10 samples
        ilens = torch.tensor([256] * 10)  # Input lengths
        enhanced, ilens, _ = model(input_tensor, ilens)
        """
        n_samples = input.shape[1]
        if self.n_imics == 1:
            assert len(input.shape) == 2
            input = input[..., None]  # [B, N, M]

        mix_std_ = torch.std(input, dim=(1, 2), keepdim=True)  # [B, 1, 1]
        input = input / mix_std_  # RMS normalization

        batch = self.enc(input, ilens)[0]  # [B, T, M, F]
        batch0 = batch.transpose(1, 2)  # [B, M, T, F]
        batch = torch.cat((batch0.real, batch0.imag), dim=1)  # [B, 2*M, T, F]
        n_batch, _, n_frames, n_freqs = batch.shape

        batch = self.conv(batch)  # [B, -1, T, F]

        for ii in range(self.n_layers):
            batch = self.blocks[ii](batch)  # [B, -1, T, F]

        batch = self.deconv(batch)  # [B, n_srcs*2, T, F]

        batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])
        batch = new_complex_like(batch0, (batch[:, :, 0], batch[:, :, 1]))

        batch = self.dec(batch.view(-1, n_frames, n_freqs), ilens)[0]  # [B, n_srcs, -1]

        batch = self.pad2(batch.view([n_batch, self.num_spk, -1]), n_samples)

        batch = batch * mix_std_  # reverse the RMS normalization

        batch = [batch[:, src] for src in range(self.num_spk)]

        return batch, ilens, OrderedDict()

    @property
    def num_spk(self):
        return self.n_srcs

    @staticmethod
    def pad2(input_tensor, target_len):
        """
        Offline TFGridNetV2.

    Compared with TFGridNet, TFGridNetV2 speeds up the code by vectorizing 
    multiple heads in self-attention and better handling Deconv1D in each 
    intra- and inter-block when emb_ks == emb_hs.

    References:
        [1] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
            "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech 
            Separation", in TASLP, 2023.
        [2] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
            "TF-GridNet: Making Time-Frequency Domain Models Great Again for 
            Monaural Speaker Separation", in ICASSP, 2023.

    Notes:
        As outlined in the References, this model works best when trained 
        with variance normalized mixture input and target, e.g., with mixture 
        of shape [batch, samples, microphones], you normalize it by dividing 
        with torch.std(mixture, (1, 2)). You must do the same for the target 
        signals. It is encouraged to do so when not using scale-invariant 
        loss functions such as SI-SDR. Specifically, use:
            std_ = std(mix)
            mix = mix / std_
            tgt = tgt / std_

    Args:
        input_dim (int): Placeholder, not used.
        n_srcs (int): Number of output sources/speakers.
        n_fft (int): STFT window size.
        stride (int): STFT stride.
        window (str): STFT window type; choose between 'hamming', 'hanning' or None.
        n_imics (int): Number of microphone channels (only fixed-array geometry supported).
        n_layers (int): Number of TFGridNetV2 blocks.
        lstm_hidden_units (int): Number of hidden units in LSTM.
        attn_n_head (int): Number of heads in self-attention.
        attn_approx_qk_dim (int): Approximate dimension of frame-level key 
            and value tensors.
        emb_dim (int): Embedding dimension.
        emb_ks (int): Kernel size for unfolding and deconv1D.
        emb_hs (int): Hop size for unfolding and deconv1D.
        activation (str): Activation function to use in the whole TFGridNetV2 
            model, e.g., 'relu' or 'elu'.
        eps (float): Small epsilon for normalization layers.
        use_builtin_complex (bool): Whether to use built-in complex type or not.

    Examples:
        >>> model = TFGridNetV2(n_srcs=2, n_fft=128)
        >>> input_tensor = torch.randn(10, 16000)  # Batch of 10 audio samples
        >>> ilens = torch.tensor([16000] * 10)  # Lengths of each sample
        >>> enhanced, ilens, _ = model(input_tensor, ilens)
        """
        input_tensor = torch.nn.functional.pad(
            input_tensor, (0, target_len - input_tensor.shape[-1])
        )
        return input_tensor


class GridNetV2Block(nn.Module):
    """
    GridNetV2Block is a neural network block used within the TFGridNetV2 model. It
is designed to process input features through intra- and inter-block recurrent
neural networks (RNNs) and multi-head self-attention mechanisms.

Attributes:
    emb_dim (int): The embedding dimension for input features.
    emb_ks (int): Kernel size for convolutions and unfolding.
    emb_hs (int): Hop size for convolutions and deconvolutions.
    n_head (int): Number of heads in the multi-head attention mechanism.

Args:
    emb_dim (int): Dimension of the input embeddings.
    emb_ks (int): Kernel size for unfolding and deconvolution.
    emb_hs (int): Hop size for unfolding and deconvolution.
    n_freqs (int): Number of frequency bins in the input features.
    hidden_channels (int): Number of hidden channels in the RNN.
    n_head (int, optional): Number of attention heads (default is 4).
    approx_qk_dim (int, optional): Approximate dimension for Q and K tensors 
        in attention (default is 512).
    activation (str, optional): Activation function to use (default is "prelu").
    eps (float, optional): Small value for numerical stability in normalization 
        layers (default is 1e-5).

Examples:
    # Create a GridNetV2Block instance
    block = GridNetV2Block(
        emb_dim=48,
        emb_ks=4,
        emb_hs=1,
        n_freqs=65,
        hidden_channels=192
    )

    # Forward pass through the block
    input_tensor = torch.randn(10, 48 * 4, 100, 50)  # Example input
    output_tensor = block(input_tensor)

Raises:
    AssertionError: If the specified activation function is not "prelu".
    """
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        n_freqs,
        hidden_channels,
        n_head=4,
        approx_qk_dim=512,
        activation="prelu",
        eps=1e-5,
    ):
        super().__init__()
        assert activation == "prelu"

        in_channels = emb_dim * emb_ks

        self.intra_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        if emb_ks == emb_hs:
            self.intra_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.intra_linear = nn.ConvTranspose1d(
                hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
            )

        self.inter_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        if emb_ks == emb_hs:
            self.inter_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.inter_linear = nn.ConvTranspose1d(
                hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
            )

        E = math.ceil(
            approx_qk_dim * 1.0 / n_freqs
        )  # approx_qk_dim is only approximate
        assert emb_dim % n_head == 0

        self.add_module("attn_conv_Q", nn.Conv2d(emb_dim, n_head * E, 1))
        self.add_module(
            "attn_norm_Q",
            AllHeadPReLULayerNormalization4DCF((n_head, E, n_freqs), eps=eps),
        )

        self.add_module("attn_conv_K", nn.Conv2d(emb_dim, n_head * E, 1))
        self.add_module(
            "attn_norm_K",
            AllHeadPReLULayerNormalization4DCF((n_head, E, n_freqs), eps=eps),
        )

        self.add_module(
            "attn_conv_V", nn.Conv2d(emb_dim, n_head * emb_dim // n_head, 1)
        )
        self.add_module(
            "attn_norm_V",
            AllHeadPReLULayerNormalization4DCF(
                (n_head, emb_dim // n_head, n_freqs), eps=eps
            ),
        )

        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                get_layer(activation)(),
                LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
            ),
        )

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x):
        """
        Forward pass of the TFGridNetV2 model.

        This method processes the input audio tensor through the TFGridNetV2 
        architecture, performing speech separation for the specified number of 
        sources. It applies normalization, convolution, and RNN operations, 
        followed by a transposed convolution to produce the output signals.

        Args:
            input (torch.Tensor): Batched multi-channel audio tensor with
                M audio channels and N samples of shape [B, N, M].
            ilens (torch.Tensor): Input lengths for each batch element of shape [B].
            additional (Dict or None): Additional data, currently unused in this model.

        Returns:
            enhanced (List[Union(torch.Tensor)]): 
                A list of length n_srcs containing mono audio tensors 
                of shape [(B, T), ...] where T is the number of samples.
            ilens (torch.Tensor): Input lengths of shape (B,).
            additional (OrderedDict): Additional data, currently unused in 
                this model, returned as part of the output.

        Examples:
            >>> model = TFGridNetV2(n_srcs=2)
            >>> input_tensor = torch.randn(8, 16000, 1)  # [B, N, M]
            >>> ilens = torch.tensor([16000] * 8)  # Lengths for each batch
            >>> enhanced, lengths, _ = model(input_tensor, ilens)

        Note:
            It is recommended to normalize the input and target signals 
            using RMS normalization as follows:
            std_ = torch.std(mix, (1, 2))
            mix = mix / std_
            tgt = tgt / std_

        Raises:
            ValueError: If the input tensor does not have the correct 
            dimensions or if n_imics is not equal to 1 when input has 
            two dimensions.
        """
        B, C, old_T, old_Q = x.shape

        olp = self.emb_ks - self.emb_hs
        T = (
            math.ceil((old_T + 2 * olp - self.emb_ks) / self.emb_hs) * self.emb_hs
            + self.emb_ks
        )
        Q = (
            math.ceil((old_Q + 2 * olp - self.emb_ks) / self.emb_hs) * self.emb_hs
            + self.emb_ks
        )

        x = x.permute(0, 2, 3, 1)  # [B, old_T, old_Q, C]
        x = F.pad(x, (0, 0, olp, Q - old_Q - olp, olp, T - old_T - olp))  # [B, T, Q, C]

        # intra RNN
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, T, Q, C]
        if self.emb_ks == self.emb_hs:
            intra_rnn = intra_rnn.view([B * T, -1, self.emb_ks * C])  # [BT, Q//I, I*C]
            intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, Q//I, H]
            intra_rnn = self.intra_linear(intra_rnn)  # [BT, Q//I, I*C]
            intra_rnn = intra_rnn.view([B, T, Q, C])
        else:
            intra_rnn = intra_rnn.view([B * T, Q, C])  # [BT, Q, C]
            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, C, Q]
            intra_rnn = F.unfold(
                intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
            )  # [BT, C*I, -1]
            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*I]

            intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]

            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
            intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
            intra_rnn = intra_rnn.view([B, T, C, Q])
            intra_rnn = intra_rnn.transpose(-2, -1)  # [B, T, Q, C]
        intra_rnn = intra_rnn + input_  # [B, T, Q, C]

        intra_rnn = intra_rnn.transpose(1, 2)  # [B, Q, T, C]

        # inter RNN
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)  # [B, Q, T, C]
        if self.emb_ks == self.emb_hs:
            inter_rnn = inter_rnn.view([B * Q, -1, self.emb_ks * C])  # [BQ, T//I, I*C]
            inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BQ, T//I, H]
            inter_rnn = self.inter_linear(inter_rnn)  # [BQ, T//I, I*C]
            inter_rnn = inter_rnn.view([B, Q, T, C])
        else:
            inter_rnn = inter_rnn.view(B * Q, T, C)  # [BQ, T, C]
            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, C, T]
            inter_rnn = F.unfold(
                inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
            )  # [BQ, C*I, -1]
            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, -1, C*I]

            inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BQ, -1, H]

            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, H, -1]
            inter_rnn = self.inter_linear(inter_rnn)  # [BQ, C, T]
            inter_rnn = inter_rnn.view([B, Q, C, T])
            inter_rnn = inter_rnn.transpose(-2, -1)  # [B, Q, T, C]
        inter_rnn = inter_rnn + input_  # [B, Q, T, C]

        inter_rnn = inter_rnn.permute(0, 3, 2, 1)  # [B, C, T, Q]

        inter_rnn = inter_rnn[..., olp : olp + old_T, olp : olp + old_Q]
        batch = inter_rnn

        Q = self["attn_norm_Q"](self["attn_conv_Q"](batch))  # [B, n_head, C, T, Q]
        K = self["attn_norm_K"](self["attn_conv_K"](batch))  # [B, n_head, C, T, Q]
        V = self["attn_norm_V"](self["attn_conv_V"](batch))  # [B, n_head, C, T, Q]
        Q = Q.view(-1, *Q.shape[2:])  # [B*n_head, C, T, Q]
        K = K.view(-1, *K.shape[2:])  # [B*n_head, C, T, Q]
        V = V.view(-1, *V.shape[2:])  # [B*n_head, C, T, Q]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [B', T, C*Q]

        K = K.transpose(2, 3)
        K = K.contiguous().view([B * self.n_head, -1, old_T])  # [B', C*Q, T]

        V = V.transpose(1, 2)  # [B', T, C, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*Q]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K) / (emb_dim**0.5)  # [B', T, T]
        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*Q]

        V = V.reshape(old_shape)  # [B', T, C, Q]
        V = V.transpose(1, 2)  # [B', C, T, Q]
        emb_dim = V.shape[1]

        batch = V.contiguous().view(
            [B, self.n_head * emb_dim, old_T, old_Q]
        )  # [B, C, T, Q])
        batch = self["attn_concat_proj"](batch)  # [B, C, T, Q])

        out = batch + inter_rnn
        return out


class LayerNormalization4DCF(nn.Module):
    """
    LayerNormalization4DCF is a layer normalization module designed for use in 
deep learning models that require normalization across specific dimensions 
for stability and performance, particularly in the context of time-frequency 
domain processing.

Attributes:
    gamma (Parameter): Learnable scale parameter for normalization.
    beta (Parameter): Learnable shift parameter for normalization.
    eps (float): A small constant added to the variance for numerical stability.

Args:
    input_dimension (Tuple[int, int]): A tuple representing the input 
        dimensions, where the first element is the number of features 
        and the second is the number of frequency bins.
    eps (float, optional): A small value to prevent division by zero 
        during normalization. Defaults to 1e-5.

Raises:
    ValueError: If the input tensor does not have 4 dimensions.

Examples:
    >>> layer_norm = LayerNormalization4DCF((128, 64))
    >>> input_tensor = torch.randn(32, 128, 10, 64)  # [B, C, T, F]
    >>> output_tensor = layer_norm(input_tensor)
    >>> print(output_tensor.shape)  # Should be [32, 128, 10, 64]
    """
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        """
        Forward pass through the TFGridNetV2 model.

        This method processes the input audio tensor and produces enhanced audio 
        signals for each source. The input is first normalized and then passed 
        through the model's layers to obtain the enhanced outputs.

        Args:
            input (torch.Tensor): Batched multi-channel audio tensor with M audio 
                channels and N samples of shape [B, N, M].
            ilens (torch.Tensor): Input lengths for each sample in the batch, 
                shape [B].
            additional (Dict or None): Other data that can be passed to the 
                model, currently unused.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]: A tuple containing:
                - enhanced (List[torch.Tensor]): A list of length n_srcs, where 
                  each tensor has shape [B, T], representing mono audio 
                  tensors with T samples.
                - ilens (torch.Tensor): The input lengths, shape [B].
                - additional (OrderedDict): The same additional data returned 
                  as output, currently unused.

        Examples:
            >>> model = TFGridNetV2(input_dim=128, n_srcs=2)
            >>> input_tensor = torch.randn(4, 512, 1)  # [B, N, M]
            >>> ilens = torch.tensor([512, 512, 512, 512])  # Input lengths
            >>> enhanced, ilens_out, _ = model(input_tensor, ilens)
            >>> print(len(enhanced))  # Should be equal to n_srcs (e.g., 2)

        Note:
            It is recommended to normalize the input tensor before passing it to 
            the model, especially when not using scale-invariant loss functions 
            like SI-SDR. Normalization can be performed as follows:
                std_ = torch.std(input, dim=(1, 2), keepdim=True)
                input = input / std_

        Raises:
            AssertionError: If the input tensor shape is not as expected or if 
            the number of input microphones is not supported.
        """
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class AllHeadPReLULayerNormalization4DCF(nn.Module):
    """
    AllHeadPReLULayerNormalization4DCF applies layer normalization across multiple 
heads in a tensor with PReLU activation.

This class normalizes the input tensor along specified dimensions, enabling 
stable training of models that utilize multiple attention heads, particularly in 
the context of deep learning architectures. It is specifically designed to work 
with tensors shaped as [B, H, E, T, F], where B is the batch size, H is the 
number of heads, E is the embedding dimension, T is the sequence length, and 
F is the number of frequency bins.

Attributes:
    gamma (Parameter): Scale parameter for normalization.
    beta (Parameter): Shift parameter for normalization.
    act (PReLU): PReLU activation function applied to the input.
    eps (float): Small value to avoid division by zero in normalization.
    H (int): Number of heads.
    E (int): Embedding dimension.
    n_freqs (int): Number of frequency bins.

Args:
    input_dimension (Tuple[int, int, int]): The input dimensions (H, E, n_freqs).
    eps (float, optional): Small epsilon for numerical stability. Default is 1e-5.

Raises:
    AssertionError: If input_dimension does not have a length of 3.

Examples:
    >>> layer_norm = AllHeadPReLULayerNormalization4DCF((4, 512, 128))
    >>> input_tensor = torch.randn(32, 4, 512, 100, 128)  # [B, H, E, T, F]
    >>> output_tensor = layer_norm(input_tensor)
    >>> output_tensor.shape
    torch.Size([32, 4, 512, 100, 128])  # Output retains the same shape as input
    """
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 3
        H, E, n_freqs = input_dimension
        param_size = [1, H, E, 1, n_freqs]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.act = nn.PReLU(num_parameters=H, init=0.25)
        self.eps = eps
        self.H = H
        self.E = E
        self.n_freqs = n_freqs

    def forward(self, x):
        """
        Perform the forward pass of the TFGridNetV2 model.

        This method processes the input audio tensor through the various layers 
        of the TFGridNetV2 model, including the STFT encoder, multiple GridNetV2 
        blocks, and the STFT decoder, to produce enhanced audio outputs.

        Args:
            input (torch.Tensor): A batched multi-channel audio tensor with 
                M audio channels and N samples, shaped as [B, N, M].
            ilens (torch.Tensor): A tensor containing the input lengths, shaped as [B].
            additional (Dict or None): A dictionary for any additional data, 
                currently unused in this model.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
                - enhanced (List[Union(torch.Tensor)]): A list of length n_srcs 
                  containing mono audio tensors with T samples, shaped as 
                  [(B, T), ...].
                - ilens (torch.Tensor): The input lengths, shaped as (B,).
                - additional (OrderedDict): The additional data, currently unused, 
                  returned in the output.

        Examples:
            >>> model = TFGridNetV2(n_srcs=2, n_fft=256, stride=128)
            >>> input_tensor = torch.randn(4, 1024, 1)  # [B, N, M]
            >>> ilens = torch.tensor([1024, 1024, 1024, 1024])  # [B]
            >>> enhanced, ilens_out, _ = model(input_tensor, ilens)

        Note:
            The model works best when trained with variance normalized mixture input 
            and target. For instance, normalize the mixture and target signals as 
            follows:
                std_ = torch.std(mixture, (1, 2))
                mixture = mixture / std_
                target = target / std_
        """
        assert x.ndim == 4
        B, _, T, _ = x.shape
        x = x.view([B, self.H, self.E, T, self.n_freqs])
        x = self.act(x)  # [B,H,E,T,F]
        stat_dim = (2, 4)
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,H,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,H,1,T,1]
        x = ((x - mu_) / std_) * self.gamma + self.beta  # [B,H,E,T,F]
        return x
