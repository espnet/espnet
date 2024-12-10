import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

from espnet2.enh.layers.complex_utils import is_complex, new_complex_like
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.torch_utils.get_layer_from_string import get_layer

if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


class TFGridNetV3(AbsSeparator):
    """
    TFGridNetV3 is an advanced model for offline time-frequency (TF) audio source 
separation, extending the capabilities of TFGridNetV2. It is designed to be 
sampling-frequency-independent (SFI) by ensuring that all layers are independent 
of the input's time and frequency dimensions.

References:
1. Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
   "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation",
   in TASLP, 2023.
2. Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
   "TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural
   Speaker Separation", in ICASSP, 2023.

Notes:
This model performs optimally when trained with variance-normalized mixture 
inputs and targets. For a mixture tensor of shape [batch, samples, microphones], 
normalize it using:
    std_ = std(mixture, (1, 2))
    mixture = mixture / std_
    target = target / std_

Attributes:
    n_srcs (int): Number of output sources/speakers.
    n_layers (int): Number of TFGridNetV3 blocks.
    n_imics (int): Number of microphone channels (only fixed-array geometry 
                   supported).
    
Args:
    input_dim (int): Placeholder, not used.
    n_srcs (int): Number of output sources/speakers (default: 2).
    n_fft (int): STFT window size.
    stride (int): STFT stride.
    window (str or None): STFT window type, can be 'hamming', 'hanning', or 
                          None.
    n_imics (int): Number of microphones channels (default: 1).
    n_layers (int): Number of TFGridNetV3 blocks (default: 6).
    lstm_hidden_units (int): Number of hidden units in LSTM (default: 192).
    attn_n_head (int): Number of heads in self-attention (default: 4).
    attn_qk_output_channel (int): Output channels for point-wise conv2d for 
                                   getting key and query (default: 4).
    emb_dim (int): Embedding dimension (default: 48).
    emb_ks (int): Kernel size for unfolding and deconv1D (default: 4).
    emb_hs (int): Hop size for unfolding and deconv1D (default: 1).
    activation (str): Activation function to use in the model, can be any 
                      torch-supported activation (default: 'prelu').
    eps (float): Small epsilon for normalization layers (default: 1.0e-5).
    use_builtin_complex (bool): Whether to use built-in complex type or not.

Examples:
    # Instantiate the model
    model = TFGridNetV3(n_srcs=3, n_layers=4)

    # Prepare input tensor
    input_tensor = torch.randn(8, 2, 512)  # Example with 8 batches, 2 channels, 512 samples
    ilens = torch.tensor([512] * 8)  # Input lengths for each batch

    # Forward pass
    enhanced, ilens, additional = model(input_tensor, ilens)
    """

    def __init__(
        self,
        input_dim,
        n_srcs=2,
        n_imics=1,
        n_layers=6,
        lstm_hidden_units=192,
        attn_n_head=4,
        attn_qk_output_channel=4,
        emb_dim=48,
        emb_ks=4,
        emb_hs=1,
        activation="prelu",
        eps=1.0e-5,
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_imics = n_imics
        assert self.n_imics == 1, self.n_imics

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * n_imics, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )

        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                GridNetV3Block(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    lstm_hidden_units,
                    n_head=attn_n_head,
                    qk_output_channel=attn_qk_output_channel,
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
        Forward pass of the TFGridNetV3 model.

        This method takes a batched multi-channel audio tensor as input and 
        processes it through the model to produce enhanced audio signals for 
        the specified number of sources (speakers).

        Args:
            input (torch.Tensor): 
                Batched multi-channel audio tensor with M audio channels 
                and N samples of shape [B, T, F].
            ilens (torch.Tensor): 
                Input lengths of shape [B].
            additional (Dict or None): 
                Other data, currently unused in this model.

        Returns:
            enhanced (List[torch.Tensor]):
                A list of length `n_srcs` containing mono audio tensors 
                of shape [(B, T), ...], where T is the number of samples.
            ilens (torch.Tensor): 
                The input lengths, returned as shape (B,).
            additional (OrderedDict): 
                Other data, currently unused in this model, returned as output.

        Examples:
            >>> model = TFGridNetV3(n_srcs=2)
            >>> input_tensor = torch.randn(4, 16000, 2)  # 4 samples, 16000 time steps
            >>> ilens = torch.tensor([16000, 16000, 16000, 16000])  # lengths
            >>> enhanced, ilens_out, _ = model(input_tensor, ilens)

        Note:
            This model works best when trained with variance normalized mixture 
            input and target. Normalize the mixture by dividing it with 
            torch.std(mixture, (1, 2)), and do the same for the target signals.
            Specifically, use:
                std_ = std(mix)
                mix = mix / std_
                tgt = tgt / std_

        Raises:
            AssertionError: If the input tensor is not in the expected shape 
            or the number of channels is not equal to 2.
        """

        # B, 2, T, (C,) F
        if is_complex(input):
            feature = torch.stack([input.real, input.imag], dim=1)
        else:
            assert input.size(-1) == 2, input.shape
            feature = input.moveaxis(-1, 1)

        assert feature.ndim == 4, "Only single-channel mixture is supported now"

        n_batch, _, n_frames, n_freqs = feature.shape

        batch = self.conv(feature)  # [B, -1, T, F]

        for ii in range(self.n_layers):
            batch = self.blocks[ii](batch)  # [B, -1, T, F]

        batch = self.deconv(batch)  # [B, n_srcs*2, T, F]

        batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])
        batch = new_complex_like(input, (batch[:, :, 0], batch[:, :, 1]))

        batch = [batch[:, src] for src in range(self.num_spk)]

        return batch, ilens, OrderedDict()

    @property
    def num_spk(self):
        return self.n_srcs


class GridNetV3Block(nn.Module):
    """
    GridNetV3 Block for processing audio features.

    This class implements a block of the GridNetV3 architecture, which is 
    designed for audio signal processing. It utilizes intra- and inter- 
    recurrent neural networks (RNNs) with attention mechanisms for enhanced 
    feature extraction.

    Attributes:
        emb_dim (int): The embedding dimension.
        emb_ks (int): Kernel size for embedding.
        emb_hs (int): Hop size for embedding.
        n_head (int): Number of heads in the attention mechanism.

    Args:
        emb_dim (int): The embedding dimension.
        emb_ks (int): Kernel size for embedding.
        emb_hs (int): Hop size for embedding.
        hidden_channels (int): Number of hidden channels in LSTM.
        n_head (int, optional): Number of heads in the attention mechanism. 
            Defaults to 4.
        qk_output_channel (int, optional): Output channels of point-wise 
            conv2d for key and query. Defaults to 4.
        activation (str, optional): Activation function to use, defaults to 
            "prelu".
        eps (float, optional): Small value for numerical stability in 
            normalization layers. Defaults to 1e-5.

    Raises:
        AssertionError: If the activation function is not "prelu".

    Examples:
        >>> block = GridNetV3Block(emb_dim=64, emb_ks=3, emb_hs=1,
        ...                         hidden_channels=128)
        >>> x = torch.randn(32, 192, 100, 50)  # Example input
        >>> output = block(x)
        >>> print(output.shape)  # Output shape should match input shape
    """
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        hidden_channels,
        n_head=4,
        qk_output_channel=4,
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

        # use constant E not to be dependent on the number of frequency bins
        E = qk_output_channel
        assert emb_dim % n_head == 0

        self.add_module("attn_conv_Q", nn.Conv2d(emb_dim, n_head * E, 1))
        self.add_module(
            "attn_norm_Q",
            AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps),
        )

        self.add_module("attn_conv_K", nn.Conv2d(emb_dim, n_head * E, 1))
        self.add_module(
            "attn_norm_K",
            AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps),
        )

        self.add_module(
            "attn_conv_V", nn.Conv2d(emb_dim, n_head * emb_dim // n_head, 1)
        )
        self.add_module(
            "attn_norm_V",
            AllHeadPReLULayerNormalization4DC((n_head, emb_dim // n_head), eps=eps),
        )

        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                get_layer(activation)(),
                LayerNormalization(emb_dim, dim=-3, total_dim=4, eps=eps),
            ),
        )

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x):
        """
        Perform the forward pass of the TFGridNetV3 model.

        This method processes the input multi-channel audio tensor and applies
        the model architecture to separate the sources. It takes a batch of
        audio signals, applies convolutional layers, and processes them through
        multiple GridNetV3 blocks before returning the enhanced audio signals.

        Args:
            input (torch.Tensor): Batched multi-channel audio tensor with
                M audio channels and N samples shaped as [B, T, F].
            ilens (torch.Tensor): Input lengths for each batch element shaped as [B].
            additional (Dict or None): Additional data, currently unused in
                this model.

        Returns:
            enhanced (List[torch.Tensor]): A list of length n_srcs containing
                mono audio tensors shaped as [(B, T), ...] with T samples each.
            ilens (torch.Tensor): Input lengths shaped as (B,).
            additional (OrderedDict): The additional data returned in the output,
                currently unused in this model.

        Examples:
            >>> model = TFGridNetV3(n_srcs=2, n_imics=1)
            >>> input_tensor = torch.randn(4, 256, 2)  # 4 batches, 256 time steps, 2 channels
            >>> ilens = torch.tensor([256, 256, 256, 256])  # lengths for each input
            >>> enhanced, lengths, _ = model(input_tensor, ilens)

        Note:
            Ensure that the input tensor is normalized as described in the
            model's notes for optimal performance. The model works best with
            variance normalized mixture input and target signals.

        Raises:
            AssertionError: If the input tensor does not have the expected shape
            or if the input tensor is not a single-channel mixture.
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


class LayerNormalization(nn.Module):
    """
    Layer normalization layer.

    This layer applies layer normalization to the input tensor along a specified
    dimension. It normalizes the input by subtracting the mean and dividing by
    the standard deviation, followed by scaling and shifting with learnable
    parameters gamma and beta.

    Attributes:
        dim (int): The dimension along which to compute the mean and variance.
        gamma (nn.Parameter): Scale parameter for normalization.
        beta (nn.Parameter): Shift parameter for normalization.
        eps (float): A small value added to the variance to avoid division by zero.

    Args:
        input_dim (int): The dimension of the input tensor to normalize.
        dim (int): The dimension along which to compute the normalization. 
                   Default is 1.
        total_dim (int): The total number of dimensions of the input tensor. 
                         Default is 4.
        eps (float): A small value to prevent division by zero during normalization. 
                     Default is 1e-5.

    Raises:
        ValueError: If the input tensor does not have the expected number of dimensions.

    Examples:
        >>> layer_norm = LayerNormalization(input_dim=64)
        >>> input_tensor = torch.randn(32, 64, 128, 256)  # [B, C, T, F]
        >>> output_tensor = layer_norm(input_tensor)
        >>> output_tensor.shape
        torch.Size([32, 64, 128, 256])  # Output has the same shape as input
    """
    def __init__(self, input_dim, dim=1, total_dim=4, eps=1e-5):
        super().__init__()
        self.dim = dim if dim >= 0 else total_dim + dim
        param_size = [1 if ii != self.dim else input_dim for ii in range(total_dim)]
        self.gamma = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        self.eps = eps

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        """
        Forward pass for the TFGridNetV3 model.

    This method takes a batched multi-channel audio tensor as input and
    processes it through the layers of the TFGridNetV3 model, outputting
    the enhanced audio signals for each source.

    Args:
        input (torch.Tensor): Batched multi-channel audio tensor with shape
            [B, T, F], where B is the batch size, T is the number of samples,
            and F is the number of audio channels.
        ilens (torch.Tensor): Input lengths of shape [B], indicating the
            length of each input sequence in the batch.
        additional (Dict or None): Other data, currently unused in this model.

    Returns:
        enhanced (List[Union(torch.Tensor)]): A list of length n_srcs, each
            containing mono audio tensors with shape [B, T].
        ilens (torch.Tensor): Tensor of shape [B] representing the input lengths.
        additional (Dict or None): Returns the additional data, currently unused
            in this model.

    Raises:
        AssertionError: If the input is not a single-channel mixture.

    Examples:
        >>> model = TFGridNetV3(n_srcs=2)
        >>> input_tensor = torch.randn(4, 256, 2)  # Batch of 4, 256 samples, 2 channels
        >>> ilens = torch.tensor([256, 256, 256, 256])  # Input lengths
        >>> enhanced, lengths, _ = model(input_tensor, ilens)
        >>> print(len(enhanced))  # Should be equal to n_srcs (2)
        >>> print(enhanced[0].shape)  # Shape of enhanced output for the first source
        """
        if x.ndim - 1 < self.dim:
            raise ValueError(
                f"Expect x to have {self.dim + 1} dimensions, but got {x.ndim}"
            )
        if x.dtype in HALF_PRECISION_DTYPES:
            dtype = x.dtype
            x = x.float()
        else:
            dtype = None
        mu_ = x.mean(dim=self.dim, keepdim=True)
        std_ = torch.sqrt(x.var(dim=self.dim, unbiased=False, keepdim=True) + self.eps)
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat.to(dtype=dtype) if dtype else x_hat


class AllHeadPReLULayerNormalization4DC(nn.Module):
    """
    Applies PReLU activation followed by layer normalization across heads.

    This layer normalizes the input tensor along specified dimensions after applying 
    the PReLU activation function. It is designed for multi-dimensional inputs, 
    particularly suited for use in attention mechanisms where inputs are structured 
    with heads and embedding dimensions.

    Attributes:
        gamma (torch.Parameter): Scale parameter for layer normalization.
        beta (torch.Parameter): Shift parameter for layer normalization.
        act (nn.PReLU): PReLU activation function applied per head.
        eps (float): Small value added for numerical stability during normalization.
        H (int): Number of heads in the input dimension.
        E (int): Embedding dimension in the input.

    Args:
        input_dimension (Tuple[int, int]): A tuple containing the number of heads (H) 
            and the embedding dimension (E).
        eps (float): Small value to prevent division by zero in normalization.

    Raises:
        AssertionError: If the input_dimension does not have a length of 2.

    Examples:
        >>> layer_norm = AllHeadPReLULayerNormalization4DC((8, 64))
        >>> input_tensor = torch.randn(32, 8, 128, 64)  # [B, H, T, F]
        >>> output = layer_norm(input_tensor)
        >>> output.shape
        torch.Size([32, 8, 128, 64])  # Normalized output shape remains the same.
    """
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2, input_dimension
        H, E = input_dimension
        param_size = [1, H, E, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.act = nn.PReLU(num_parameters=H, init=0.25)
        self.eps = eps
        self.H = H
        self.E = E

    def forward(self, x):
        """
        Perform the forward pass of the TFGridNetV3 model.

        This method processes the input audio tensor and returns the enhanced
        audio signals along with their corresponding lengths and any additional
        data.

        Args:
            input (torch.Tensor): A batched multi-channel audio tensor with
                shape [B, T, F], where B is the batch size, T is the number of
                time frames, and F is the number of frequency bins.
            ilens (torch.Tensor): A tensor containing the lengths of each input
                sequence in the batch, shape [B].
            additional (Dict or None): Additional data that may be required
                for processing. Currently unused in this model.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
                - enhanced (List[torch.Tensor]): A list of enhanced mono audio
                  tensors of shape [(B, T), ...] where the length of the list
                  is equal to n_srcs (number of output sources).
                - ilens (torch.Tensor): A tensor of shape [B] containing the
                  lengths of the enhanced audio signals.
                - additional (OrderedDict): The additional data returned as-is,
                  currently unused in this model.

        Examples:
            >>> model = TFGridNetV3(n_srcs=2)
            >>> input_tensor = torch.randn(8, 100, 2)  # 8 samples, 100 time frames, 2 channels
            >>> ilens = torch.tensor([100] * 8)  # All samples have length 100
            >>> enhanced, lengths, _ = model(input_tensor, ilens)
            >>> print([e.shape for e in enhanced])  # Output shapes of enhanced signals

        Note:
            Ensure the input tensor is properly normalized. This model works best
            when the input mixture and target signals are variance normalized.

        Raises:
            AssertionError: If the input tensor does not have the correct number
            of channels or if any other assertion within the method fails.
        """
        assert x.ndim == 4
        B, _, T, F = x.shape
        x = x.view([B, self.H, self.E, T, F])
        x = self.act(x)  # [B,H,E,T,F]
        stat_dim = (2,)
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,H,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,H,1,T,1]
        x = ((x - mu_) / std_) * self.gamma + self.beta  # [B,H,E,T,F]
        return x
