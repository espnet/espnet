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


class TFGridNet(AbsSeparator):
    """
    Offline TFGridNet for speech separation.

    This class implements the TF-GridNet model, which integrates
    full- and sub-band modeling for speech separation, as described in
    the following references:

    [1] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation",
    in arXiv preprint arXiv:2211.12433, 2022.

    [2] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural
    Speaker Separation", in arXiv preprint arXiv:2209.03952, 2022.

    Note:
        The model performs optimally when trained with variance-normalized
        mixture input and target signals. For instance, for a mixture
        tensor of shape [batch, samples, microphones], normalize it by
        dividing with torch.std(mixture, (1, 2)). The same normalization
        should be applied to the target signals, especially when not using
        scale-invariant loss functions like SI-SDR.

    Args:
        input_dim (int): Placeholder, not used.
        n_srcs (int): Number of output sources/speakers.
        n_fft (int): STFT window size.
        stride (int): STFT stride.
        window (str): STFT window type; options are 'hamming', 'hanning', or None.
        n_imics (int): Number of microphone channels (only fixed-array geometry supported).
        n_layers (int): Number of TFGridNet blocks.
        lstm_hidden_units (int): Number of hidden units in LSTM.
        attn_n_head (int): Number of heads in self-attention.
        attn_approx_qk_dim (int): Approximate dimension of frame-level key and
            value tensors.
        emb_dim (int): Embedding dimension.
        emb_ks (int): Kernel size for unfolding and deconvolution (deconv1D).
        emb_hs (int): Hop size for unfolding and deconvolution (deconv1D).
        activation (str): Activation function to use in the TFGridNet model,
            e.g., 'relu' or 'elu'.
        eps (float): Small epsilon for normalization layers.
        use_builtin_complex (bool): Whether to use built-in complex type or not.
        ref_channel (int): Reference channel for the input signals, default is -1.

    Returns:
        Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
            A tuple containing:
                - enhanced (List[torch.Tensor]): List of mono audio tensors
                  with shape [(B, T), ...] for each source.
                - ilens (torch.Tensor): Input lengths of shape (B,).
                - additional (OrderedDict): Currently unused data, returned
                  as part of the output.

    Examples:
        >>> model = TFGridNet(n_srcs=2, n_fft=256)
        >>> input_tensor = torch.randn(10, 16000, 1)  # [B, N, M]
        >>> ilens = torch.tensor([16000] * 10)  # Lengths
        >>> enhanced, ilens, _ = model(input_tensor, ilens)
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
        ref_channel=-1,
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_imics = n_imics
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1
        self.ref_channel = ref_channel

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
                GridNetBlock(
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
        Forward pass for the TFGridNet model.

        This method processes the input multi-channel audio tensor to perform
        speech separation using the TFGridNet architecture.

        Args:
            input (torch.Tensor): Batched multi-channel audio tensor with
                M audio channels and N samples of shape [B, N, M].
            ilens (torch.Tensor): Input lengths of shape [B].
            additional (Dict or None): Other data, currently unused in this model.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
                - enhanced (List[torch.Tensor]): A list of length n_srcs
                  containing mono audio tensors with T samples of shape
                  [(B, T), ...].
                - ilens (torch.Tensor): Input lengths of shape (B,).
                - additional (OrderedDict): Other data, currently unused in
                  this model, returned as part of the output.

        Examples:
            >>> model = TFGridNet(n_srcs=2, n_fft=128)
            >>> input_tensor = torch.randn(4, 16000, 2)  # [B, N, M]
            >>> ilens = torch.tensor([16000, 16000, 16000, 16000])  # [B]
            >>> enhanced, ilens_out, _ = model(input_tensor, ilens)

        Note:
            As outlined in the model documentation, this model works best when
            trained with variance normalized mixture input and target. For a
            mixture of shape [batch, samples, microphones], normalize it by
            dividing with `torch.std(mixture, (1, 2))`. The same should be done
            for the target signals, especially when not using scale-invariant
            loss functions such as SI-SDR.
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
        """
            Offline TFGridNet

        Reference:
        [1] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
        "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation",
        in arXiv preprint arXiv:2211.12433, 2022.
        [2] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
        "TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural
        Speaker Separation", in arXiv preprint arXiv:2209.03952, 2022.

        NOTES:
        As outlined in the Reference, this model works best when trained with variance
        normalized mixture input and target, e.g., with mixture of shape [batch, samples,
        microphones], you normalize it by dividing with torch.std(mixture, (1, 2)). You
        must do the same for the target signals. It is encouraged to do so when not using
        scale-invariant loss functions such as SI-SDR.

        Args:
            input_dim: placeholder, not used
            n_srcs: number of output sources/speakers.
            n_fft: stft window size.
            stride: stft stride.
            window: stft window type choose between 'hamming', 'hanning' or None.
            n_imics: number of microphones channels (only fixed-array geometry supported).
            n_layers: number of TFGridNet blocks.
            lstm_hidden_units: number of hidden units in LSTM.
            attn_n_head: number of heads in self-attention.
            attn_approx_qk_dim: approximate dimension of frame-level key and value tensors.
            emb_dim: embedding dimension.
            emb_ks: kernel size for unfolding and deconv1D.
            emb_hs: hop size for unfolding and deconv1D.
            activation: activation function to use in the whole TFGridNet model,
                you can use any torch supported activation e.g. 'relu' or 'elu'.
            eps: small epsilon for normalization layers.
            use_builtin_complex: whether to use builtin complex type or not.

        Examples:
            >>> model = TFGridNet(n_srcs=3, n_fft=256)
            >>> input_tensor = torch.randn(2, 1000, 1)  # [B, N, M]
            >>> ilens = torch.tensor([1000, 1000])  # input lengths
            >>> output, ilens_out, _ = model(input_tensor, ilens)

        Attributes:
            num_spk: Returns the number of output sources/speakers.
        """
        return self.n_srcs

    @staticmethod
    def pad2(input_tensor, target_len):
        """
        Pads the input tensor to the specified target length.

        This method uses PyTorch's functional pad operation to add zeros to the
        end of the input tensor until it reaches the desired length. If the
        input tensor is already longer than the target length, it will remain
        unchanged.

        Args:
            input_tensor (torch.Tensor): The input tensor to be padded.
                It is expected to be of shape [B, C, T] where B is the batch
                size, C is the number of channels, and T is the length of the
                tensor along the last dimension.
            target_len (int): The desired length of the last dimension after
                padding.

        Returns:
            torch.Tensor: The padded tensor with the shape [B, C, target_len].

        Examples:
            >>> input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> target_len = 5
            >>> padded_tensor = TFGridNet.pad2(input_tensor, target_len)
            >>> print(padded_tensor)
            tensor([[1, 2, 3, 0, 0],
                    [4, 5, 6, 0, 0]])

        Note:
            If the input tensor's last dimension is already equal to or
            greater than target_len, the output tensor will be the same as
            the input tensor.
        """
        input_tensor = torch.nn.functional.pad(
            input_tensor, (0, target_len - input_tensor.shape[-1])
        )
        return input_tensor


class GridNetBlock(nn.Module):
    """
    A block in the TFGridNet architecture for processing audio features.

    This class implements a single GridNetBlock, which consists of
    intra and inter temporal processing layers, followed by an
    attention mechanism. It applies LSTM layers for sequential data
    processing and utilizes convolutional layers for feature
    transformations.

    Attributes:
        emb_dim (int): The embedding dimension for the input features.
        emb_ks (int): The kernel size for convolution operations.
        emb_hs (int): The hop size for convolution operations.
        n_head (int): The number of attention heads in the attention mechanism.

    Args:
        emb_dim (int): The dimensionality of the input features.
        emb_ks (int): The kernel size for convolution operations.
        emb_hs (int): The hop size for convolution operations.
        n_freqs (int): The number of frequency bins.
        hidden_channels (int): The number of hidden channels in LSTM.
        n_head (int, optional): The number of heads in self-attention. Defaults to 4.
        approx_qk_dim (int, optional): Approximate dimension for key and value
            tensors. Defaults to 512.
        activation (str, optional): The activation function to use. Defaults to 'prelu'.
        eps (float, optional): Small value for numerical stability in normalization.
            Defaults to 1e-5.

    Examples:
        >>> block = GridNetBlock(emb_dim=64, emb_ks=3, emb_hs=1, n_freqs=128,
        ...                       hidden_channels=128, n_head=4)
        >>> input_tensor = torch.randn(8, 128, 16, 32)  # Example input
        >>> output_tensor = block(input_tensor)
        >>> output_tensor.shape
        torch.Size([8, 64, 16, 32])  # Output shape after processing

    Raises:
        ValueError: If the input tensor does not have 4 dimensions.
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

        in_channels = emb_dim * emb_ks

        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.intra_linear = nn.ConvTranspose1d(
            hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
        )

        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.inter_linear = nn.ConvTranspose1d(
            hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
        )

        E = math.ceil(
            approx_qk_dim * 1.0 / n_freqs
        )  # approx_qk_dim is only approximate
        assert emb_dim % n_head == 0
        for ii in range(n_head):
            self.add_module(
                "attn_conv_Q_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_K_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_V_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, emb_dim // n_head, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((emb_dim // n_head, n_freqs), eps=eps),
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
        Forward pass for the GridNetBlock.

        This method processes the input audio tensor through the model,
        performing operations such as normalization, encoding, and
        applying the GridNetBlock layers.

        Args:
            input (torch.Tensor): Batched multi-channel audio tensor with
                M audio channels and N samples, shaped as [B, N, M].
            ilens (torch.Tensor): Input lengths, shaped as [B].
            additional (Dict or None): Other data, currently unused in this model.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
                - enhanced (List[torch.Tensor]): A list of length n_srcs,
                  containing mono audio tensors with T samples each.
                - ilens (torch.Tensor): The input lengths, shaped as [B].
                - additional (OrderedDict): Other data, currently unused in
                  this model, returned in the output.

        Examples:
            >>> model = TFGridNet(n_srcs=2, n_fft=128)
            >>> input_tensor = torch.randn(4, 16000, 1)  # [B, N, M]
            >>> ilens = torch.tensor([16000] * 4)  # Input lengths
            >>> enhanced, ilens_out, _ = model(input_tensor, ilens)

        Note:
            The model works best when trained with variance normalized mixture
            input and target. Normalize input by dividing with
            torch.std(mixture, (1, 2)) and do the same for the target signals.
            This is encouraged when not using scale-invariant loss functions
            such as SI-SDR.
        """
        B, C, old_T, old_Q = x.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = F.pad(x, (0, Q - old_Q, 0, T - old_T))

        # intra RNN
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, C, T, Q]
        intra_rnn = (
            intra_rnn.transpose(1, 2).contiguous().view(B * T, C, Q)
        )  # [BT, C, Q]
        intra_rnn = F.unfold(
            intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BT, C*emb_ks, -1]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*emb_ks]
        intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
        intra_rnn = intra_rnn.view([B, T, C, Q])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, Q]
        intra_rnn = intra_rnn + input_  # [B, C, T, Q]

        # inter RNN
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)  # [B, C, T, F]
        inter_rnn = (
            inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)
        )  # [BF, C, T]
        inter_rnn = F.unfold(
            inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BF, C*emb_ks, -1]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, -1, C*emb_ks]
        inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BF, -1, H]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, H, -1]
        inter_rnn = self.inter_linear(inter_rnn)  # [BF, C, T]
        inter_rnn = inter_rnn.view([B, Q, C, T])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()  # [B, C, T, Q]
        inter_rnn = inter_rnn + input_  # [B, C, T, Q]

        # attention
        inter_rnn = inter_rnn[..., :old_T, :old_Q]
        batch = inter_rnn

        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            all_Q.append(self["attn_conv_Q_%d" % ii](batch))  # [B, C, T, Q]
            all_K.append(self["attn_conv_K_%d" % ii](batch))  # [B, C, T, Q]
            all_V.append(self["attn_conv_V_%d" % ii](batch))  # [B, C, T, Q]

        Q = torch.cat(all_Q, dim=0)  # [B', C, T, Q]
        K = torch.cat(all_K, dim=0)  # [B', C, T, Q]
        V = torch.cat(all_V, dim=0)  # [B', C, T, Q]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [B', T, C*Q]
        K = K.transpose(1, 2)
        K = K.flatten(start_dim=2)  # [B', T, C*Q]
        V = V.transpose(1, 2)  # [B', T, C, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*Q]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]
        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*Q]

        V = V.reshape(old_shape)  # [B', T, C, Q]
        V = V.transpose(1, 2)  # [B', C, T, Q]
        emb_dim = V.shape[1]

        batch = V.view([self.n_head, B, emb_dim, old_T, -1])  # [n_head, B, C, T, Q])
        batch = batch.transpose(0, 1)  # [B, n_head, C, T, Q])
        batch = batch.contiguous().view(
            [B, self.n_head * emb_dim, old_T, -1]
        )  # [B, C, T, Q])
        batch = self["attn_concat_proj"](batch)  # [B, C, T, Q])

        out = batch + inter_rnn
        return out


class LayerNormalization4D(nn.Module):
    """
    4D Layer Normalization.

    This class implements layer normalization for 4-dimensional tensors.
    It normalizes the input tensor along specified dimensions and scales
    and shifts the result using learnable parameters.

    Args:
        input_dimension (int): The size of the input feature dimension.
        eps (float, optional): A small constant added to the variance for
            numerical stability. Default is 1e-5.

    Raises:
        ValueError: If the input tensor does not have 4 dimensions.

    Examples:
        >>> layer_norm = LayerNormalization4D(input_dimension=64)
        >>> input_tensor = torch.randn(10, 64, 32, 32)  # [B, C, H, W]
        >>> output_tensor = layer_norm(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([10, 64, 32, 32])

    Note:
        The input tensor is expected to have 4 dimensions, where the
        dimensions represent batch size, number of channels, height,
        and width respectively.
    """

    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        """
        Forward pass through the TFGridNet model.

        This method processes the input multi-channel audio tensor and returns
        the enhanced audio signals for each source.

        Args:
            input (torch.Tensor): Batched multi-channel audio tensor with
                M audio channels and N samples [B, N, M].
            ilens (torch.Tensor): Input lengths [B].
            additional (Dict or None): Other data, currently unused in this model.

        Returns:
            enhanced (List[Union(torch.Tensor)]): A list of length n_srcs,
                containing mono audio tensors with T samples for each source.
            ilens (torch.Tensor): The input lengths [B].
            additional (OrderedDict): Other data, currently unused in this model,
                returned in output.

        Examples:
            >>> model = TFGridNet(n_srcs=2)
            >>> input_tensor = torch.randn(4, 16000, 1)  # [B, N, M]
            >>> ilens = torch.tensor([16000, 16000, 16000, 16000])  # [B]
            >>> enhanced, ilens_out, _ = model(input_tensor, ilens)
            >>> len(enhanced)  # Should be equal to n_srcs
            2

        Note:
            The model works best when trained with variance normalized mixture input
            and target. For example, normalize the mixture of shape [batch, samples,
            microphones] by dividing with torch.std(mixture, (1, 2)). This must also
            be done for the target signals, especially when not using scale-invariant
            loss functions such as SI-SDR.
        """
        if x.ndim == 4:
            _, C, _, _ = x.shape
            stat_dim = (1,)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class LayerNormalization4DCF(nn.Module):
    """
    Layer normalization for 4D tensors with respect to channel and frequency dimensions.

    This layer applies layer normalization across the specified dimensions of the
    input tensor. It normalizes the input tensor such that the mean and variance
    are computed across the specified dimensions, enabling stable training of deep
    learning models.

    Attributes:
        gamma (Parameter): Learnable scale parameter of shape
            [1, input_dimension[0], 1, input_dimension[1]].
        beta (Parameter): Learnable shift parameter of shape
            [1, input_dimension[0], 1, input_dimension[1]].
        eps (float): A small value added for numerical stability during normalization.

    Args:
        input_dimension (tuple): A tuple specifying the dimensions for normalization.
            It should have a length of 2, corresponding to the channel and
            frequency dimensions.
        eps (float): Small epsilon value to avoid division by zero during
            normalization. Default is 1e-5.

    Raises:
        ValueError: If the input tensor does not have 4 dimensions.

    Examples:
        >>> layer_norm = LayerNormalization4DCF((64, 128))
        >>> input_tensor = torch.randn(32, 64, 10, 128)  # [B, C, T, F]
        >>> output_tensor = layer_norm(input_tensor)
        >>> print(output_tensor.shape)  # Should match input_tensor shape [32, 64, 10, 128]

    Note:
        The input tensor should be of shape [B, C, T, F] where:
            - B is the batch size,
            - C is the number of channels,
            - T is the time dimension,
            - F is the frequency dimension.
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
        Perform the forward pass of the TFGridNet model.

        This method takes in a batched multi-channel audio input and performs
        the forward computation through the network layers to produce enhanced
        audio outputs for the specified number of sources.

        Args:
            input (torch.Tensor): Batched multi-channel audio tensor with
                shape [B, N, M] where B is the batch size, N is the number
                of samples, and M is the number of audio channels.
            ilens (torch.Tensor): Input lengths with shape [B], indicating
                the length of each input sequence in the batch.
            additional (Dict or None): Other data that may be passed to the
                model. Currently unused in this model.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]: A tuple
            containing:
                - enhanced (List[torch.Tensor]): A list of length `n_srcs`
                  containing mono audio tensors with shape [B, T], where T
                  is the number of output samples for each source.
                - ilens (torch.Tensor): The input lengths [B].
                - additional (OrderedDict): Other data, currently unused,
                  returned for consistency.

        Examples:
            >>> model = TFGridNet(n_srcs=2, n_fft=128)
            >>> input_tensor = torch.randn(4, 512, 2)  # 4 samples, 512 time steps, 2 channels
            >>> input_lengths = torch.tensor([512, 512, 512, 512])
            >>> enhanced, lengths, _ = model.forward(input_tensor, input_lengths)
            >>> print(len(enhanced))  # Should print 2, for 2 sources
            >>> print(enhanced[0].shape)  # Output shape for the first source

        Note:
            It is recommended to normalize the input and target signals by
            their RMS values before feeding them into the model for optimal
            performance, especially when not using scale-invariant loss
            functions such as SI-SDR.
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
