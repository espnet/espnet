from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.enh.decoder.stft_decoder import STFTDecoder
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.layers.beamformer import tik_reg, to_double
from espnet2.enh.layers.tcndenseunet import TCNDenseUNet
from espnet2.enh.separator.abs_separator import AbsSeparator

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class iNeuBe(AbsSeparator):
    """
    iNeuBe, iterative neural/beamforming enhancement.

This class implements the iNeuBe model for multi-channel speech enhancement 
using iterative neural and beamforming techniques. It is based on the work 
by Lu et al. (ICASSP 2022), which emphasizes low-distortion multi-channel 
speech enhancement.

Reference:
Lu, Y. J., Cornell, S., Chang, X., Zhang, W., Li, C., Ni, Z., ... & Watanabe, S.
Towards Low-Distortion Multi-Channel Speech Enhancement: 
The ESPNET-Se Submission to the L3DAS22 Challenge. ICASSP 2022 p. 9201-9205.

Notes:
As outlined in the Reference, this model works best when coupled with the 
MultiResL1SpecLoss defined in criterions/time_domain.py. The model is trained 
with variance normalized mixture input and target. For example, with a mixture 
of shape [batch, microphones, samples], normalize it by dividing with 
torch.std(mixture, (1, 2)). The same normalization must be applied to the 
target signal. In the Reference, the variance normalization was performed 
offline; however, normalizing each input and target separately also yields 
good results.

Attributes:
    n_spk (int): Number of output sources/speakers.
    output_from (str): Output the estimate from 'dnn1', 'mfmcwf', or 'dnn2'.
    n_chunks (int): Number of future and past frames for mfMCWF computation.
    freeze_dnn1 (bool): Whether to freeze dnn1 parameters during training of dnn2.
    tik_eps (float): Diagonal loading in the mfMCWF computation.

Args:
    n_spk (int): Number of output sources/speakers.
    n_fft (int): STFT window size.
    stride (int): STFT stride.
    window (str): STFT window type, choose between 'hamming', 'hanning', or None.
    mic_channels (int): Number of microphone channels (only fixed-array geometry supported).
    hid_chans (int): Number of channels in the subsampling/upsampling conv layers.
    hid_chans_dense (int): Number of channels in the densenet layers (reduce to save VRAM).
    ksz_dense (tuple): Kernel size in the densenet layers through iNeuBe.
    ksz_tcn (int): Kernel size in the TCN submodule.
    tcn_repeats (int): Number of repetitions of blocks in the TCN submodule.
    tcn_blocks (int): Number of blocks in the TCN submodule.
    tcn_channels (int): Number of channels in the TCN submodule.
    activation (str): Activation function to use, e.g., 'relu' or 'elu'.
    output_from (str): Output from 'dnn1', 'mfmcwf', or 'dnn2'.
    n_chunks (int): Number of future and past frames for mfMCWF computation.
    freeze_dnn1 (bool): Freeze dnn1 parameters during dnn2 training.
    tik_eps (float): Diagonal loading in mfMCWF computation.

Examples:
    # Instantiate the iNeuBe model
    model = iNeuBe(n_spk=2, n_fft=512, stride=128)

    # Forward pass with random input
    input_tensor = torch.randn(4, 16000, 2)  # 4 samples, 16000 time steps, 2 mics
    ilens = torch.tensor([16000] * 4)  # Lengths of the inputs
    enhanced, lengths, additional = model(input_tensor, ilens)

Raises:
    AssertionError: If the installed torch version is lower than 1.9.0.
    NotImplementedError: If an unsupported output_from option is provided.
    """

    def __init__(
        self,
        n_spk=1,
        n_fft=512,
        stride=128,
        window="hann",
        mic_channels=1,
        hid_chans=32,
        hid_chans_dense=32,
        ksz_dense=(3, 3),
        ksz_tcn=3,
        tcn_repeats=4,
        tcn_blocks=7,
        tcn_channels=384,
        activation="elu",
        output_from="dnn1",
        n_chunks=3,
        freeze_dnn1=False,
        tik_eps=1e-8,
    ):
        super().__init__()
        self.n_spk = n_spk
        self.output_from = output_from
        self.n_chunks = n_chunks
        self.freeze_dnn1 = freeze_dnn1
        self.tik_eps = tik_eps
        fft_c_channels = n_fft // 2 + 1
        assert is_torch_1_9_plus, (
            "iNeuBe model requires torch>=1.9.0, "
            "please install latest torch version."
        )

        self.enc = STFTEncoder(n_fft, n_fft, stride, window=window)
        self.dec = STFTDecoder(n_fft, n_fft, stride, window=window)

        self.dnn1 = TCNDenseUNet(
            n_spk,
            fft_c_channels,
            mic_channels,
            hid_chans,
            hid_chans_dense,
            ksz_dense,
            ksz_tcn,
            tcn_repeats,
            tcn_blocks,
            tcn_channels,
            activation=activation,
        )

        self.dnn2 = TCNDenseUNet(
            1,
            fft_c_channels,
            mic_channels + 2,
            hid_chans,
            hid_chans_dense,
            ksz_dense,
            ksz_tcn,
            tcn_repeats,
            tcn_blocks,
            tcn_channels,
            activation=activation,
        )

    @staticmethod
    def unfold(tf_rep, chunk_size):
        """
        unfold(tf_rep, chunk_size)

    Unfolds the Short-Time Fourier Transform (STFT) representation to add context 
    in the microphone channels. This method is useful for preparing the input data 
    for multi-frame processing by expanding the representation with additional frames 
    from past and future.

    Args:
        tf_rep (torch.Tensor): 3D tensor (monaural complex STFT) of shape 
            [B, T, F] where B is the batch size, T is the number of frames, 
            and F is the number of frequency bins.
        chunk_size (int): The number of past and future frames to consider 
            for each time frame. If set to 0, the input will be returned unchanged.

    Returns:
        est_unfolded (torch.Tensor): Complex 3D tensor STFT with context channels. 
            The shape is now [B, T, C, F], where C is the total number of context 
            frames (2 * chunk_size + 1), effectively creating a multi-channel STFT 
            representation.

    Examples:
        >>> import torch
        >>> tf_rep = torch.randn(2, 100, 256)  # Example STFT representation
        >>> chunk_size = 2
        >>> unfolded_rep = iNeuBe.unfold(tf_rep, chunk_size)
        >>> print(unfolded_rep.shape)  # Output: [2, 100, 5, 256]

    Note:
        This method uses PyTorch's `unfold` operation to create the context 
        channels from the input tensor.
        """
        bsz, freq, _ = tf_rep.shape
        if chunk_size == 0:
            return tf_rep

        est_unfolded = torch.nn.functional.unfold(
            torch.nn.functional.pad(
                tf_rep, (chunk_size, chunk_size), mode="constant"
            ).unsqueeze(-1),
            kernel_size=(2 * chunk_size + 1, 1),
            padding=(0, 0),
            stride=(1, 1),
        )
        n_chunks = est_unfolded.shape[-1]
        est_unfolded = est_unfolded.reshape(bsz, freq, 2 * chunk_size + 1, n_chunks)
        est_unfolded = est_unfolded.transpose(1, 2)
        return est_unfolded

    @staticmethod
    def mfmcwf(mixture, estimate, n_chunks, tik_eps):
        """
        multi-frame multi-channel wiener filter.

    This method applies the multi-frame multi-channel Wiener filter (mfMCWF) 
    to enhance the target source estimate from a multi-channel STFT complex 
    mixture. It leverages context from both past and future frames for improved 
    estimation accuracy.

    Args:
        mixture (torch.Tensor): multi-channel STFT complex mixture tensor,
            of shape [B, T, C, F] where B is batch size, T is the number of 
            frames, C is the number of microphones, and F is the number of 
            frequency bins.
        estimate (torch.Tensor): monaural STFT complex estimate of the target 
            source, shaped as [B, T, F] where B is batch size, T is the 
            number of frames, and F is the number of frequency bins.
        n_chunks (int): number of past and future frames to consider for 
            mfMCWF computation. If set to 0, the method defaults to standard 
            multi-channel Wiener filtering (MCWF).
        tik_eps (float): diagonal loading parameter for matrix inversion in 
            MCWF computation to ensure numerical stability.

    Returns:
        beamformed (torch.Tensor): monaural STFT complex estimate of the 
            target source after applying mfMCWF, shaped as [B, T, F].

    Examples:
        >>> mixture = torch.rand(2, 100, 3, 256)  # Example multi-channel input
        >>> estimate = torch.rand(2, 100, 256)    # Example estimate for target
        >>> n_chunks = 3
        >>> tik_eps = 1e-8
        >>> result = iNeuBe.mfmcwf(mixture, estimate, n_chunks, tik_eps)
        >>> print(result.shape)  # Should output: torch.Size([2, 100, 256])

    Note:
        This method assumes the input tensors are on the same device and 
        have been properly prepared prior to calling this function. Ensure 
        that the dimensions of the input tensors conform to the expected 
        shapes to avoid runtime errors.
        """
        mixture = mixture.permute(0, 2, 3, 1)
        estimate = estimate.transpose(-1, -2)

        bsz, mics, _, frames = mixture.shape

        mix_unfolded = iNeuBe.unfold(
            mixture.reshape(bsz * mics, -1, frames), n_chunks
        ).reshape(bsz, mics * (2 * n_chunks + 1), -1, frames)

        mix_unfolded = to_double(mix_unfolded)
        estimate1 = to_double(estimate)
        zeta = torch.einsum("bmft, bft->bmf", mix_unfolded, estimate1.conj())
        scm_mix = torch.einsum("bmft, bnft->bmnf", mix_unfolded, mix_unfolded.conj())
        inv_scm_mix = torch.inverse(
            tik_reg(scm_mix.permute(0, 3, 1, 2), tik_eps)
        ).permute(0, 2, 3, 1)
        bf_vector = torch.einsum("bmnf, bnf->bmf", inv_scm_mix, zeta)

        beamformed = torch.einsum("...mf,...mft->...ft", bf_vector.conj(), mix_unfolded)
        beamformed = beamformed.to(mixture)

        return beamformed.transpose(-1, -2)

    @staticmethod
    def pad2(input_tensor, target_len):
        """
        pad2(input_tensor, target_len)

    Pads the input tensor to the specified target length along the last dimension.

    This function is particularly useful when the output tensor from a neural 
    network needs to be adjusted to match a specified length, which is often 
    required for further processing or for consistency in input shapes across 
    different batches.

    Args:
        input_tensor (torch.Tensor): The input tensor to be padded. It should 
            have at least one dimension, and the last dimension is the one 
            that will be padded.
        target_len (int): The desired length of the last dimension after 
            padding. If the input tensor is already longer than this length, 
            no padding will be applied.

    Returns:
        torch.Tensor: The padded tensor, which has the same shape as the 
        input tensor, except the last dimension will be equal to `target_len`.

    Examples:
        >>> import torch
        >>> input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> padded_tensor = pad2(input_tensor, 5)
        >>> print(padded_tensor)
        tensor([[1, 2, 3, 0, 0],
                [4, 5, 6, 0, 0]])
        """
        input_tensor = torch.nn.functional.pad(
            input_tensor, (0, target_len - input_tensor.shape[-1])
        )
        return input_tensor

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """
        Forward pass for the iNeuBe model, processing multi-channel audio input.

    This method takes a batched multi-channel audio tensor and computes the 
    enhanced audio output through the model's forward network.

    Args:
        input (Union[torch.Tensor, ComplexTensor]): Batched multi-channel audio 
            tensor with C audio channels and T samples of shape [B, T, C].
        ilens (torch.Tensor): Input lengths of shape [Batch].
        additional (Optional[Dict]): Other data, currently unused in this model.

    Returns:
        Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, 
        OrderedDict]: A tuple containing:
            - enhanced (List[Union[torch.Tensor, ComplexTensor]]): 
                A list of length n_spk, containing mono audio tensors with 
                T samples for each speaker.
            - ilens (torch.Tensor): The input lengths as a tensor of shape (B,).
            - additional (OrderedDict): Other data, currently unused in this 
                model; returned for compatibility.

    Raises:
        NotImplementedError: If an unsupported output type is specified.

    Examples:
        >>> model = iNeuBe(n_spk=2)
        >>> input_tensor = torch.randn(4, 16000, 2)  # 4 samples, 16000 time steps, 2 channels
        >>> ilens = torch.tensor([16000, 16000, 16000, 16000])  # Input lengths
        >>> outputs, lengths, _ = model.forward(input_tensor, ilens)

    Note:
        The model is designed to enhance speech signals from multi-channel 
        audio input, utilizing various internal submodules for processing. 
        Make sure to provide the correct input shapes and types to avoid 
        runtime errors.
        """
        # B, T, C
        bsz, mixture_len, mics = input.shape

        mix_stft = self.enc(input, ilens)[0]
        # B, T, C, F
        est_dnn1 = self.dnn1(mix_stft)
        if self.freeze_dnn1:
            est_dnn1 = est_dnn1.detach()
        _, _, frames, freq = est_dnn1.shape
        output_dnn1 = self.dec(
            est_dnn1.reshape(bsz * self.num_spk, frames, freq), ilens
        )[0]
        output_dnn1 = self.pad2(output_dnn1.reshape(bsz, self.num_spk, -1), mixture_len)
        output_dnn1 = [output_dnn1[:, src] for src in range(output_dnn1.shape[1])]
        others = OrderedDict()
        if self.output_from == "dnn1":
            return output_dnn1, ilens, others
        elif self.output_from in ["mfmcwf", "dnn2"]:
            others["dnn1"] = output_dnn1
            est_mfmcwf = iNeuBe.mfmcwf(
                mix_stft,
                est_dnn1.reshape(bsz * self.n_spk, frames, freq),
                self.n_chunks,
                self.tik_eps,
            ).reshape(bsz, self.n_spk, frames, freq)
            output_mfmcwf = self.dec(
                est_mfmcwf.reshape(bsz * self.num_spk, frames, freq), ilens
            )[0]
            output_mfmcwf = self.pad2(
                output_mfmcwf.reshape(bsz, self.num_spk, -1), mixture_len
            )
            if self.output_from == "mfmcwf":
                return (
                    [output_mfmcwf[:, src] for src in range(output_mfmcwf.shape[1])],
                    ilens,
                    others,
                )
            elif self.output_from == "dnn2":
                others["dnn1"] = output_dnn1
                others["beam"] = output_mfmcwf
                est_dnn2 = self.dnn2(
                    torch.cat(
                        (
                            mix_stft.repeat(self.num_spk, 1, 1, 1),
                            est_dnn1.reshape(
                                bsz * self.num_spk, frames, freq
                            ).unsqueeze(2),
                            est_mfmcwf.reshape(
                                bsz * self.num_spk, frames, freq
                            ).unsqueeze(2),
                        ),
                        2,
                    )
                )

                output_dnn2 = self.dec(est_dnn2[:, 0], ilens)[0]
                output_dnn2 = self.pad2(
                    output_dnn2.reshape(bsz, self.num_spk, -1), mixture_len
                )
                return (
                    [output_dnn2[:, src] for src in range(output_dnn2.shape[1])],
                    ilens,
                    others,
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    @property
    def num_spk(self):
        return self.n_spk
