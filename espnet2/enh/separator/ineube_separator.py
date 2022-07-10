from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.decoder.stft_decoder import STFTDecoder
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.layers.beamformer import tik_reg, to_double
from espnet2.enh.layers.tcndenseunet import TCNDenseUNet
from espnet2.enh.separator.abs_separator import AbsSeparator


class iNeuBe(AbsSeparator):
    """iNeuBe, iterative neural/beamforming enhancement

    Reference:
    Lu, Y. J., Cornell, S., Chang, X., Zhang, W., Li, C., Ni, Z., ... & Watanabe, S.
    Towards Low-Distortion Multi-Channel Speech Enhancement:
    The ESPNET-Se Submission to the L3DAS22 Challenge. ICASSP 2022 p. 9201-9205.

    Args:
        n_spk: number of output sources/speakers.
        n_fft: stft window size.
        stride: stft stride.
        window: stft window type choose between 'hamming', 'hanning' or None.
        mic_channels: number of microphones channels
            (only fixed-array geometry supported).
        hid_chans: number of channels in the subsampling/upsampling conv layers.
        hid_chans_dense: number of channels in the densenet layers
            (reduce this to reduce VRAM requirements).
        ksz_dense: kernel size in the densenet layers thorough iNeuBe.
        ksz_tcn: kernel size in the TCN submodule.
        tcn_repeats: number of repetitions of blocks in the TCN submodule.
        tcn_blocks: number of blocks in the TCN submodule.
        tcn_channels: number of channels in the TCN submodule.
        activation: activation function to use in the whole iNeuBe model,
            you can use any torch supported activation e.g. 'relu' or 'elu'.
        output_from: output the estimate from 'dnn1', 'mfmcwf' or 'dnn2'.
        n_chunks: number of future and past frames to consider for mfMCWF computation.
        freeze_dnn1: whether or not freezing dnn1 parameters during training of dnn2.
        tik_eps: diagonal loading in the mfMCWF computation.
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
        """unfolding STFT representation to add context in the mics channel.

        Args:
            mixture (torch.Tensor): 3D tensor (monaural complex STFT)
                of shape [B, T, F] batch, frames, microphones, frequencies.
            n_chunks (int): number of past and future to consider.

        Returns:
            est_unfolded (torch.Tensor): complex 3D tensor STFT with context channel.
                shape now is [B, T, C, F] batch, frames, context, frequencies.
                Basically same shape as a multi-channel STFT with C microphones.

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
        """multi-frame multi-channel wiener filter.

        Args:
            mixture (torch.Tensor): multi-channel STFT complex mixture tensor,
                of shape [B, T, C, F] batch, frames, microphones, frequencies.
            estimate (torch.Tensor): monaural STFT complex estimate
                of target source [B, T, F] batch, frames, frequencies.
            n_chunks (int): number of past and future mfMCWF frames.
                If 0 then standard MCWF.
            tik_eps (float): diagonal loading for matrix inversion in MCWF computation.

        Returns:
            beamformed (torch.Tensor): monaural STFT complex estimate
                of target source after MFMCWF [B, T, F] batch, frames, frequencies.
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
        """Forward.

        Args:
            input (torch.Tensor): batched multi-channel audio tensor with
                    C audio channels and T samples [B, T, C]
            ilens (torch.Tensor): input lengths [Batch]
            additional (Dict or None): other data, currently unused in this model.

        Returns:
            enhanced (List[Union(torch.Tensor, ComplexTensor)]):
                    [(B, T), ...] list of len n_spk
                    of mono audio tensors with T samples.
            ilens (torch.Tensor): (B,)
                others predicted data, e.g. masks: OrderedDict[
                    'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                    'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                    ...
                    'mask_spkn': torch.Tensor(Batch, Frames, Freq),
                ]
            additional (Dict or None): other data, currently unused in this model,
                    we return it also in output.
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
