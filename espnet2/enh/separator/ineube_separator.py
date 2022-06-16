import torch
from collections import OrderedDict
from typing import List
from typing import Tuple
from typing import Union
from torch_complex.tensor import ComplexTensor
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.enh.layers.tcndenseunet import TCNDenseUNet
from espnet2.enh.layers.beamformer import tik_reg, to_double
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.decoder.stft_decoder import STFTDecoder

class iNeuBe(AbsSeparator):
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
        input_from="dnn1",
        n_chunks=3,
        freeze_dnn1=False,
        tik_eps=1e-8
    ):
        super().__init__()
        self.n_spk = n_spk
        self.input_from = input_from
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
            n_spk,
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
        bsz, freq, _ = tf_rep.shape

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

        return beamformed

    @staticmethod
    def pad2(input_tensor, target_len):
        input_tensor = torch.nn.functional.pad(input_tensor, (0, 0, 0, target_len - input_tensor.shape[0]))
        return input_tensor

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor, **kwargs) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        # B, T, C
        bsz, mixture_len, mics = input.shape

        mix_stft = self.enc(input)
        # B, T, C, F
        mix_stft = mix_stft.permute(0, 2, 3, 1)
        est_dnn1 = self.dnn1(mix_stft)
        if self.freeze_dnn1:
            est_dnn1 = est_dnn1.detach()

        output_dnn1 = self.pad2(self.dec(est_dnn1), mixture_len)
        output_dnn1 = [output_dnn1[..., src] for src in range(output_dnn1.shape[-1])]
        others = OrderedDict()
        if self.output_from == "dnn1":
            return output_dnn1, ilens, others
        elif self.output_from in ["beam", "dnn2"]:
            others["dnn1"] = output_dnn1
            est_mfmcwf = iNeuBe.mfmcwf(mix_stft, est_dnn1, self.n_chunks, self.tik_eps)
            output_mfmcwf = self.pad2(self.dec(est_mfmcwf), mixture_len)
            if self.output_from == "mfmcwf":
                return [output_mfmcwf[..., src] for src in range(output_mfmcwf.shape[-1])], ilens, others
            elif self.output_from == "dnn2":
                others["dnn1"] = output_dnn1
                others["beam"] = output_mfmcwf
                est_dnn2 = self.dnn2(
                    torch.cat((mix_stft, est_dnn1.unsqueeze(1), est_mfmcwf.unsqueeze(1)), 1)
                )
                est_dnn2 = self.pad2(self.dec(est_dnn2), mixture_len)
                return [est_dnn2[..., src] for src in est_dnn2.shape[-1]], ilens, others
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    @property
    def num_spk(self):
        return self.n_spk
