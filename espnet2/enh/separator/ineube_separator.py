import torch
import asteroid_filterbanks.transforms as af_transforms
from collections import OrderedDict
from typing import List
from typing import Tuple
from typing import Union
from torch_complex.tensor import ComplexTensor
from espnet2.enh.separator.abs_separator import AbsSeparator
from asteroid.dsp.beamforming import condition_scm
from asteroid_filterbanks import make_enc_dec
from asteroid.utils.torch_utils import pad_x_to_y
from espnet2.enh.layers.tcndenseunet import TCNDenseUNet


class iNeuBe(AbsSeparator):
    def __init__(self, n_spk=1,
                 n_fft=512,
                 stride=128,
                 window="hanning",
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
        super().__init__()
        self.n_spk = n_spk
        assert self.n_spk == 1, "not tested"

        if window == "hanning":
            window = torch.hann_window(n_fft)
        elif window == "hamming":
            window = torch.hamming_window(n_fft)
        elif window is None:
            pass
        else:
            raise NotImplementedError
        fft_c_channels = n_fft // 2 + 1
        self.enc, self.dec = make_enc_dec("torch_stft", n_fft, n_fft, stride, window=window)
        self.dnn1 = TCNDenseUNet(n_spk, fft_c_channels,
        mic_channels,
        hid_chans,
        hid_chans_dense,
        ksz_dense,
        ksz_tcn,
        tcn_repeats,
        tcn_blocks,
        tcn_channels,
        activation=activation)


        self.dnn2 = TCNDenseUNet(n_spk, fft_c_channels,
        mic_channels + 2,
        hid_chans,
        hid_chans_dense,
        ksz_dense,
        ksz_tcn,
        tcn_repeats,
        tcn_blocks,
        tcn_channels,
        activation=activation)

    def unfold(self, tf_rep, chunk_size):
        bsz, freq, _ = tf_rep.shape

        est_unfolded = torch.nn.functional.unfold(
            torch.nn.functional.pad(tf_rep, (chunk_size, chunk_size), mode="constant").unsqueeze(-1),
            kernel_size=(2 * chunk_size + 1, 1),
            padding=(0, 0),
            stride=(1, 1),
        )
        n_chunks = est_unfolded.shape[-1]
        est_unfolded = est_unfolded.reshape(bsz, freq, 2 * chunk_size + 1, n_chunks)
        est_unfolded = est_unfolded.transpose(1, 2)
        return est_unfolded

    def mfmcwf(self, mixture, estimate, n_chunks):
        bsz, mics, _, frames = mixture.shape
        mix_unfolded = self.unfold(mixture.reshape(bsz * mics, -1, frames), n_chunks).reshape(bsz, mics * (2*n_chunks +1), -1, frames)

        mix_unfolded = af_transforms.to_torch_complex(mix_unfolded.double())
        estimate1 = af_transforms.to_torch_complex(estimate.double())
        zeta = torch.einsum("bmft, bft->bmf", mix_unfolded, estimate1.conj())
        scm_mix = torch.einsum("bmft, bnft->bmnf", mix_unfolded, mix_unfolded.conj())
        inv_scm_mix = torch.inverse(condition_scm(scm_mix.permute(0, 3, 1, 2), 1e-4)).permute(0, 2, 3, 1)
        bf_vector = torch.einsum("bmnf, bnf->bmf", inv_scm_mix, zeta)

        beamformed = torch.einsum("...mf,...mft->...ft", bf_vector.conj(), mix_unfolded)
        beamformed = af_transforms.from_torch_complex(beamformed).float()

        return beamformed

    def forward(
            self, input: Union[torch.Tensor, ComplexTensor], ilens: torch.Tensor, output_from="dnn1", n_chunks=3,
            freeze_dnn1 = False
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:

        mix_stft = self.enc(input)
        others = OrderedDict()
        est_dnn1 = self.dnn1(mix_stft)
        if freeze_dnn1:
            est_dnn1 = est_dnn1.detach()

        if output_from == "dnn1":
            return [pad_x_to_y(self.dec(est_dnn1), input)], ilens, others
        elif output_from == "beam":
            others["dnn1"] = pad_x_to_y(self.dec(est_dnn1), input)
            est_mfmcwf = self.mfmcwf(mix_stft, est_dnn1, n_chunks)
            return [pad_x_to_y(self.dec(est_mfmcwf), input)], ilens, others
        elif output_from == "dnn2":
            others["dnn1"] = pad_x_to_y(self.dec(est_dnn1), input)
            est_mfmcwf = self.mfmcwf(mix_stft, est_dnn1, n_chunks)
            others["beam"] = pad_x_to_y(self.dec(est_mfmcwf), input)
            est_dnn2 = self.dnn2(torch.cat((mix_stft, est_dnn1.unsqueeze(1), est_mfmcwf.unsqueeze(1)), 1))
            est_dnn2 = pad_x_to_y(self.dec(est_dnn2), input)
            return [est_dnn2], ilens, others
        else:
            raise NotImplementedError

    @property
    def num_spk(self):
        return self.n_spk


